"""
database.py
-----------
SQLAlchemy-backed database layer for the receptionist application.

* Uses `office.db` (SQLite) stored alongside this module.
* Provides every helper that was previously in database1.py, rewritten to
  use SQLAlchemy ORM sessions instead of raw sqlite3 connections.
* Public API is fully backward-compatible with both database.py v1 and
  database1.py — callers need not change.
"""

from __future__ import annotations

import difflib
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from sqlalchemy import and_, create_engine, or_
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
    Conversation,
    Employee,
    Meeting,
    ReceptionLog,
    Settings,
    Visitor,
)

# ──────────────────────────────────────────────────────────────────────────────
# Engine / session setup
# ──────────────────────────────────────────────────────────────────────────────

_db_path = Path(__file__).resolve().parent / "office.db"
DATABASE_URL = f"sqlite:///{_db_path.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ──────────────────────────────────────────────────────────────────────────────

ALL_SLOTS_1H: List[str] = [
    "09:00",
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "16:00",
    "17:00",
]


def init_db() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", _db_path)


# ──────────────────────────────────────────────────────────────────────────────
# Internal normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_date(meeting_date: str) -> Optional[str]:
    """Normalise a human-readable date string to ``YYYY-MM-DD``."""
    if not meeting_date:
        return None
    s = str(meeting_date).strip().lower()

    # Already YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except ValueError:
        pass

    today = datetime.now().date()

    if s in {"today", "now"}:
        return today.strftime("%Y-%m-%d")
    if s == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.match(r"^in\s+(\d+)\s+days?$", s)
    if m:
        return (today + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

    m = re.match(
        r"^(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s
    )
    if m:
        weekday_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        target = weekday_map[m.group(2)]
        days_ahead = (target - today.weekday() + 7) % 7 or 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return None


def _normalize_time(meeting_time: str) -> Optional[str]:
    """Normalise a human-readable time string to ``HH:MM`` (24-hour)."""
    if not meeting_time:
        return None

    s = (
        str(meeting_time)
        .strip()
        .lower()
        .replace("p.m.", "pm")
        .replace("a.m.", "am")
        .replace(".", ":")
        .replace(" ", "")
    )

    if re.match(r"^\d{2}:\d{2}$", s):
        return s

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            return None
        if mer == "pm" and hour != 12:
            hour += 12
        if mer == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?$", s)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return f"{hour:02d}:{minute:02d}"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Reception log
# ──────────────────────────────────────────────────────────────────────────────


def log_reception_entry(
    person_name: str,
    person_type: str,
    notes: str = "",
    linked_visitor_id: Optional[int] = None,
    linked_employee_id: Optional[int] = None,
) -> int:
    """Add an entry to the reception log. Returns the new log entry id."""
    session = SessionLocal()
    try:
        entry = ReceptionLog(
            person_name=person_name,
            person_type=person_type,
            linked_visitor_id=linked_visitor_id,
            linked_employee_id=linked_employee_id,
            check_in_time=datetime.utcnow().isoformat(),
            notes=notes,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry.id
    except Exception as exc:
        logger.error("log_reception_entry failed: %s", exc)
        session.rollback()
        return -1
    finally:
        session.close()


def log_reception_checkout(log_id: int) -> None:
    """Mark a reception log entry as checked out."""
    session = SessionLocal()
    try:
        entry = session.query(ReceptionLog).filter(ReceptionLog.id == log_id).first()
        if entry:
            entry.check_out_time = datetime.utcnow().isoformat()
            session.commit()
    except Exception as exc:
        logger.error("log_reception_checkout failed: %s", exc)
        session.rollback()
    finally:
        session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Conversations
# ──────────────────────────────────────────────────────────────────────────────


def save_conversation(user_text: str, ai_response: str) -> None:
    """Persist a single AI conversation turn."""
    session = SessionLocal()
    try:
        convo = Conversation(
            user_text=user_text,
            ai_response=ai_response,
            timestamp=datetime.utcnow().isoformat(),
        )
        session.add(convo)
        session.commit()
    except Exception as exc:
        logger.error("save_conversation failed: %s", exc)
        session.rollback()
    finally:
        session.close()


def get_all_conversations() -> List[Conversation]:
    """Return all conversation records."""
    session = SessionLocal()
    try:
        return session.query(Conversation).all()
    except Exception as exc:
        logger.error("get_all_conversations failed: %s", exc)
        return []
    finally:
        session.close()


def get_recent_conversations(limit: int = 5) -> List[Conversation]:
    """Return the *limit* most recent conversations, ordered oldest → newest."""
    session = SessionLocal()
    try:
        rows = (
            session.query(Conversation)
            .order_by(Conversation.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))
    except Exception as exc:
        logger.error("get_recent_conversations failed: %s", exc)
        return []
    finally:
        session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────


def set_setting(key: str, value: str) -> None:
    """Upsert an application setting."""
    session = SessionLocal()
    try:
        existing = session.query(Settings).filter(Settings.key == key).first()
        if existing:
            existing.value = value
        else:
            session.add(Settings(key=key, value=value))
        session.commit()
    except Exception as exc:
        logger.error("set_setting failed: %s", exc)
        session.rollback()
    finally:
        session.close()


def get_setting(key: str) -> str:
    """Retrieve an application setting value; returns ``""`` if not found."""
    session = SessionLocal()
    try:
        row = session.query(Settings).filter(Settings.key == key).first()
        return row.value if row else ""
    except Exception as exc:
        logger.error("get_setting failed: %s", exc)
        return ""
    finally:
        session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Visitors
# ──────────────────────────────────────────────────────────────────────────────


def _generate_badge_id(visitor_id: int) -> str:
    year = datetime.utcnow().year
    return f"VIS-{year}-{visitor_id:04d}"


def add_visitor(name: str, meeting_with: str, purpose: str) -> tuple[str, int]:
    """
    Insert a visitor record and return ``(badge_id, visitor_id)``.
    Caller is responsible for logging to reception_log separately.
    """
    session = SessionLocal()
    try:
        check_in_time = datetime.utcnow().isoformat()
        visitor = Visitor(
            name=name,
            meeting_with=meeting_with,
            purpose=purpose,
            check_in_time=check_in_time,
            checkin_time=datetime.utcnow(),
        )
        session.add(visitor)
        session.flush()  # assigns visitor.id before commit

        badge_id = _generate_badge_id(visitor.id)
        visitor.badge_id = badge_id
        session.commit()
        session.refresh(visitor)
        return badge_id, visitor.id
    except Exception as exc:
        logger.error("add_visitor failed: %s", exc)
        session.rollback()
        return "", -1
    finally:
        session.close()


def checkout_visitor(badge_id: str) -> None:
    """Set check_out_time for the visitor with the given badge_id."""
    session = SessionLocal()
    try:
        visitor = session.query(Visitor).filter(Visitor.badge_id == badge_id).first()
        if visitor:
            visitor.check_out_time = datetime.utcnow().isoformat()
            session.commit()
    except Exception as exc:
        logger.error("checkout_visitor failed: %s", exc)
        session.rollback()
    finally:
        session.close()


def get_visitor_by_name(name: str) -> Optional[Visitor]:
    """Return the most recent visitor record matching *name* (case-insensitive)."""
    session = SessionLocal()
    try:
        return (
            session.query(Visitor)
            .filter(Visitor.name.ilike(name))
            .order_by(Visitor.id.desc())
            .first()
        )
    except Exception as exc:
        logger.error("get_visitor_by_name failed: %s", exc)
        return None
    finally:
        session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Employees — internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_employee(search_term: str) -> Optional[Employee]:
    """
    Lightweight internal lookup: find an employee by name or role fragment
    using SQLAlchemy ILIKE.  Used by the scheduling helpers.
    """
    if not search_term or not str(search_term).strip():
        return None
    term = str(search_term).strip()
    session = SessionLocal()
    try:
        return (
            session.query(Employee)
            .filter(
                or_(
                    Employee.name.ilike(f"%{term}%"),
                    Employee.role.ilike(f"%{term}%"),
                )
            )
            .first()
        )
    except Exception as exc:
        logger.error("_resolve_employee failed for %r: %s", search_term, exc)
        return None
    finally:
        session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Employees — public API
# ──────────────────────────────────────────────────────────────────────────────


def get_employee_by_name_or_role(search_term: str) -> Optional[Employee]:
    """Return an Employee by name or role fragment (scheduling helper)."""
    return _resolve_employee(search_term)


def get_employee_by_name(name: str) -> Optional[Employee]:
    """
    Strict directory lookup.  Tries, in order:
    1. Exact full-name match
    2. Substring / prefix match
    3. difflib fuzzy match (cutoff 0.6)
    Only considers ``is_public = 1`` employees.
    """
    if not name:
        return None

    name_clean = name.lower().strip()
    session = SessionLocal()
    try:
        public = session.query(Employee).filter(Employee.is_public == 1)

        # 1. Exact match
        row = public.filter(Employee.name.ilike(name_clean)).first()
        if row:
            return row

        # 2. Substring match
        row = public.filter(
            or_(
                Employee.name.ilike(f"%{name_clean}%"),
                Employee.name.ilike(f"{name_clean}%"),
            )
        ).first()
        if row:
            return row

        # 3. difflib fuzzy match
        all_employees = public.all()
        emp_names = [e.name for e in all_employees]
        matches = difflib.get_close_matches(name_clean, emp_names, n=1, cutoff=0.6)
        if matches:
            for emp in all_employees:
                if emp.name == matches[0]:
                    return emp

        return None
    except Exception as exc:
        logger.error("get_employee_by_name failed: %s", exc)
        return None
    finally:
        session.close()


def get_employee_by_name_and_department(
    name: str, department: str
) -> Optional[Employee]:
    """
    Verify an employee by name fragment AND department/role fragment.
    Used during identity confirmation to avoid false positives.
    """
    if not name or not department:
        return None

    name_clean = name.lower().strip()
    dept_clean = department.lower().strip()
    session = SessionLocal()
    try:
        return (
            session.query(Employee)
            .filter(
                Employee.name.ilike(f"%{name_clean}%"),
                or_(
                    Employee.department.ilike(f"%{dept_clean}%"),
                    Employee.role.ilike(f"%{dept_clean}%"),
                ),
                Employee.is_public == 1,
            )
            .first()
        )
    except Exception as exc:
        logger.error("get_employee_by_name_and_department failed: %s", exc)
        return None
    finally:
        session.close()


def get_similar_employee(name: str, cutoff: float = 0.55) -> Optional[Employee]:
    """
    Check if *name* resembles any employee name.
    Used to decide whether to ask 'are you a visitor or employee?'.

    Checks for exact first-/last-name matches first, then falls back to
    difflib similarity.  Lower cutoff than ``get_employee_by_name`` so it
    catches partial matches like 'Rahul' → 'Rahul Sharma'.
    """
    if not name:
        return None

    name_lower = name.lower().strip()
    session = SessionLocal()
    try:
        all_employees = session.query(Employee).filter(Employee.is_public == 1).all()
        if not all_employees:
            return None

        # First-name / last-name exact part match
        for emp in all_employees:
            if name_lower in emp.name.lower().split():
                return emp

        # difflib fallback
        emp_names = [e.name for e in all_employees]
        matches = difflib.get_close_matches(name, emp_names, n=1, cutoff=cutoff)
        if matches:
            for emp in all_employees:
                if emp.name == matches[0]:
                    return emp

        return None
    except Exception as exc:
        logger.error("get_similar_employee failed: %s", exc)
        return None
    finally:
        session.close()


def get_hr(name: str = "HR") -> Optional[Employee]:
    """
    Return the HR Manager (preferred) or any HR employee.
    ``name`` parameter is accepted for API compatibility but not used.
    """
    session = SessionLocal()
    try:
        # Prefer HR Manager
        emp = (
            session.query(Employee)
            .filter(
                Employee.department.ilike("hr"),
                Employee.role.ilike("%manager%"),
                Employee.is_public == 1,
            )
            .first()
        )
        if not emp:
            emp = (
                session.query(Employee)
                .filter(
                    Employee.department.ilike("hr"),
                    Employee.is_public == 1,
                )
                .first()
            )
        return emp
    except Exception as exc:
        logger.error("get_hr failed: %s", exc)
        return None
    finally:
        session.close()


def get_department_manager(department: str) -> Optional[Employee]:
    """Return the manager for *department*, with a fuzzy fallback."""
    session = SessionLocal()
    try:
        emp = (
            session.query(Employee)
            .filter(
                Employee.department.ilike(department),
                Employee.role.ilike("%manager%"),
                Employee.is_public == 1,
            )
            .first()
        )
        if not emp:
            emp = (
                session.query(Employee)
                .filter(
                    Employee.department.ilike(f"%{department}%"),
                    Employee.role.ilike("%manager%"),
                    Employee.is_public == 1,
                )
                .first()
            )
        return emp
    except Exception as exc:
        logger.error("get_department_manager failed: %s", exc)
        return None
    finally:
        session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Meetings
# ──────────────────────────────────────────────────────────────────────────────


def get_employee_meetings(employee_name: str, meeting_date: str) -> List[Meeting]:
    """
    Return all scheduled meetings for *employee_name* on *meeting_date*
    (``YYYY-MM-DD``), ordered by meeting_time.
    """
    session = SessionLocal()
    try:
        return (
            session.query(Meeting)
            .filter(
                Meeting.employee_name.ilike(employee_name),
                Meeting.meeting_date == meeting_date,
                Meeting.status == "scheduled",
            )
            .order_by(Meeting.meeting_time)
            .all()
        )
    except Exception as exc:
        logger.error("get_employee_meetings failed: %s", exc)
        return []
    finally:
        session.close()


def get_available_slots(employee_name: str, meeting_date: str) -> List[str]:
    """
    Return available 1-hour slot start times (``HH:MM``) between 09:00–17:00
    for *employee_name* on *meeting_date*.

    Checks both the rich ``meeting_date / meeting_time`` columns (database1
    style) and the legacy ``scheduled_time`` DateTime column (old database.py
    style) so the function works regardless of how meetings were created.
    """
    normalized_date = _normalize_date(meeting_date)
    if not normalized_date:
        return []

    session = SessionLocal()
    try:
        booked_times: set[str] = set()

        # Rich-column meetings (database1 style)
        rich_booked = (
            session.query(Meeting)
            .filter(
                Meeting.employee_name.ilike(employee_name),
                Meeting.meeting_date == normalized_date,
                Meeting.status == "scheduled",
            )
            .all()
        )
        for m in rich_booked:
            if m.meeting_time:
                booked_times.add(str(m.meeting_time)[:5])

        # Legacy DateTime meetings (old database.py style)
        start_dt = datetime.strptime(normalized_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=1)
        legacy_booked = (
            session.query(Meeting)
            .filter(
                Meeting.employee_name.ilike(employee_name),
                Meeting.scheduled_time >= start_dt,
                Meeting.scheduled_time < end_dt,
                Meeting.status != "cancelled",
            )
            .all()
        )
        for m in legacy_booked:
            if m.scheduled_time:
                booked_times.add(m.scheduled_time.strftime("%H:%M"))

        return [s for s in ALL_SLOTS_1H if s not in booked_times]
    except Exception as exc:
        logger.error("get_available_slots failed: %s", exc)
        return []
    finally:
        session.close()


def schedule_meeting(
    organizer_name: str,
    organizer_type: str,
    employee_name: str,
    meeting_date: str,
    meeting_time: str,
    purpose: str = "",
) -> Optional[int]:
    """
    Persist a meeting and return its id.

    Resolves the employee record to pull the correct email.
    Normalises date/time inputs automatically.
    Returns the existing meeting id if the slot is already taken.
    """
    normalized_date = _normalize_date(meeting_date)
    normalized_time = _normalize_time(meeting_time)
    if not normalized_date or not normalized_time:
        return None

    emp = _resolve_employee(employee_name)
    canonical_name = emp.name if emp else employee_name
    employee_email = emp.email if emp else ""

    session = SessionLocal()
    try:
        # Deduplicate by (employee, date, time)
        existing = (
            session.query(Meeting)
            .filter(
                Meeting.employee_name.ilike(canonical_name),
                Meeting.meeting_date == normalized_date,
                Meeting.meeting_time == normalized_time,
                Meeting.status != "cancelled",
            )
            .first()
        )
        if existing:
            return existing.id

        # Build a DateTime for the legacy scheduled_time column as well
        scheduled_dt = datetime.strptime(
            f"{normalized_date} {normalized_time}", "%Y-%m-%d %H:%M"
        )

        new_meeting = Meeting(
            organizer_name=organizer_name or "Visitor",
            organizer_type=organizer_type or "visitor",
            visitor_name=organizer_name or "Visitor",  # legacy compat
            employee_name=canonical_name,
            employee_email=employee_email,
            meeting_date=normalized_date,
            meeting_time=normalized_time,
            scheduled_time=scheduled_dt,  # legacy compat
            purpose=purpose,
            status="scheduled",
            created_at=datetime.now().isoformat(),
        )
        session.add(new_meeting)
        session.commit()
        session.refresh(new_meeting)
        logger.info(
            "Meeting scheduled: %s with %s at %s %s",
            new_meeting.organizer_name,
            canonical_name,
            normalized_date,
            normalized_time,
        )
        return new_meeting.id
    except Exception as exc:
        logger.error("schedule_meeting failed: %s", exc)
        session.rollback()
        return None
    finally:
        session.close()


def cancel_meeting(meeting_id: int) -> None:
    """Set a meeting's status to ``'cancelled'``."""
    session = SessionLocal()
    try:
        meeting = session.query(Meeting).filter(Meeting.id == meeting_id).first()
        if meeting:
            meeting.status = "cancelled"
            session.commit()
    except Exception as exc:
        logger.error("cancel_meeting failed: %s", exc)
        session.rollback()
    finally:
        session.close()
