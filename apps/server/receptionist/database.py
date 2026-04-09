"""
database.py
-----------
Normalized SQLAlchemy-backed database layer for the receptionist application.

* Uses `office.db` (SQLite) stored alongside this module.
* Utilizes relational models (Foreign Keys, strict Entities vs Events).
* Includes robust date/time natural language parsing helpers.
"""

from __future__ import annotations

import difflib
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from sqlalchemy import and_, create_engine, or_, cast, Date
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
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
    "09:00", "10:00", "11:00", "12:00", "13:00", 
    "14:00", "15:00", "16:00", "17:00",
]

def init_db() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", _db_path)


# ──────────────────────────────────────────────────────────────────────────────
# Internal Date/Time Normalisation Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_date(meeting_date: str) -> Optional[str]:
    if not meeting_date: return None
    s = str(meeting_date).strip().lower()

    try: 
        return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except ValueError: 
        pass

    today = datetime.now().date()
    if s in {"today", "now"}: return today.strftime("%Y-%m-%d")
    if s == "tomorrow": return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.match(r"^in\s+(\d+)\s+days?$", s)
    if m: return (today + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

    m = re.match(r"^(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s)
    if m:
        weekday_map = {"monday":0, "tuesday":1, "wednesday":2, "thursday":3, "friday":4, "saturday":5, "sunday":6}
        target = weekday_map[m.group(2)]
        days_ahead = (target - today.weekday() + 7) % 7 or 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return None

def _normalize_time(meeting_time: str) -> Optional[str]:
    if not meeting_time: return None
    s = str(meeting_time).strip().lower().replace("p.m.", "pm").replace("a.m.", "am").replace(".", ":").replace(" ", "")

    if re.match(r"^\d{2}:\d{2}$", s): return s

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if not (1 <= hour <= 12 and 0 <= minute <= 59): return None
        if mer == "pm" and hour != 12: hour += 12
        if mer == "am" and hour == 12: hour = 0
        return f"{hour:02d}:{minute:02d}"

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?$", s)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if not (0 <= hour <= 23 and 0 <= minute <= 59): return None
        return f"{hour:02d}:{minute:02d}"

    return None

def _get_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    n_date = _normalize_date(date_str)
    n_time = _normalize_time(time_str)
    if not n_date or not n_time:
        return None
    return datetime.strptime(f"{n_date} {n_time}", "%Y-%m-%d %H:%M")


# ──────────────────────────────────────────────────────────────────────────────
# Visitor & Reception Log (Events vs Profiles)
# ──────────────────────────────────────────────────────────────────────────────

def get_or_create_visitor(name: str, session) -> Visitor:
    visitor = session.query(Visitor).filter(Visitor.name.ilike(name)).first()
    if not visitor:
        visitor = Visitor(name=name)
        session.add(visitor)
        session.flush()
    return visitor

def log_reception_entry(person_name: str, person_type: str, notes: str = "", linked_visitor_id: Optional[int] = None, linked_employee_id: Optional[int] = None) -> int:
    """Legacy function brought back: Logs an ad-hoc reception entry."""
    session = SessionLocal()
    try:
        # If they didn't pass a linked ID but provided a name, map them to a Visitor profile
        if not linked_visitor_id and not linked_employee_id and person_name:
            visitor = get_or_create_visitor(person_name, session)
            linked_visitor_id = visitor.id

        entry = ReceptionLog(
            visitor_id=linked_visitor_id,
            employee_id=linked_employee_id,
            person_type=person_type,
            notes=notes,
            check_in_time=datetime.utcnow()
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
    """Legacy function brought back: Checks out an ad-hoc reception log."""
    session = SessionLocal()
    try:
        entry = session.query(ReceptionLog).filter(ReceptionLog.id == log_id).first()
        if entry:
            entry.check_out_time = datetime.utcnow()
            session.commit()
    except Exception as exc:
        logger.error("log_reception_checkout failed: %s", exc)
        session.rollback()
    finally:
        session.close()

def add_visitor(name: str, meeting_with: str, purpose: str) -> Tuple[str, int]:
    session = SessionLocal()
    try:
        visitor = get_or_create_visitor(name, session)
        employee = _resolve_employee(meeting_with, session)
        
        log_entry = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=employee.id if employee else None,
            person_type="VISITOR",
            purpose=purpose,
            check_in_time=datetime.utcnow()
        )
        session.add(log_entry)
        session.flush() 

        year = datetime.utcnow().year
        badge_id = f"VIS-{year}-{log_entry.id:04d}"
        log_entry.badge_id = badge_id
        
        session.commit()
        return badge_id, visitor.id
    except Exception as exc:
        logger.error("add_visitor failed: %s", exc)
        session.rollback()
        return "", -1
    finally:
        session.close()

def checkout_visitor(badge_id: str) -> None:
    session = SessionLocal()
    try:
        log_entry = session.query(ReceptionLog).filter(
            ReceptionLog.badge_id == badge_id,
            ReceptionLog.check_out_time.is_(None)
        ).first()
        
        if log_entry:
            log_entry.check_out_time = datetime.utcnow()
            session.commit()
    except Exception as exc:
        logger.error("checkout_visitor failed: %s", exc)
        session.rollback()
    finally:
        session.close()

def get_visitor_by_name(name: str) -> Optional[Visitor]:
    session = SessionLocal()
    try:
        return session.query(Visitor).filter(Visitor.name.ilike(name)).order_by(Visitor.id.desc()).first()
    except Exception as exc:
        logger.error("get_visitor_by_name failed: %s", exc)
        return None
    finally:
        session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Settings & Company Details
# ──────────────────────────────────────────────────────────────────────────────

def set_setting(key: str, value: str) -> None:
    session = SessionLocal()
    try:
        existing = session.query(Settings).filter(Settings.key == key).first()
        if existing: existing.value = value
        else: session.add(Settings(key=key, value=value))
        session.commit()
    except Exception as exc:
        logger.error("set_setting failed: %s", exc)
        session.rollback()
    finally: session.close()

def get_setting(key: str) -> str:
    session = SessionLocal()
    try:
        row = session.query(Settings).filter(Settings.key == key).first()
        return row.value if row else ""
    except Exception as exc:
        logger.error("get_setting failed: %s", exc)
        return ""
    finally: session.close()

def set_company_details(name: str, address: str = "", phone: str = "", email: str = "", website: str = "") -> None:
    set_setting("company_name", name)
    if address: set_setting("company_address", address)
    if phone: set_setting("company_phone", phone)
    if email: set_setting("company_email", email)
    if website: set_setting("company_website", website)

def get_company_details() -> Dict[str, str]:
    return {
        "name": get_setting("company_name"),
        "address": get_setting("company_address"),
        "phone": get_setting("company_phone"),
        "email": get_setting("company_email"),
        "website": get_setting("company_website"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Employees
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_employee(search_term: str, session_obj=None) -> Optional[Employee]:
    if not search_term or not str(search_term).strip(): return None
    term = str(search_term).strip()
    session = session_obj or SessionLocal()
    try:
        return session.query(Employee).filter(
            or_(Employee.name.ilike(f"%{term}%"), Employee.role.ilike(f"%{term}%")),
            Employee.is_public == True
        ).first()
    except Exception as exc:
        logger.error("_resolve_employee failed for %r: %s", search_term, exc)
        return None
    finally:
        if not session_obj: session.close()

def get_employee_by_name_or_role(search_term: str) -> Optional[Employee]:
    return _resolve_employee(search_term)

def get_employee_by_name(name: str) -> Optional[Employee]:
    if not name: return None
    name_clean = name.lower().strip()
    session = SessionLocal()
    try:
        public = session.query(Employee).filter(Employee.is_public == True)
        row = public.filter(Employee.name.ilike(name_clean)).first()
        if row: return row

        row = public.filter(or_(Employee.name.ilike(f"%{name_clean}%"), Employee.name.ilike(f"{name_clean}%"))).first()
        if row: return row

        all_employees = public.all()
        emp_names = [e.name for e in all_employees]
        matches = difflib.get_close_matches(name_clean, emp_names, n=1, cutoff=0.6)
        if matches:
            for emp in all_employees:
                if emp.name == matches[0]: return emp
        return None
    except Exception as exc:
        logger.error("get_employee_by_name failed: %s", exc)
        return None
    finally:
        session.close()

def get_employee_by_name_and_department(name: str, department: str) -> Optional[Employee]:
    if not name or not department: return None
    name_clean = name.lower().strip()
    dept_clean = department.lower().strip()
    session = SessionLocal()
    try:
        return session.query(Employee).filter(
            Employee.name.ilike(f"%{name_clean}%"),
            or_(Employee.department.ilike(f"%{dept_clean}%"), Employee.role.ilike(f"%{dept_clean}%")),
            Employee.is_public == True,
        ).first()
    except Exception as exc:
        logger.error("get_employee_by_name_and_department failed: %s", exc)
        return None
    finally:
        session.close()

def get_similar_employee(name: str, cutoff: float = 0.55) -> Optional[Employee]:
    if not name: return None
    name_lower = name.lower().strip()
    session = SessionLocal()
    try:
        all_employees = session.query(Employee).filter(Employee.is_public == True).all()
        if not all_employees: return None

        for emp in all_employees:
            if name_lower in emp.name.lower().split(): return emp

        emp_names = [e.name for e in all_employees]
        matches = difflib.get_close_matches(name, emp_names, n=1, cutoff=cutoff)
        if matches:
            for emp in all_employees:
                if emp.name == matches[0]: return emp
        return None
    except Exception as exc:
        logger.error("get_similar_employee failed: %s", exc)
        return None
    finally:
        session.close()

def get_hr(name: str = "HR") -> Optional[Employee]:
    session = SessionLocal()
    try:
        emp = session.query(Employee).filter(Employee.department.ilike("hr"), Employee.role.ilike("%manager%"), Employee.is_public == True).first()
        if not emp:
            emp = session.query(Employee).filter(Employee.department.ilike("hr"), Employee.is_public == True).first()
        return emp
    except Exception as exc:
        logger.error("get_hr failed: %s", exc)
        return None
    finally:
        session.close()

def get_department_manager(department: str) -> Optional[Employee]:
    session = SessionLocal()
    try:
        emp = session.query(Employee).filter(Employee.department.ilike(department), Employee.role.ilike("%manager%"), Employee.is_public == True).first()
        if not emp:
            emp = session.query(Employee).filter(Employee.department.ilike(f"%{department}%"), Employee.role.ilike("%manager%"), Employee.is_public == True).first()
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
    norm_date = _normalize_date(meeting_date)
    if not norm_date: return []
    
    session = SessionLocal()
    try:
        emp = _resolve_employee(employee_name, session)
        if not emp: return []

        start_dt = datetime.strptime(norm_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=1)

        return session.query(Meeting).filter(
            Meeting.host_employee_id == emp.id,
            Meeting.scheduled_start >= start_dt,
            Meeting.scheduled_start < end_dt,
            Meeting.status == "scheduled"
        ).order_by(Meeting.scheduled_start).all()
    except Exception as exc:
        logger.error("get_employee_meetings failed: %s", exc)
        return []
    finally:
        session.close()

def get_available_slots(employee_name: str, meeting_date: str) -> List[str]:
    norm_date = _normalize_date(meeting_date)
    if not norm_date: return []

    session = SessionLocal()
    try:
        emp = _resolve_employee(employee_name, session)
        if not emp: return ALL_SLOTS_1H

        start_dt = datetime.strptime(norm_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=1)

        booked_meetings = session.query(Meeting).filter(
            Meeting.host_employee_id == emp.id, 
            Meeting.scheduled_start >= start_dt, 
            Meeting.scheduled_start < end_dt, 
            Meeting.status == "scheduled"
        ).all()

        booked_times = {m.scheduled_start.strftime("%H:%M") for m in booked_meetings if m.scheduled_start}

        return [s for s in ALL_SLOTS_1H if s not in booked_times]
    except Exception as exc:
        logger.error("get_available_slots failed: %s", exc)
        return []
    finally:
        session.close()

def schedule_meeting(organizer_name: str, organizer_type: str, employee_name: str, meeting_date: str, meeting_time: str, purpose: str = "") -> Optional[int]:
    scheduled_dt = _get_datetime(meeting_date, meeting_time)
    if not scheduled_dt: return None

    session = SessionLocal()
    try:
        emp = _resolve_employee(employee_name, session)
        if not emp: 
            logger.warning("Employee '%s' not found for meeting schedule.", employee_name)
            return None

        visitor = get_or_create_visitor(organizer_name, session)

        existing = session.query(Meeting).filter(
            Meeting.host_employee_id == emp.id,
            Meeting.scheduled_start == scheduled_dt,
            Meeting.status == "scheduled"
        ).first()
        if existing: 
            return existing.id

        new_meeting = Meeting(
            host_employee_id=emp.id,
            visitor_id=visitor.id,
            scheduled_start=scheduled_dt,
            purpose=purpose,
            status="scheduled",
            created_at=datetime.utcnow()
        )
        
        session.add(new_meeting)
        session.commit()
        session.refresh(new_meeting)
        return new_meeting.id
    except Exception as exc:
        logger.error("schedule_meeting failed: %s", exc)
        session.rollback()
        return None
    finally:
        session.close()

def cancel_meeting(meeting_id: int) -> None:
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