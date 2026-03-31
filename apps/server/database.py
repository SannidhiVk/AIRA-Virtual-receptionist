import logging
import sqlite3
from pathlib import Path
import re
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import and_, or_

from receptionist.database import SessionLocal
from receptionist.models import Employee, Meeting

logger = logging.getLogger(__name__)

_db_path = Path(__file__).resolve().parent / "receptionist" / "office.db"


def get_connection():
    """
    Legacy SQLite connection helper.
    Kept for backward compatibility with any legacy scripts that may still
    rely on direct sqlite3 access.
    """
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    return conn


ALL_SLOTS_1H = [
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


def _normalize_date(meeting_date: str) -> Optional[str]:
    if not meeting_date:
        return None
    s = str(meeting_date).strip()
    for fmt in ("%Y-%m-%d",):
        try:
            d = datetime.strptime(s, fmt).date()
            return d.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_time(meeting_time: str) -> Optional[str]:
    """
    Normalize to "HH:MM" (24h) from messy inputs like "4 pm", "4:00pm", "16:00".
    """
    if not meeting_time:
        return None

    s = str(meeting_time).strip().lower()
    s = s.replace("p.m.", "pm").replace("a.m.", "am").replace(".", ":").replace(" ", "")

    # Already in 24h "HH:MM"
    if re.match(r"^\d{2}:\d{2}$", s):
        return s

    # 12h "H(:MM)?(am|pm)"
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        mer = m.group(3)

        if hour < 1 or hour > 12 or minute < 0 or minute > 59:
            return None

        if mer == "pm" and hour != 12:
            hour += 12
        if mer == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    # "H" or "H:MM" without meridiem (assume 24h)
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?$", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None
        return f"{hour:02d}:{minute:02d}"

    return None


def _resolve_employee(search_term: str) -> Optional[Employee]:
    """
    Find an employee record by name or role/title fragment.
    """
    if not search_term:
        return None
    term = str(search_term).strip()
    if not term:
        return None

    session = SessionLocal()
    try:
        emp = (
            session.query(Employee)
            .filter(
                or_(Employee.name.ilike(f"%{term}%"), Employee.role.ilike(f"%{term}%"))
            )
            .first()
        )
        return emp
    except Exception as e:
        logger.error("Employee lookup failed for %r: %s", search_term, e)
        return None
    finally:
        session.close()


# -----------------------------------------------------------------------------
# Backward-compatible re-exports
# -----------------------------------------------------------------------------
# The meeting scheduler DB helpers are now implemented in
# `apps/server/receptionist/database.py`.
from receptionist.database import (  # noqa: E402
    get_available_slots as _get_available_slots,
    get_employee_by_name_or_role as _get_employee_by_name_or_role,
    schedule_meeting as _schedule_meeting,
)

get_employee_by_name_or_role = _get_employee_by_name_or_role
get_available_slots = _get_available_slots
schedule_meeting = _schedule_meeting

"""
NOTE:
This module only re-exports meeting scheduling DB helpers from
`apps/server/receptionist/database.py`.
"""
