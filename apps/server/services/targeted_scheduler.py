import threading
import logging
from datetime import datetime

from .calendar_service import send_calendar_invite as send_google_api_invite
from .notify_email import send_calendar_invite as send_smtp_ics_invite

logger = logging.getLogger(__name__)


def _fire_targeted_scheduling(
    employee_email: str,
    employee_name: str,
    visitor_email: str,
    visitor_name: str,
    date_str: str,
    time_str: str,
    purpose: str,
):
    try:
        dt_start = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except Exception:
        return

    # 1. Google API for Internal Employee
    if employee_email:
        try:
            send_google_api_invite(
                visitor_name=visitor_name, employee_email=employee_email, dt=dt_start
            )
        except Exception as e:
            logger.error(f"Google API failed: {e}")

    # 2. SMTP/ICS for External Visitor
    if visitor_email:
        try:
            send_smtp_ics_invite(
                employee_name=visitor_name,
                employee_email=visitor_email,
                organizer_name=employee_name,
                meeting_date=date_str,
                meeting_time=time_str,
                purpose=purpose,
                organizer_email=None,
            )
        except Exception as e:
            logger.error(f"SMTP invite failed: {e}")


def schedule_meeting_targeted(
    employee_email: str,
    employee_name: str,
    visitor_email: str,
    visitor_name: str,
    date_str: str,
    time_str: str,
    purpose: str,
):
    threading.Thread(
        target=_fire_targeted_scheduling,
        args=(
            employee_email,
            employee_name,
            visitor_email,
            visitor_name,
            date_str,
            time_str,
            purpose,
        ),
    ).start()
