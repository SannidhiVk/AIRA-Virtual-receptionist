import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def send_meeting_notification(
    employee_name: str,
    employee_email: Optional[str],
    organizer_name: str,
    meeting_date: str,
    meeting_time: str,
    purpose: str,
) -> bool:
    """
    Internal notification hook for meeting creation.
    Replace this with Slack/Teams/SMTP integration as needed.
    """
    try:
        logger.info(
            "[NOTIFY] Meeting scheduled | employee=%s email=%s organizer=%s date=%s time=%s purpose=%s",
            employee_name,
            employee_email,
            organizer_name,
            meeting_date,
            meeting_time,
            purpose,
        )
        # Keep async behavior explicit for non-blocking task scheduling.
        await asyncio.sleep(0)
        return True
    except Exception as exc:
        logger.error("send_meeting_notification failed: %s", exc)
        return False
