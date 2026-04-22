import logging
import requests
import threading
import os
from concurrent.futures import ThreadPoolExecutor

# Import load_dotenv to ensure environment variables are read from .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# ✅ Correctly initialized executor
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="slack_notifier")

# ✅ Tracking notified visitors
_last_notified: dict = {}
_notify_lock = threading.Lock()


def _send_slack_notification_thread(
    employee_name: str, visitor_name: str, visitor_type: str, purpose: str
):
    # Log that the thread actually started
    logger.info(f"Slack thread started for {visitor_name} -> {employee_name}")

    if not SLACK_WEBHOOK_URL:
        logger.error("CRITICAL: SLACK_WEBHOOK_URL is NOT SET in environment variables.")
        return

    message = (
        f"🛎️ *Visitor Arrival for {employee_name}*\n"
        f"• *Visitor Name:* {visitor_name}\n"
        f"• *Category:* {visitor_type}\n"
        f"• *Purpose:* {purpose}\n\n"
        f"_Please head to the front desk._"
    )
    payload = {"text": message}

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(
                f"✅ Successfully posted Slack notification for {employee_name}."
            )
        else:
            logger.error(
                f"❌ Failed to post to Slack. Status: {response.status_code}, Body: {response.text}"
            )
    except Exception as e:
        logger.error(f"❌ Slack Webhook exception: {e}")


def send_slack_arrival(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
):
    """Submit notification to the thread pool; deduplicate per session."""

    # DEBUG: Check if URL exists when called
    if not SLACK_WEBHOOK_URL:
        print("DEBUG: SLACK_WEBHOOK_URL is missing!")

    with _notify_lock:
        # Check for duplicates within the same session
        if _last_notified.get(session_id) == visitor_name:
            logger.warning(
                f"Blocking duplicate notification for {visitor_name} in session {session_id}"
            )
            return

        _last_notified[session_id] = visitor_name

    logger.info(f"Queuing Slack notification for {visitor_name}...")

    # Submit to the pool
    _executor.submit(
        _send_slack_notification_thread,
        employee_name,
        visitor_name,
        visitor_type,
        purpose,
    )


def clear_session(session_id: str):
    """Clean up the notification history for a session."""
    with _notify_lock:
        _last_notified.pop(session_id, None)
        logger.info(f"Cleared Slack notification cache for session {session_id}")
