import logging
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

import os
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# ✅ Fix 1: Use a bounded thread pool instead of unbounded thread spawning
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="slack_notifier")

# ✅ Fix 2: Track the last notified visitor to prevent duplicate/leaked notifications
_last_notified: dict = {}
_notify_lock = threading.Lock()


def _send_slack_notification_thread(
    employee_name: str, visitor_name: str, visitor_type: str, purpose: str
):
    if not SLACK_WEBHOOK_URL:
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
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code == 200:
            logger.info(f"Successfully posted Slack notification for {employee_name}.")
        else:
            logger.error(f"Failed to post to Slack. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Slack Webhook exception: {e}")


def send_slack_arrival(
    employee_name: str, visitor_name: str, visitor_type: str, purpose: str,
    session_id: str  # ✅ Fix 2: tie each notification to a visitor session
):
    """Submit notification to the thread pool; deduplicate per session."""
    with _notify_lock:
        # ✅ Fix 2: Only send once per unique visitor session
        if _last_notified.get(session_id) == visitor_name:
            logger.warning(f"Duplicate Slack notification blocked for session {session_id}")
            return
        _last_notified[session_id] = visitor_name

    # ✅ Fix 1: submit() to bounded pool — no runaway thread creation
    _executor.submit(
        _send_slack_notification_thread,
        employee_name, visitor_name, visitor_type, purpose
    )


def clear_session(session_id: str):
    """Call this when a visitor session ends (wake word resets)."""
    with _notify_lock:
        _last_notified.pop(session_id, None)