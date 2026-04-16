import logging
import requests
import threading

logger = logging.getLogger(__name__)

# Paste your Slack Webhook URL here
import os

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def _send_slack_notification_thread(
    employee_name: str, visitor_name: str, visitor_type: str, purpose: str
):
    if not SLACK_WEBHOOK_URL:
        return

    # Slack uses a simple Markdown payload
    message = (
        f"🛎️ *Visitor Arrival for {employee_name}*\n"
        f"• *Visitor Name:* {visitor_name}\n"
        f"• *Category:* {visitor_type}\n"
        f"• *Purpose:* {purpose}\n\n"
        f"_Please head to the front desk._"
    )

    payload = {"text": message}

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            logger.info(f"Successfully posted Slack notification for {employee_name}.")
        else:
            logger.error(f"Failed to post to Slack. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Slack Webhook exception: {e}")


def send_slack_arrival(
    employee_name: str, visitor_name: str, visitor_type: str, purpose: str
):
    """Spawns a thread so Sannika doesn't freeze while waiting for Slack to respond."""
    thread = threading.Thread(
        target=_send_slack_notification_thread,
        args=(employee_name, visitor_name, visitor_type, purpose),
    )
    thread.start()
