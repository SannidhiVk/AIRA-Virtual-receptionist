import logging
import requests
import json
import threading
import os

logger = logging.getLogger(__name__)
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")

def _send_teams_notification_thread(employee_name: str, visitor_name: str, visitor_type: str, purpose: str):
    if not TEAMS_WEBHOOK_URL:
        return
    card_payload = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "contentUrl": None,
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.2",
                "body": [
                    {"type": "TextBlock", "text": f"🛎️ Visitor Arrival for {employee_name}", "weight": "bolder", "size": "Medium"},
                    {"type": "FactSet", "facts": [
                        {"title": "Visitor Name:", "value": visitor_name},
                        {"title": "Category:", "value": visitor_type},
                        {"title": "Purpose:", "value": purpose}
                    ]},
                    {"type": "TextBlock", "text": "Please head to the front desk to meet your visitor.", "wrap": True}
                ]
            }
        }]
    }
    try:
        requests.post(TEAMS_WEBHOOK_URL, data=json.dumps(card_payload), headers={"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"Teams Webhook exception: {e}")

def send_teams_arrival(employee_name: str, visitor_name: str, visitor_type: str, purpose: str):
    threading.Thread(target=_send_teams_notification_thread, args=(employee_name, visitor_name, visitor_type, purpose)).start()