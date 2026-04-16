import os.path
import datetime
import logging
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def get_calendar_service():
    """Shows basic usage of the Google Calendar API."""
    creds = None

    # Always resolve credentials/token under apps/server
    server_dir = Path(__file__).resolve().parent.parent
    token_path = server_dir / "token.json"
    creds_path = server_dir / "credentials.json"

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)

            # This will pop up a browser window on the user's laptop asking for Google consent.
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    return service


def send_calendar_invite(
    visitor_name: str, employee_email: str, dt: datetime.datetime
) -> Optional[str]:
    """
    Creates a Google Calendar event and sends native email invitations.
    """
    try:
        service = get_calendar_service()
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Calendar: {e}")
        return None

    # The event usually lasts e.g. 30 minutes
    end_dt = dt + datetime.timedelta(minutes=30)

    event = {
        "summary": f"Meeting: {visitor_name} & Receptionist AI",
        "location": "Office",
        "description": f"A meeting scheduled by Receptionist AI between {visitor_name} and you.",
        "start": {
            "dateTime": dt.isoformat(),
            "timeZone": "Asia/Kolkata",  # Change this timezone if needed!
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "attendees": [
            {"email": employee_email},
            # You can add more people here, like a central HR email if needed
        ],
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 24 * 60},
                {"method": "popup", "minutes": 10},
            ],
        },
    }

    try:
        # sendUpdates='all' natively pushes the email invite from Google Calendar
        event_result = (
            service.events()
            .insert(calendarId="primary", body=event, sendUpdates="all")
            .execute()
        )

        logger.info(
            "Calendar event successfully created: %s" % (event_result.get("htmlLink"))
        )
        return event_result.get("htmlLink")
    except Exception as e:
        logger.error(f"Error creating calendar event: {e}")
        return None


def schedule_google_meeting_background(
    visitor_name: str,
    employee_email: str,
    date_str: str,  # "YYYY-MM-DD"
    time_str: str,  # "HH:MM"
) -> None:
    """
    Thin wrapper called by query_router._commit_meeting.
    Parses date/time strings and delegates to send_calendar_invite.
    """
    try:
        dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        link = send_calendar_invite(
            visitor_name=visitor_name,
            employee_email=employee_email,
            dt=dt,
        )
        if link:
            logger.info("Google Calendar invite sent: %s", link)
        else:
            logger.warning("Calendar invite returned no link for %s", employee_email)
    except Exception as e:
        logger.error("schedule_google_meeting_background failed: %s", e)
