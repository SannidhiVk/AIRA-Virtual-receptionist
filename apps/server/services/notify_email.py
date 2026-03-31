import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# We are using Gmail's SMTP server
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def send_calendar_invite(
    employee_name: str,
    employee_email: str,
    organizer_name: str,
    meeting_date: str,
    meeting_time: str,
    purpose: str,
    organizer_email: str = None,
):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("⚠️ Missing EMAIL_SENDER or EMAIL_PASSWORD in .env file. Skipping email.")
        return

    if not employee_email:
        print(f"⚠️ No email address found for {employee_name}. Skipping email.")
        return

    # 1. Calculate Start and End times for the calendar event (Assuming a 1-hour meeting)
    start_dt = datetime.strptime(f"{meeting_date} {meeting_time}", "%Y-%m-%d %H:%M")
    end_dt = start_dt + timedelta(hours=1)

    # Format dates for the ICS file (YYYYMMDDTHHMMSS)
    dtstart = start_dt.strftime("%Y%m%dT%H%M%S")
    dtend = end_dt.strftime("%Y%m%dT%H%M%S")
    dtstamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # 2. Build the ICS Calendar string
    # Include organizer as an ATTENDEE in the invite if their email is known
    organizer_attendee_line = (
        f"\nATTENDEE;RSVP=TRUE:mailto:{organizer_email}" if organizer_email else ""
    )

    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//AlmostHuman Receptionist//EN
METHOD:REQUEST
BEGIN:VEVENT
UID:{dtstamp}-{organizer_name.replace(" ", "")}@almosthuman.local
DTSTAMP:{dtstamp}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:Meeting with {organizer_name}
DESCRIPTION:Purpose: {purpose if purpose else 'Not specified'}
ORGANIZER;CN=AlmostHuman Reception:mailto:{EMAIL_SENDER}
ATTENDEE;RSVP=TRUE:mailto:{employee_email}{organizer_attendee_line}
END:VEVENT
END:VCALENDAR"""

    # 3. Create the Email content — send to attendee, CC organizer if email known
    msg = EmailMessage()
    msg["From"] = f"AlmostHuman Reception <{EMAIL_SENDER}>"
    msg["To"] = employee_email
    if organizer_email:
        msg["Cc"] = organizer_email
    msg["Subject"] = f"📅 New Meeting Request: {organizer_name}"

    body = f"""Hello {employee_name},

A new meeting has been scheduled for you by the reception desk.

Organizer: {organizer_name}
Date: {meeting_date}
Time: {meeting_time}
Purpose: {purpose if purpose else 'Not specified'}

Please open the attached invite to add this to your calendar.
"""
    if organizer_email:
        body += f"\n{organizer_name} has been CC'd on this email as a confirmation.\n"

    msg.set_content(body)

    # Attach the calendar file
    msg.add_attachment(
        ics_content.encode("utf-8"),
        maintype="text",
        subtype="calendar",
        params={"method": "REQUEST"},
        filename="invite.ics",
    )

    # 4. Send the Email via Gmail SMTP
    all_recipients = [employee_email]
    if organizer_email:
        all_recipients.append(organizer_email)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, all_recipients, msg.as_string())
        server.quit()
        print(
            f"✅ Calendar Invite successfully sent to {employee_email}"
            + (f" (CC: {organizer_email})" if organizer_email else "")
        )
    except Exception as e:
        print(f"❌ Failed to send Calendar Invite: {e}")
