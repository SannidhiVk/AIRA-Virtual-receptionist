import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from receptionist.database import (
    get_available_slots,
    get_employee_by_name_or_role,
    schedule_meeting,
)
from models.groq_processor import GroqProcessor
from client_context import get_last_employee_name

logger = logging.getLogger(__name__)


MEETING_TRIGGERS = [
    "schedule",
    "book",
    "arrange",
    "meeting with",
    "appointment with",
    "want to meet",
    "fix a meeting",
    "i want to meet",
    "meet",
]

CANCEL_PHRASES = [
    "cancel",
    "stop",
    "nevermind",
    "abort",
    "exit",
    "thank you",
    "thanks",
]

CONFIRM_YES = ["yes", "yeah", "confirm", "sure", "okay", "ok", "yep", "please"]
CONFIRM_NO = ["no", "nope", "dont", "don't", "not now", "cancel"]

_client_meeting_state: Dict[str, Dict[str, Any]] = {}


def _new_state() -> Dict[str, Any]:
    return {
        "meeting_state": "IDLE",  # IDLE -> IN_PROGRESS -> GET_PURPOSE -> CONFIRM -> IDLE
        "employee_query": None,  # raw name/title from user
        "employee_name": None,  # resolved Employee.name
        "employee_email": None,
        "employee_cabin": None,
        "meeting_date": None,  # YYYY-MM-DD
        "meeting_time": None,  # HH:MM (24h)
        "purpose": None,
        "organizer_name": None,  # visitor name if we can extract it
        "organizer_type": "visitor",
    }


def _get_client_state(client_id: str) -> Dict[str, Any]:
    if client_id not in _client_meeting_state:
        _client_meeting_state[client_id] = _new_state()
    return _client_meeting_state[client_id]


def _clear_client_state(client_id: str) -> None:
    _client_meeting_state[client_id] = _new_state()


def _normalize_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    s = str(date_str).strip().lower()

    # Already YYYY-MM-DD
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        return d.strftime("%Y-%m-%d")
    except ValueError:
        pass

    today = datetime.now().date()

    if s in {"today", "now"}:
        return today.strftime("%Y-%m-%d")
    if s == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if s.startswith("in "):
        m = re.match(r"^in\s+(\d+)\s+days?$", s)
        if m:
            return (today + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

    # next monday / next tuesday / ...
    m = re.match(
        r"^(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s
    )
    if m:
        weekday = m.group(2)
        weekday_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        target = weekday_map[weekday]
        days_ahead = (target - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return None


def _normalize_time(time_str: str) -> Optional[str]:
    if not time_str:
        return None
    s = str(time_str).strip().lower()
    s = s.replace("p.m.", "pm").replace("a.m.", "am").replace(".", ":").replace(" ", "")

    # HH:MM
    if re.match(r"^\d{2}:\d{2}$", s):
        return s

    # H(:MM)?(am|pm)
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

    return None


def _format_date_for_speech(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%A, %B %d").replace(" 0", " ")
    except Exception:
        return str(date_str)


def _format_time_for_speech(time_str: str) -> str:
    """
    Converts "16:00" -> "4 PM" / "09:00" -> "9 AM"
    """
    try:
        h, m = map(int, time_str.split(":"))
        mer = "AM" if h < 12 else "PM"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        if m == 0:
            return f"{h12} {mer}"
        return f"{h12}:{m:02d} {mer}"
    except Exception:
        return str(time_str)


def _format_slots_for_speech(slots: List[str]) -> str:
    return (
        ", ".join(_format_time_for_speech(s) for s in slots)
        if slots
        else "no available slots"
    )


async def _extract_meeting_details(text: str) -> Dict[str, Optional[str]]:
    """
    Meeting-specific extraction with deterministic temperature=0.
    Expected JSON:
      { "employee_name": str|null, "date": "YYYY-MM-DD"|null, "time": "HH:MM"|null, "purpose": str|null }
    """
    system_prompt = """
You are extracting meeting scheduling details for a receptionist.
Return ONLY a valid JSON object (no markdown, no explanations).

Rules:
1. employee_name: the person to meet (name or role/title). If unknown, use null.
2. date: resolve relative dates like today/tomorrow/next monday into YYYY-MM-DD. If unknown, use null.
3. time: convert into HH:MM using 24-hour clock (e.g., "4 pm" -> "16:00", "9:30 am" -> "09:30"). If unknown, use null.
4. purpose: short reason for the meeting (e.g., "interview", "discussion", "demo"). If unknown, use null.
"""
    try:
        groq = GroqProcessor.get_instance()
        response = await groq.client.chat.completions.create(
            model=groq.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)

        for k in ["employee_name", "date", "time", "purpose"]:
            if k in parsed and isinstance(parsed[k], str):
                if parsed[k].strip().lower() in ["null", "none", ""]:
                    parsed[k] = None
        return {
            "employee_name": parsed.get("employee_name"),
            "date": parsed.get("date"),
            "time": parsed.get("time"),
            "purpose": parsed.get("purpose"),
        }
    except Exception as e:
        logger.error("Meeting extraction failed: %s", e)
        return {"employee_name": None, "date": None, "time": None, "purpose": None}


async def handle_meeting_request(
    client_id: str,
    text: str,
    speak_and_emit: Any = None,
) -> Tuple[bool, str]:
    """
    Slot-filling meeting scheduler.
    Returns: (handled, reply_text)
    """
    state = _get_client_state(client_id)
    meeting_state = state.get("meeting_state", "IDLE")

    text = (text or "").strip()
    text_lower = text.lower()

    # If we're already in the meeting flow, treat cancellation as handled.
    if meeting_state != "IDLE":
        if any(phrase in text_lower for phrase in CANCEL_PHRASES):
            logger.info("Meeting cancelled for client %s", client_id)
            _clear_client_state(client_id)
            return (
                True,
                "Okay, I've cancelled the meeting request. How else can I help?",
            )

    # 1) If idle, detect meeting intent.
    if meeting_state == "IDLE":
        triggers_hit = any(t in text_lower for t in MEETING_TRIGGERS)

        if not triggers_hit:
            # Low-cost intent check via Ollama extraction (still async).
            try:
                groq = GroqProcessor.get_instance()
                extracted = await groq.extract_intent_and_entities(text)
                intent = extracted.get("intent")
                triggers_hit = intent == "schedule_meeting"
            except Exception:
                triggers_hit = False

        if not triggers_hit:
            return False, ""

        state["meeting_state"] = "IN_PROGRESS"
        meeting_state = "IN_PROGRESS"

    # 2) CONFIRM state
    if meeting_state == "CONFIRM":
        if any(w in text_lower for w in CONFIRM_YES):
            emp_name = state.get("employee_name") or state.get("employee_query")
            meeting_date = state.get("meeting_date")
            meeting_time = state.get("meeting_time")
            purpose = state.get("purpose") or ""
            organizer_name = state.get("organizer_name") or "Visitor"
            organizer_type = state.get("organizer_type") or "visitor"

            if not (emp_name and meeting_date and meeting_time):
                logger.warning("Missing scheduling fields at CONFIRM for %s", client_id)
                _clear_client_state(client_id)
                return (
                    True,
                    "I couldn't confirm everything. What date and time should we use?",
                )

            meeting_id = schedule_meeting(
                organizer_name=organizer_name,
                organizer_type=organizer_type,
                employee_name=emp_name,
                meeting_date=meeting_date,
                meeting_time=meeting_time,
                purpose=purpose,
            )

            if not meeting_id:
                _clear_client_state(client_id)
                return (
                    True,
                    "Sorry - I had trouble saving that meeting. Could you try again?",
                )

            # Non-blocking Google Calendar invite
            if state.get("employee_email"):
                try:
                    from services.calendar_service import send_calendar_invite

                    invite_dt = datetime.strptime(
                        f"{meeting_date} {meeting_time}", "%Y-%m-%d %H:%M"
                    )

                    asyncio.create_task(
                        asyncio.to_thread(
                            send_calendar_invite,
                            organizer_name,  # visitor_name
                            state["employee_email"],  # employee_email
                            invite_dt,  # datetime
                        )
                    )
                except Exception as e:
                    logger.error("Invite scheduling failed: %s", e)

            friendly_date = _format_date_for_speech(meeting_date)
            friendly_time = _format_time_for_speech(meeting_time)
            cabin = state.get("employee_cabin")
            emp_display = state.get("employee_name") or emp_name

            _clear_client_state(client_id)
            if cabin:
                return (
                    True,
                    f"Done! I've scheduled your meeting with {emp_display} on {friendly_date} at {friendly_time}. Please head to cabin {cabin}.",
                )
            return (
                True,
                f"Done! I've scheduled your meeting with {emp_display} on {friendly_date} at {friendly_time}.",
            )

        if any(w in text_lower for w in CONFIRM_NO):
            _clear_client_state(client_id)
            return True, "No problem - I won't confirm the meeting."

        return True, "Sorry, should I confirm the meeting? Please say yes or no."

    # 3) GET_PURPOSE state
    if meeting_state == "GET_PURPOSE":
        # Accept verbatim purpose; keeps it natural for speech.
        state["purpose"] = text
        state["meeting_state"] = "CONFIRM"

        meeting_date = state.get("meeting_date")
        meeting_time = state.get("meeting_time")
        emp_display = (
            state.get("employee_name") or state.get("employee_query") or "the person"
        )

        friendly_date = _format_date_for_speech(meeting_date) if meeting_date else ""
        friendly_time = _format_time_for_speech(meeting_time) if meeting_time else ""
        return (
            True,
            f"Just to confirm - meeting with {emp_display} on {friendly_date} at {friendly_time} for {text}. Shall I confirm this?",
        )

    # 4) IN_PROGRESS state: extract and fill missing slots.
    if meeting_state == "IN_PROGRESS":
        # Extract whatever we can from the latest user text.
        try:
            groq = GroqProcessor.get_instance()
            extracted_basic = await groq.extract_intent_and_entities(text)
            entities = extracted_basic.get("entities") or {}
            if not state.get("organizer_name") and entities.get("visitor_name"):
                state["organizer_name"] = str(entities.get("visitor_name")).strip()
        except Exception:
            pass

        extracted = await _extract_meeting_details(text)

        if extracted.get("employee_name") and not state.get("employee_query"):
            state["employee_query"] = extracted.get("employee_name")
        if extracted.get("date") and not state.get("meeting_date"):
            state["meeting_date"] = _normalize_date(extracted.get("date"))
        if extracted.get("time") and not state.get("meeting_time"):
            state["meeting_time"] = _normalize_time(extracted.get("time"))
        if extracted.get("purpose") and not state.get("purpose"):
            state["purpose"] = extracted.get("purpose")

        # ✅ Pronoun resolution: if no employee was extracted (e.g. user said
        # "schedule a meeting with him/her"), fall back to the last looked-up
        # employee from this client's session.
        PRONOUNS = {"him", "her", "them", "he", "she", "that person", "someone", "this guy"}
        raw_emp = (state.get("employee_query") or "").strip().lower()
        if not state.get("employee_query") or raw_emp in PRONOUNS:
            fallback = get_last_employee_name(client_id)
            if fallback:
                logger.info(
                    "Pronoun/missing employee resolved from context: %r -> %r",
                    state.get("employee_query"),
                    fallback,
                )
                state["employee_query"] = fallback
            elif raw_emp in PRONOUNS:
                # Pronoun but no prior context — clear the bad value and ask
                state["employee_query"] = None

        # Resolve employee record
        if not state.get("employee_query"):
            return (
                True,
                "Sure - who would you like to meet? Please tell me their name or role.",
            )

        emp = get_employee_by_name_or_role(state["employee_query"])
        if not emp:
            # Keep query so they can correct it on next turn
            return (
                True,
                f"Sorry, I couldn't find '{state['employee_query']}' in our staff directory. Who did you mean?",
            )

        state["employee_name"] = emp.name
        state["employee_email"] = getattr(emp, "email", None)
        state["employee_cabin"] = getattr(emp, "cabin_number", None)

        # Date
        if not state.get("meeting_date"):
            return True, f"What date would you like to meet {emp.name}?"

        meeting_date = state["meeting_date"]
        if not _normalize_date(meeting_date):
            state["meeting_date"] = None
            return True, "I didn't catch the date. What date should I book?"

        # Slots for date
        slots = get_available_slots(emp.name, meeting_date)
        if not slots:
            state["meeting_date"] = None
            state["meeting_time"] = None
            return (
                True,
                f"{emp.name} is fully booked on {_format_date_for_speech(meeting_date)}. What other date works?",
            )

        # Time
        if not state.get("meeting_time"):
            return (
                True,
                f"{emp.name} has availability on {_format_date_for_speech(meeting_date)} at: {_format_slots_for_speech(slots)}. What time works for you?",
            )

        meeting_time = state["meeting_time"]
        if meeting_time not in slots:
            state["meeting_time"] = None
            return (
                True,
                f"Sorry, {emp.name} isn't available at {_format_time_for_speech(meeting_time)}. Available times are: {_format_slots_for_speech(slots)}.",
            )

        # Purpose
        if not state.get("purpose"):
            state["meeting_state"] = "GET_PURPOSE"
            return True, f"Great. What's the purpose of the meeting with {emp.name}?"

        # Ready to confirm
        state["meeting_state"] = "CONFIRM"
        friendly_date = _format_date_for_speech(meeting_date)
        friendly_time = _format_time_for_speech(meeting_time)
        return (
            True,
            f"Just to confirm - meeting with {emp.name} on {friendly_date} at {friendly_time} for {state['purpose']}. Shall I confirm this?",
        )

    # Should never happen, but keep it safe.
    return False, ""