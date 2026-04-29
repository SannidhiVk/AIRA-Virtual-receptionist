import logging
import re
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from sqlalchemy import or_, and_

# Database and Model Imports
from receptionist.database import (
    SessionLocal,
    get_company_details,
    get_available_slots,
    schedule_meeting,
    get_employee_by_name,
)
from receptionist.models import Employee, Visitor, Meeting, ReceptionLog
from models.groq_processor import BASE_SYSTEM_PROMPT, GroqProcessor
from services.notify_slack import send_slack_arrival, clear_session as clear_slack_cache
from services.calendar_service import schedule_google_meeting_background

# Logger Configuration
logger = logging.getLogger(__name__)

# Constants
AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Private Limited."
SESSION_TIMEOUT_SECONDS = 300

NAME_BLACKLIST = {
    "jarvis",
    "davis",
    "darwis",
    "darvis",
    "jarves",
    "dervis",
    "bruce",
    "chalves",
    "travis",
    "unknown",
    "none",
    "null",
    "it",
    "alexa",
}
PRONOUNS = {
    "him",
    "her",
    "them",
    "he",
    "she",
    "they",
    "it",
    "that person",
    "someone",
    "this guy",
    "this person",
}
FOOD_DELIVERY_KEYWORDS = {
    "zomato",
    "swiggy",
    "food",
    "bistro",
    "blinkit",
    "danzo",
}

PACKAGE_DELIVERY_KEYWORDS = {
    "amazon",
    "flipkart",
    "ajio",
    "savana",
    "delivery",
    "parcel",
    "courier",
    "toing",
}


class State:
    """Conversation States for logic tracking."""

    INIT = "INIT"
    COLLECTING = "COLLECTING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"


# Global In-Memory Session Storage
_client_sessions: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────


def get_session_state(client_id: str) -> Dict[str, Any]:
    now = datetime.utcnow()
    if client_id in _client_sessions:
        last_active = _client_sessions[client_id].get("last_active")
        if (
            last_active
            and (now - last_active).total_seconds() > SESSION_TIMEOUT_SECONDS
        ):
            clear_session_state(client_id)

    if client_id not in _client_sessions:
        _client_sessions[client_id] = _fresh_state()

    _client_sessions[client_id]["last_active"] = now
    return _client_sessions[client_id]


def clear_session_state(client_id: str) -> None:
    """Physical deletion of session data and external cache resets."""
    if client_id in _client_sessions:
        session_id = _client_sessions[client_id].get("session_id")
        if session_id:
            clear_slack_cache(session_id)
        del _client_sessions[client_id]

    _client_sessions[client_id] = _fresh_state()
    try:
        GroqProcessor.get_instance().reset_history(client_id)
    except Exception as e:
        logger.error(f"Hardware reset failed for {client_id}: {e}")


def _fresh_state() -> Dict[str, Any]:
    return {
        "session_id": str(uuid.uuid4()),
        "conv_state": State.INIT,
        "last_active": datetime.utcnow(),
        "visitor_name": None,
        "visitor_email": None,
        "visitor_type": "Visitor/Guest",
        "greeting_sent": False,  # FIX: Prevent repeating "Good Morning"
        "meeting_with_raw": None,
        "meeting_with_resolved": None,
        "host_details": None,  # Store role for Slack
        "is_employee": False,
        "purpose": None,
        "is_delivery": False,
        "scheduling_active": False,
        "sched_employee_raw": None,
        "sched_employee_name": None,
        "sched_employee_email": None,
        "sched_date": None,
        "sched_time": None,
        "sched_purpose": None,
        "sched_pending_confirm": False,
        "attendees_finalized": False,
        "identity_updated": False,
        "host_ask_count": 0,
        "thank_you_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION & LOOKUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _is_jarvis(name: str) -> bool:
    if not name:
        return False
    return name.lower().strip().replace(".", "") in NAME_BLACKLIST


def _determine_visitor_type(text: str, purpose: str, current_type: str) -> str:
    combined = f"{text} {purpose}".lower()

    if re.search(r"\b(interview|candidate)\b", combined):
        return "Interviewee"

    # --- Checks the Food list ---
    if any(k in combined for k in FOOD_DELIVERY_KEYWORDS):
        return "Food Delivery"

    # --- Checks the Package list ---
    if any(k in combined for k in PACKAGE_DELIVERY_KEYWORDS):
        return "Package Delivery"

    # --- Expanded Maintenance list ---
    if re.search(
        r"\b(vendor|electrician|plumber|maintenance|ac|fix|leak|broken)\b", combined
    ):
        return "Contractor/Vendor"

    if re.search(r"\b(client|customer|demo)\b", combined):
        return "Client"

    return current_type or "Visitor/Guest"


def _normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().lower()
    today = datetime.now().date()

    # FIX: Handle "Next Friday", "Next Monday", etc.
    match = re.search(
        r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s
    )
    if match:
        weekday_map = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        target_day = weekday_map[match.group(1)[:3]]
        current_day = today.weekday()
        days_ahead = (target_day - current_day + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    if s in ("today", "now"):
        return today.strftime("%Y-%m-%d")
    if s == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except:
        return None


def _normalize_time(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = (
        str(raw)
        .strip()
        .lower()
        .replace("p.m.", "pm")
        .replace("a.m.", "am")
        .replace(" ", "")
    )
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if mer == "pm" and hour != 12:
            hour += 12
        if mer == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    return s if re.match(r"^\d{2}:\d{2}$", s) else None


def _lookup_employee(search_term: str) -> Optional[Any]:
    if not search_term:
        return None
    clean = search_term.strip().lower()
    db = SessionLocal()
    try:
        # Step 1: Exact Name/Role/Dept Match (File A Logic)
        emp = (
            db.query(Employee)
            .filter(
                or_(
                    Employee.name.ilike(f"%{clean}%"),
                    Employee.role.ilike(f"%{clean}%"),
                    Employee.department.ilike(f"%{clean}%"),
                )
            )
            .first()
        )

        # Step 2: Fallbacks (File A)
        if not emp:
            if "hr" in clean:
                emp = db.query(Employee).filter(Employee.department.ilike("hr")).first()
            elif "sales" in clean:
                emp = (
                    db.query(Employee)
                    .filter(Employee.department.ilike("sales"))
                    .first()
                )
            elif "admin" in clean:
                emp = (
                    db.query(Employee)
                    .filter(Employee.department.ilike("admin"))
                    .first()
                )
        return emp
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY MERGING (ROLE AWARE & CONTEXT LOCKING)
# ─────────────────────────────────────────────────────────────────────────────


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], user_query: str, intent: str
) -> None:
    query_low = user_query.lower()

    # 1. IDENTITY & SWITCH DETECTION (File A)
    v_name = entities.get("visitor_name")
    if v_name and v_name.lower() not in NAME_BLACKLIST:
        new_name = v_name.capitalize()

        # FIX: Check if this "visitor" is actually an employee in our DB
        emp_record = get_employee_by_name(new_name)
        if emp_record or "i am an employee" in query_low:
            state["is_employee"] = True
            state["visitor_type"] = "Employee"

        if state.get("visitor_name") and state["visitor_name"] != new_name:
            state["visitor_name"] = new_name
            state["identity_updated"] = True
            state["conv_state"] = State.INIT
        elif not state.get("visitor_name"):
            state["visitor_name"] = new_name

    # 2. EMAIL CAPTURE (File B)
    if entities.get("visitor_email"):
        state["visitor_email"] = entities["visitor_email"]

    # 3. SCHEDULING DATA LOCKING
    if entities.get("employee_name"):
        state["sched_employee_raw"] = entities["employee_name"]
        emp = _lookup_employee(entities["employee_name"])
        if emp:
            state["sched_employee_name"] = emp.name
            state["sched_employee_email"] = emp.email
            state["meeting_with_resolved"] = emp.name
            state["host_details"] = f"{emp.role} in {emp.department}"

    if entities.get("date"):
        state["sched_date"] = _normalize_date(str(entities["date"]))
    if entities.get("time"):
        state["sched_time"] = _normalize_time(str(entities["time"]))
    if entities.get("purpose"):
        state["sched_purpose"] = entities["purpose"]
        state["purpose"] = entities["purpose"]

    # 5. VISITOR TYPE MAPPING (File A)
    if "interview" in query_low:
        state["visitor_type"] = "Interviewee"
    elif any(k in query_low for k in FOOD_DELIVERY_KEYWORDS):
        state["visitor_type"] = "Food Delivery"
        state["purpose"] = "Dropping off food"
    elif any(k in query_low for k in PACKAGE_DELIVERY_KEYWORDS):
        state["visitor_type"] = "Package Delivery"
        state["purpose"] = "Dropping off a package"
    elif any(
        k in query_low
        for k in [
            "contractor",
            "maintenance",
            "ac",
            "fix",
            "leak",
            "plumber",
            "electrician",
            "broken",
        ]
    ):
        state["visitor_type"] = "Contractor/Vendor"
        # Only auto-assign Admin if no specific host was mentioned yet
        if not state.get("meeting_with_resolved"):
            state["meeting_with_resolved"] = "Administration Team"
    elif "demo" in query_low or "client" in query_low:
        state["visitor_type"] = "Client"  # Lock it in raw too


# ─────────────────────────────────────────────────────────────────────────────
# COMMIT & NOTIFICATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────


# 3. IMPROVED DATABASE LOGGING (Detailed Notes)
def _commit_checkin(state: Dict[str, Any], client_id: str, user_query: str) -> bool:
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name") or "Guest"
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()

        host_raw = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
        host_emp = _lookup_employee(host_raw)

        v_type = state.get("visitor_type", "Visitor/Guest")
        purpose = state.get("purpose") or "General Visit"

        # Note should reflect the real Host
        target_name = host_emp.name if host_emp else host_raw or "Admin Team"

        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=host_emp.id if host_emp else None,
            person_type=v_type,
            purpose=purpose,
            notes=f"[{v_type}] for {target_name} via Jarvis. Reason: {purpose}",
            check_in_time=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

        # SLACK
        send_slack_arrival(target_name, v_name, v_type, purpose, state["session_id"])
        return True
    except Exception as e:
        logger.error(f"Commit Failed: {e}")
        return False
    finally:
        db.close()


# FIX: Database Commit for the 'meetings' table
def _commit_meeting_to_db(state: Dict[str, Any]) -> bool:
    try:
        res = schedule_meeting(
            state.get("visitor_name") or "Guest",
            "visitor",
            state.get("sched_employee_name"),
            state.get("sched_date"),
            state.get("sched_time"),
            state.get("sched_purpose") or "Meeting",
        )
        if not res:
            return False

        # Slack
        send_slack_arrival(
            state["sched_employee_name"],
            state["visitor_name"],
            "Scheduled Meeting",
            state["sched_purpose"],
            state["session_id"],
        )

        # Google Calendar (Background)
        if state.get("sched_employee_email"):
            schedule_google_meeting_background(
                visitor_name=state.get("visitor_name") or "Guest",
                employee_email=state["sched_employee_email"],
                date_str=state["sched_date"],
                time_str=state["sched_time"],
            )
        return True
    except Exception as e:
        logger.error(f"Meeting Commit Failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY MERGING (UPGRADED: Persona Switch & Regex Fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _clean_entity(val: Any) -> Optional[str]:
    s = str(val).strip() if val else ""
    return (
        s
        if s and s.lower() not in ("null", "none", "") and s.lower() not in PRONOUNS
        else None
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE AI ROUTING & LLM
# ─────────────────────────────────────────────────────────────────────────────


async def _llm_reply(
    situation: str, state: Dict[str, Any], user_query: str = None, client_id: str = None
) -> str:
    llm = GroqProcessor.get_instance()

    # Strictly define the roles for the LLM
    visitor = state.get("visitor_name") or "Unknown Visitor"
    host = (
        state.get("meeting_with_resolved")
        or state.get("sched_employee_name")
        or "Administration Team"
    )

    info_block = f"""
    - TALKING TO: {visitor}
    - WANT TO SEE: {host}
    - STATUS: {'Meeting Scheduled' if state['conv_state'] == State.COMPLETED else 'Collecting Info'}
    - DATE: {state.get('sched_date') or 'None'}
    - TIME: {state.get('sched_time') or 'None'}
    """
    # FIX: Logic to check if we already greeted them
    is_first_turn = not state.get("greeting_sent", False)

    prompt = f"""
    {BASE_SYSTEM_PROMPT}
    KNOWLEDGE BASE:
    {info_block}

    STRICT RULES:
    1. GREETING: {"Give a warm greeting (Good Morning/Afternoon)" if is_first_turn else "DO NOT use a time-based greeting like Good Morning. Simply respond."}
    2. TARGET: Speak to {visitor}.
    3. GOAL: {situation}
    4. USER SAID: "{user_query}"
    """
    reply = await llm.get_raw_response(prompt, client_id=client_id)
    state["greeting_sent"] = True  # Mark that we have greeted
    return reply


async def _handle_availability_check(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    """Direct database query to check for free slots."""
    host_raw = state.get("meeting_with_raw")
    if not host_raw:
        return await _llm_reply(
            "Ask who they want to check availability for.", state, user_query, client_id
        )

    today = datetime.now().strftime("%Y-%m-%d")
    # Real DB Query
    slots = get_available_slots(host_raw, today)

    if not slots:
        return await _llm_reply(
            f"Tell them {host_raw} has no slots available for today.",
            state,
            user_query,
            client_id,
        )

    # Take first 3 slots for brevity
    slot_str = ", ".join(slots[:3])
    return await _llm_reply(
        f"Inform them {host_raw} is free at: {slot_str}. Ask if they want to book one.",
        state,
        user_query,
        client_id,
    )


async def _handle_scheduling(
    client_id: str,
    user_query: str,
    state: Dict[str, Any],
    entities: Dict[str, Any],
    intent: str,
) -> str:

    # Sync meeting_with_raw to sched_employee_raw as a fallback
    if not state.get("sched_employee_raw") and state.get("meeting_with_raw"):
        state["sched_employee_raw"] = state["meeting_with_raw"]

    # 1. SMART TIME VALIDATION
    if state["sched_date"] and state["sched_time"]:
        now = datetime.now()
        try:
            sched_dt = datetime.strptime(
                f"{state['sched_date']} {state['sched_time']}", "%Y-%m-%d %H:%M"
            )
            if sched_dt < now:
                state["sched_time"] = None
                return await _llm_reply(
                    "Explain that the time is in the past.",
                    state,
                    user_query,
                    client_id,
                )
        except:
            pass

    # 2. SHORT-CIRCUIT: Check for missing data in logical order
    if not state.get("visitor_name") and not state.get("is_employee"):
        return await _llm_reply(
            "Ask for their name politely.", state, user_query, client_id
        )

    if not state.get("sched_employee_raw"):
        return await _llm_reply(
            "Ask who they want to schedule the meeting with.",
            state,
            user_query,
            client_id,
        )

    if not state.get("sched_employee_name"):
        emp = _lookup_employee(state["sched_employee_raw"])
        if emp:
            state["sched_employee_name"] = emp.name
            state["sched_employee_email"] = emp.email
        else:
            state["sched_employee_name"] = state["sched_employee_raw"]

    if not state.get("sched_date"):
        return await _llm_reply(
            "Ask for the date of the meeting.", state, user_query, client_id
        )
    if not state.get("sched_time"):
        return await _llm_reply(
            "Ask for the time of the meeting.", state, user_query, client_id
        )
    if not state.get("sched_purpose"):
        return await _llm_reply(
            "Ask for the purpose of the meeting.", state, user_query, client_id
        )

    # 3. ACTION & CONFIRMATION
    if intent == "confirm" or any(
        w in user_query.lower() for w in ["yes", "yeah", "correct", "confirm", "sure"]
    ):
        _commit_meeting_to_db(state)
        state["scheduling_active"] = False
        state["conv_state"] = State.COMPLETED
        return await _llm_reply(
            "Confirm booking successful and invite sent.", state, user_query, client_id
        )

    state["sched_pending_confirm"] = True
    return await _llm_reply(
        "Summarize meeting details and ask for confirmation.",
        state,
        user_query,
        client_id,
    )


async def _handle_directory_lookup(
    client_id: str, user_query: str, state: Dict[str, Any]
) -> str:

    search_term = user_query.lower()
    for word in ["who", "is", "the", "manager", "director", "of", "this", "company"]:
        search_term = search_term.replace(word, "").strip()

    emp = _lookup_employee(search_term or user_query)
    if emp:
        state["meeting_with_raw"] = emp.name
        state["meeting_with_resolved"] = emp.name
        situation = f"Tell them our {emp.role or emp.department} is {emp.name}. They are located at {emp.location or 'the main office area'}."
    else:
        situation = "Apologize and say you couldn't find them, but offer to notify the Admin team."
    return await _llm_reply(situation, state, user_query, client_id)


async def route_query(client_id: str, user_query: str) -> str:
    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_clean = user_query.lower().strip()

    # 1. Wake word (File A)
    if any(
        x in query_clean for x in ["hey jarvis", "hi jarvis", "wake_word_triggered"]
    ):
        clear_session_state(client_id)
        # Manually return greeting only on wake
        state = get_session_state(client_id)
        from datetime import datetime

        hour = datetime.now().hour
        greet = (
            "Good Morning"
            if 5 <= hour < 12
            else "Good Afternoon" if 12 <= hour < 17 else "Good Evening"
        )
        state["greeting_sent"] = True
        return f"{greet}! Welcome to {COMPANY_NAME}. I am {AI_NAME}, how can I help you today?"

    extracted = await llm.extract_intent_and_entities(user_query)
    entities = extracted.get("entities", {})
    intent = extracted.get("intent", "general")

    _merge_checkin_entities(state, entities, user_query, intent)

    # 2. Terminal Goodbye
    if any(w in query_clean for w in ["thank you", "thanks", "bye"]):
        reply = await llm.get_raw_response(
            f"The user said '{user_query}'. Warm closing.", client_id=client_id
        )
        clear_session_state(client_id)
        return reply

    # 3. Flow Routing
    if intent == "employee_lookup" or any(
        x in query_clean for x in ["who is", "where is"]
    ):
        return await _handle_directory_lookup(client_id, user_query, state)

    if intent == "schedule_meeting" or state["scheduling_active"]:
        state["scheduling_active"] = True
        return await _handle_scheduling(client_id, user_query, state, entities, intent)

    if (
        intent == "check_in"
        or state["meeting_with_raw"]
        or state["meeting_with_resolved"]
    ):
        return await _advance_checkin(state, user_query, client_id)

    # General Chat logic
    reply = await llm.get_response(
        client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
    )
    state["greeting_sent"] = True
    return reply


# 4. FIX THE "WHO DO YOU WANT TO MEET" LOOP
async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    # Resolve host
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    if not host and state.get("meeting_with_raw"):
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name
            host = emp.name

    # Check for Identity Switch or Arrival (File A)
    if state.get("visitor_name") and host:
        if state["conv_state"] != State.COMPLETED or state.get("identity_updated"):
            _commit_checkin(state, client_id, user_query)
            state["conv_state"] = State.COMPLETED
            state["identity_updated"] = False
        return await _llm_reply(
            f"Acknowledge arrival. Confirm {host} has been notified via Slack.",
            state,
            user_query,
            client_id,
        )

    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name.", state, user_query, client_id)
    if not host:
        return await _llm_reply(
            "Ask who they are here to see.", state, user_query, client_id
        )
    return await _llm_reply("Assist with check-in.", state, user_query, client_id)


async def _complete_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    """Modified to fix the argument mismatch when calling _commit_checkin."""
    # FIXED: Added user_query to the call to match function definition
    _commit_checkin(state, client_id, user_query)
    state["conv_state"] = State.COMPLETED
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    return await _llm_reply(
        f"Tell them to leave the package. You will ping {host} immediately.",
        state,
        user_query,
        client_id,
    )
