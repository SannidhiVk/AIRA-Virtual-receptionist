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
        "visitor_email": None,  # Added from B
        "visitor_type": "Visitor/Guest",
        "meeting_with_raw": None,
        "meeting_with_resolved": None,
        "is_employee": False,
        "purpose": None,
        "is_delivery": False,
        "scheduling_active": False,
        "sched_employee_raw": None,
        "sched_employee_name": None,
        "sched_employee_email": None,  # Added from B
        "sched_date": None,
        "sched_time": None,
        "sched_purpose": None,
        "sched_pending_confirm": False,
        "attendees_finalized": False,  # From A
        "identity_updated": False,  # From A
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
        # 1. Try exact name match
        emp = db.query(Employee).filter(Employee.name.ilike(clean)).first()

        # 2. Try partial name/role match
        if not emp:
            emp = (
                db.query(Employee)
                .filter(
                    or_(
                        Employee.name.ilike(f"%{clean}%"),
                        Employee.role.ilike(f"%{clean}%"),
                    )
                )
                .first()
            )

        # 3. Department fallbacks
        if not emp:
            if "hr" in clean:
                emp = db.query(Employee).filter(Employee.department.ilike("hr")).first()
            elif "finance" in clean:  # Added Finance for Ravi
                emp = (
                    db.query(Employee)
                    .filter(Employee.department.ilike("finance"))
                    .first()
                )
        return emp
    except Exception as e:
        logger.error(f"DB Lookup Error: {e}")
        return None
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
    # --- Checks the Food list ---
    elif any(k in query_low for k in FOOD_DELIVERY_KEYWORDS):
        state["visitor_type"] = "Food Delivery"
        state["purpose"] = "Dropping off food"
    # --- Checks the Package list ---
    elif any(k in query_low for k in PACKAGE_DELIVERY_KEYWORDS):
        state["visitor_type"] = "Package Delivery"
        state["purpose"] = "Dropping off a package"
    # Expand the maintenance list:
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
        state["meeting_with_resolved"] = "Administration Team"
        state["meeting_with_raw"] = "Administration Team"  # Lock it in raw too


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

        # LOGS (Enhanced from File B)
        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=host_emp.id if host_emp else None,
            person_type=v_type,
            purpose=purpose,
            notes=f"[{v_type}] via Jarvis Assistant. Reason: {purpose}",
            check_in_time=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

        # SLACK
        target_name = host_emp.name if host_emp else "Admin Team"
        send_slack_arrival(target_name, v_name, v_type, purpose, state["session_id"])
        return True
    except Exception as e:
        logger.error(f"Commit Failed: {e}")
        return False
    finally:
        db.close()


# FIX: Database Commit for the 'meetings' table
def _commit_meeting_to_db(state: Dict[str, Any]) -> bool:
    """Combines DB, Slack, and Google Calendar from File B."""
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

        res = schedule_meeting(
            visitor_name, "visitor", emp_name, date_str, time_str, purpose
        )
        if not res:
            return False

        # Slack (Appends _meeting to force a new thread)
        send_slack_arrival(
            emp_name, visitor_name, visitor_type, purpose, f"{session_id}_meeting"
        )

        # Google Calendar (Background)
        if emp_email:
            schedule_google_meeting_background(
                visitor_name=visitor_name,
                employee_email=emp_email,
                date_str=date_str,
                time_str=time_str,
            )
        return True
    except Exception as e:
        logger.error(f"Meeting Commit Failed: {e}")
        return False


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

    visitor = state.get("visitor_name") or "the visitor"
    # Get host details or default
    host_name = (
        state.get("meeting_with_resolved")
        or state.get("meeting_with_raw")
        or "the relevant person"
    )

    # Determine the specific instruction based on visitor type
    v_type = state.get("visitor_type", "Guest")
    delivery_instruction = ""
    if "Delivery" in v_type:
        delivery_instruction = "IMPORTANT: Instruct the person to leave the package/food at the reception desk."
    else:
        delivery_instruction = "Instruct the visitor to take a seat in the lobby area."

    prompt = f"""
    {BASE_SYSTEM_PROMPT}
    CONTEXT:
    - Current Visitor: {visitor}
    - Visitor Category: {v_type}
    - Visiting: {host_name}
    - {delivery_instruction}

    TASK: {situation}
    USER INPUT: "{user_query}"

    STRICT RULES:
    1. Do not greet again. 
    2. If it's a delivery, tell them to leave it at the desk.
    3. If they are meeting someone, tell them they've been notified.
    4. Be extremely concise (max 2 sentences).
    """
    return await llm.get_raw_response(prompt, client_id=client_id)


async def _handle_availability_check(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    host_raw = state.get("meeting_with_raw")
    if not host_raw:
        return await _llm_reply(
            "Ask who they want to check availability for.", state, user_query, client_id
        )

    today = datetime.now().strftime("%Y-%m-%d")
    slots = get_available_slots(host_raw, today)

    if not slots:
        return await _llm_reply(
            f"Tell them {host_raw} has no slots available for today.",
            state,
            user_query,
            client_id,
        )

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
    """Restored the strict, step-by-step logic from File A"""

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
                    "Explain that the time is in the past and ask for a valid time.",
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
        # Finalize using File B's robust DB commit
        _commit_meeting_to_db(state)

        state["scheduling_active"] = False
        state["conv_state"] = State.COMPLETED
        return await _llm_reply(
            "Confirm the booking is successful and you have sent a calendar invite.",
            state,
            user_query,
            client_id,
        )

    state["sched_pending_confirm"] = True
    return await _llm_reply(
        "Summarize the meeting details and ask for confirmation.",
        state,
        user_query,
        client_id,
    )


async def _handle_directory_lookup(
    client_id: str, user_query: str, state: Dict[str, Any]
) -> str:
    """Grounded Lookup from File A (The Ajay Case)."""
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
        return f"Welcome to {COMPANY_NAME}. I am {AI_NAME}, how can I help?"

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

    if intent == "check_in" or state["meeting_with_raw"]:
        return await _advance_checkin(state, user_query, client_id)

    return await llm.get_response(
        client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
    )


async def _handle_directory_lookup(client_id: str, user_query: str, state: dict) -> str:
    search_term = user_query.lower()
    for word in ["who", "is", "the", "manager", "director", "of", "this", "company"]:
        search_term = search_term.replace(word, "").strip()

    emp = _lookup_employee(search_term or user_query)

    if emp:
        state["meeting_with_raw"] = emp.name
        state["meeting_with_resolved"] = emp.name
        situation = f"Tell the visitor that our {emp.role or emp.department} is {emp.name}. They are located at {emp.location or 'the main office area'}."
    else:
        situation = "Apologize and say you couldn't find that person in the directory, but offer to notify the administration team to assist them."

    return await _llm_reply(situation, state, user_query, client_id)


async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    if state["meeting_with_raw"] and not state.get("meeting_with_resolved"):
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name
            state["meeting_with_raw"] = emp.name

    # THE GATE: Identity Update Logic Restored
    if state.get("visitor_name") and state.get("meeting_with_raw"):
        if state["conv_state"] != State.COMPLETED or state.get("identity_updated"):
            _commit_checkin(state, user_query, client_id)
            state["conv_state"] = State.COMPLETED
            state["identity_updated"] = False
# 4. FIX THE "WHO DO YOU WANT TO MEET" LOOP
async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    # 1. Try to resolve host if not already done
    if not state.get("meeting_with_resolved") and state.get("meeting_with_raw"):
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name
            # If we find the employee, we can update the visitor type
            if state.get("visitor_type") == "Visitor/Guest":
                state["visitor_type"] = "Guest"

    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    v_name = state.get("visitor_name")

    # 2. Logic for missing name
    if not v_name:
        return await _llm_reply(
            "Politely ask the visitor for their name.", state, user_query, client_id
        )

    # 3. Logic for missing host
    if not host:
        return await _llm_reply(
            "Ask who they are here to meet.", state, user_query, client_id
        )

    # 4. Success State: Commit to DB and Notify
    if state["conv_state"] != State.COMPLETED:
        success = _commit_checkin(state, client_id, user_query)
        if not success:
            # Fallback instead of crashing if DB fails
            return "I've noted your arrival and will inform them immediately. Please wait a moment."
        state["conv_state"] = State.COMPLETED

    # Build the success situation
    v_type = state.get("visitor_type", "")
    if "Delivery" in v_type:
        situation = (
            f"Tell them to leave the delivery at the desk. Confirm {host} was notified."
        )
    else:
        situation = f"Confirm {host} has been notified via Slack and tell them to wait in the lobby."

    return await _llm_reply(situation, state, user_query, client_id)


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
