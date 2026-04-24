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
from models.groq_processor import GroqProcessor
from services.notify_slack import send_slack_arrival, clear_session as clear_slack_cache

# Logger Configuration
logger = logging.getLogger(__name__)

# Constants
AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Pvt. Ltd."
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

DELIVERY_KEYWORDS = {
    "zomato",
    "swiggy",
    "amazon",
    "flipkart",
    "delivery",
    "parcel",
    "food",
    "courier",
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
    """Modified to include the identity_updated flag for switching visitors."""
    return {
        "session_id": str(uuid.uuid4()),
        "conv_state": State.INIT,
        "last_active": datetime.utcnow(),
        "visitor_name": None,
        "visitor_email": None,
        "visitor_type": "Visitor/Guest",
        "meeting_with_raw": None,
        "meeting_with_resolved": None,
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
        "host_ask_count": 0,
        "thank_you_count": 0,
        "identity_updated": False,  # NEW: Tracks if a new person replaced the previous one
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
    if any(k in combined for k in DELIVERY_KEYWORDS):
        return "Delivery"
    if re.search(r"\b(vendor|electrician|plumber|maintenance)\b", combined):
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
    if not search_term or not search_term.strip():
        return None
    clean = search_term.strip().lower()
    db = SessionLocal()
    try:
        # Step 1: Exact Name Match
        # Step 2: Role/Department Match (e.g., "HR Manager", "Sales")
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

        # Step 3: Hardcoded Role Fallbacks
        if not emp:
            if "hr" in clean:
                emp = db.query(Employee).filter(Employee.department.ilike("hr")).first()
            elif "sales" in clean:
                emp = (
                    db.query(Employee)
                    .filter(Employee.department.ilike("sales"))
                    .first()
                )
            elif "admin" in clean or "administrator" in clean:
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

    # 1. PERSONA SWITCH: If a new person says 'Delivery' while Sudha/Jack is logged in, reset.
    new_v_type = entities.get("visitor_type")
    if new_v_type == "Delivery" or any(k in query_low for k in DELIVERY_KEYWORDS):
        if state.get("is_employee"):
            state["visitor_name"] = None
            state["is_employee"] = False
        state["is_delivery"] = True
        state["visitor_type"] = "Delivery"

    # 2. IDENTITY LOCKING: Trust the LLM extraction, but only if it's not a blacklist word.
    v_name = entities.get("visitor_name")
    if v_name and v_name.lower() not in NAME_BLACKLIST:
        state["visitor_name"] = v_name.capitalize()

    # 3. ROLE DETECTION
    if new_v_type == "Employee" or any(
        k in query_low
        for k in ["intern", "employee", "work here", "staff", "director", "manager"]
    ):
        state["is_employee"] = True
        state["visitor_type"] = "Internal Staff"

    # 4. HOST & SCHEDULING DATA
    if entities.get("employee_name"):
        state["meeting_with_raw"] = entities["employee_name"]
    if entities.get("date"):
        state["sched_date"] = _normalize_date(str(entities["date"]))
    if entities.get("time"):
        state["sched_time"] = _normalize_time(str(entities["time"]))
    if entities.get("purpose"):
        state["purpose"] = entities["purpose"]
        state["sched_purpose"] = entities["purpose"]


# ─────────────────────────────────────────────────────────────────────────────
# COMMIT & NOTIFICATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────


# 3. IMPROVED DATABASE LOGGING (Detailed Notes)
def _commit_checkin(state: Dict[str, Any], client_id: str, user_query: str) -> bool:
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name")

        # 1. CROSS-CHECK: Is this 'visitor' actually an employee?
        emp_record = db.query(Employee).filter(Employee.name.ilike(v_name)).first()
        if emp_record:
            state["visitor_type"] = "Employee"
            state["is_employee"] = True

        # 2. Resolve Visitor Profile
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()

        # 3. Resolve Host
        host_raw = state.get("meeting_with_raw")
        host_emp = _lookup_employee(host_raw)

        v_type = state.get("visitor_type", "Visitor/Guest")
        purpose = state.get("purpose") or "Meeting"

        # 4. Save to reception_logs (Matching your exact screenshot format)
        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=host_emp.id if host_emp else None,
            person_type=v_type,
            purpose=purpose,
            notes=f"[{v_type}] Purpose: {purpose} | Met via Jarvis Assistant",
            check_in_time=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

        # 5. TRIGGER SLACK (Synchronized)
        target_name = host_emp.name if host_emp else "Admin Team"
        send_slack_arrival(target_name, v_name, v_type, purpose, state["session_id"])

        return True
    except Exception as e:
        logger.error(f"Sync Commit Failed: {e}")
        return False
    finally:
        db.close()


# FIX: Database Commit for the 'meetings' table
def _commit_meeting_to_db(state: Dict[str, Any]) -> bool:
    """This ensures data is actually written to the meetings table."""
    try:
        from receptionist.database import schedule_meeting

        # This calls the SQLAlchemy function in database.py
        meeting_id = schedule_meeting(
            organizer_name=state.get("visitor_name") or "Guest",
            organizer_type=state.get("visitor_type", "Visitor/Guest"),
            employee_name=state.get("sched_employee_name")
            or state.get("meeting_with_resolved"),
            meeting_date=state.get("sched_date"),
            meeting_time=state.get("sched_time"),
            purpose=state.get("sched_purpose") or "General Meeting",
        )

        if meeting_id:
            logger.info(f"Successfully saved meeting ID {meeting_id} to database.")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to save meeting: {e}")
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


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], user_query: str, intent: str
) -> None:
    """Modified to detect identity switches (e.g., Sanjay to Deepak) and improved category mapping."""
    query_low = user_query.lower()

    # 1. IDENTITY CHANGE DETECTION (The Sanjay/Deepak Fix)
    v_name = entities.get("visitor_name")
    if v_name and v_name.lower() not in NAME_BLACKLIST:
        new_name = v_name.capitalize()
        # If a name is already stored but the NEW name is different, trigger a reset
        if state.get("visitor_name") and state["visitor_name"] != new_name:
            logger.info(
                f"Identity Switch Detected: {state['visitor_name']} -> {new_name}"
            )
            state["visitor_name"] = new_name
            state["identity_updated"] = True  # Flag to force a new Slack notification
            state["conv_state"] = (
                State.INIT
            )  # Reset logic gate so it re-logs the new person
        elif not state.get("visitor_name"):
            state["visitor_name"] = new_name

    # 2. IMPROVED VISITOR TYPE MAPPING
    if any(k in query_low for k in ["employee", "staff", "intern", "work here"]):
        state["visitor_type"] = "Employee"
        state["is_employee"] = True
    elif "interview" in query_low:
        state["visitor_type"] = "Interviewee"
    elif any(k in query_low for k in ["zomato", "somato", "swiggy", "food"]):
        state["visitor_type"] = "Food Delivery"
        state["is_delivery"] = True
    elif any(k in query_low for k in ["amazon", "flipkart", "parcel", "delivery"]):
        state["visitor_type"] = "Delivery"
        state["is_delivery"] = True
    elif any(k in query_low for k in ["demo", "client", "hdfc", "bank"]):
        state["visitor_type"] = "Client"
    elif any(
        k in query_low
        for k in ["urban company", "fix", "maintenance", "leak", "ac", "plumber"]
    ):
        state["visitor_type"] = "Contractor/Vendor"
        # Auto-assign Administration Team for maintenance tasks
        state["meeting_with_resolved"] = "Administration Team"
        state["meeting_with_raw"] = "Administration Team"
    elif not state.get("visitor_type"):
        state["visitor_type"] = entities.get("visitor_type") or "Visitor/Guest"

    # 3. HOST & SCHEDULING DATA
    new_host = entities.get("employee_name")
    if new_host and not _is_jarvis(new_host) and not state.get("meeting_with_resolved"):
        emp = _lookup_employee(new_host)
        if emp:
            state["meeting_with_resolved"] = emp.name
            state["meeting_with_raw"] = emp.name
        else:
            state["meeting_with_raw"] = new_host

    if entities.get("date"):
        state["sched_date"] = _normalize_date(str(entities["date"]))
    if entities.get("time"):
        state["sched_time"] = _normalize_time(str(entities["time"]))
    if entities.get("purpose"):
        state["sched_purpose"] = entities["purpose"]
        state["purpose"] = entities["purpose"]


# ─────────────────────────────────────────────────────────────────────────────
# CORE AI ROUTING & LLM
# ─────────────────────────────────────────────────────────────────────────────


async def _llm_reply(
    situation: str,
    state: Dict[str, Any],
    user_query: str = None,
    client_id: str = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    llm = GroqProcessor.get_instance()
    ctx = context if context else state

    # KNOWLEDGE BASE: This ensures the AI 'sees' what is already collected.
    info_block = f"""
    - User Identity: {ctx.get('visitor_name') or 'UNKNOWN'}
    - User Role: {'Employee/Staff' if state.get('is_employee') else state.get('visitor_type')}
    - Target Recipient: {ctx.get('meeting_with_resolved') or ctx.get('meeting_with_raw') or 'UNKNOWN'}
    - Meeting Date: {ctx.get('sched_date') or 'NOT SET'}
    - Meeting Time: {ctx.get('sched_time') or 'NOT SET'}
    """

    prompt = f"""
    You are {AI_NAME}, the intelligent receptionist.
    KNOWLEDGE BASE:
    {info_block}

    STRICT RULES:
    1. If an item in 'KNOWLEDGE BASE' is NOT 'UNKNOWN', NEVER ask the user for it.
    2. If the user is Staff/Employee, be respectful and skip the 'Welcome to Sharp Software' robotic intro.
    3. If 'TASK' asks for info you already have, ignore that part of the task.
    4. USER SAID: "{user_query}"
    5. GOAL: {situation}
    """
    return await llm.get_raw_response(prompt, client_id=client_id)


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
    # 1. SMART TIME VALIDATION
    if state["sched_date"] and state["sched_time"]:
        now = datetime.now()
        try:
            sched_dt = datetime.strptime(
                f"{state['sched_date']} {state['sched_time']}", "%Y-%m-%d %H:%M"
            )
            if sched_dt < now:
                state["sched_time"] = None  # Wipe the past time
                return await _llm_reply(
                    "Explain that the time is in the past and ask for a valid time.",
                    state,
                    user_query,
                    client_id,
                )
        except:
            pass

    # 2. SHORT-CIRCUIT: Check for missing data
    if not state.get("visitor_name") and not state.get("is_employee"):
        return await _llm_reply("Ask for their name.", state, user_query, client_id)

    if not state["sched_employee_raw"]:
        return await _llm_reply(
            "Ask who they want to meet.", state, user_query, client_id
        )

    if not state["sched_employee_name"]:
        emp = _lookup_employee(state["sched_employee_raw"])
        if emp:
            state["sched_employee_name"] = emp.name
        else:
            state["sched_employee_name"] = state["sched_employee_raw"]

    if not state["sched_date"]:
        return await _llm_reply(
            "Ask for the date of the meeting.", state, user_query, client_id
        )
    if not state["sched_time"]:
        return await _llm_reply(
            "Ask for the time of the meeting.", state, user_query, client_id
        )
    if not state["sched_purpose"]:
        return await _llm_reply(
            "Ask for the purpose of the meeting.", state, user_query, client_id
        )

    # 3. ACTION
    if intent == "confirm" or any(
        w in user_query.lower() for w in ["yes", "yeah", "correct"]
    ):
        # Finalize
        # Note: _commit_meeting doesn't need client_id based on your current definition,
        # but check the session_id usage
        _commit_meeting_to_db(state)
        state["scheduling_active"] = False
        state["conv_state"] = State.COMPLETED
        return await _llm_reply(
            "Confirm the booking is successful.", state, user_query, client_id
        )

    state["sched_pending_confirm"] = True
    return await _llm_reply(
        "Summarize details and ask for confirmation.", state, user_query, client_id
    )


async def route_query(client_id: str, user_query: str) -> str:
    """Modified to prioritize Database Commits so they happen before the session is cleared on 'Thank You'."""
    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_clean = user_query.lower().strip()

    # 1. NLU EXTRACTION
    extracted = await llm.extract_intent_and_entities(user_query)
    entities = extracted.get("entities", {})
    intent = extracted.get("intent", "general")

    # 2. WAKE WORD TRIGGER
    if user_query == "WAKE_WORD_TRIGGERED" or any(
        x in query_clean for x in ["hey jarvis", "hi jarvis"]
    ):
        clear_session_state(client_id)
        state = get_session_state(client_id)
        return f"Welcome to {COMPANY_NAME}. I am {AI_NAME}, how can I assist you today?"

    # 3. SMART ENTITY MERGING
    _merge_checkin_entities(state, entities, user_query, intent)

    # 4. CRITICAL FIX: COMMIT SCHEDULING BEFORE EXITING
    # This prevents the 'meetings' table from being empty if the user says 'thank you' at the end.
    is_confirming = (intent == "confirm") or any(
        w in query_clean for w in ["yes", "correct", "thank you", "confirm"]
    )
    if state.get("scheduling_active") and is_confirming:
        if state.get("sched_employee_name") or state.get("meeting_with_resolved"):
            _commit_meeting_to_db(state)  # Force DB write
            state["scheduling_active"] = False
            state["conv_state"] = State.COMPLETED

    # 5. TERMINAL HANDLING (Goodbye)
    if any(w in query_clean for w in ["thank you", "thanks", "bye", "goodbye"]):
        reply = await llm.get_raw_response(
            f"The user {state.get('visitor_name', '')} said '{user_query}'. Give a warm closing.",
            client_id,
        )
        clear_session_state(client_id)
        return reply

    # 6. FLOW ROUTING
    if intent == "employee_lookup" or any(
        x in query_clean for x in ["who is", "where is"]
    ):
        return await _handle_directory_lookup(client_id, user_query, state)

    if intent == "schedule_meeting" or state["scheduling_active"]:
        state["scheduling_active"] = True
        return await _handle_scheduling(client_id, user_query, state, entities, intent)

    if intent == "check_in" or state["meeting_with_raw"] or state["is_delivery"]:
        return await _advance_checkin(state, user_query, client_id)

    return await llm.get_response(
        client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEW: DIRECTORY LOOKUP (GROUNDED IN DATABASE)
# ─────────────────────────────────────────────────────────────────────────────


# 2. IMPROVED DIRECTORY LOOKUP (Grounding for Ajay)
async def _handle_directory_lookup(client_id: str, user_query: str, state: dict) -> str:
    search_term = user_query.lower()
    # Strip common conversational fluff to find the core role/name
    for word in ["who", "is", "the", "manager", "director", "of", "this", "company"]:
        search_term = search_term.replace(word, "").strip()

    emp = _lookup_employee(search_term or user_query)

    if emp:
        # Save the result so the check-in flow doesn't ask again!
        state["meeting_with_raw"] = emp.name
        state["meeting_with_resolved"] = emp.name
        situation = f"Tell the visitor that our {emp.role or emp.department} is {emp.name}. They are located at {emp.location or 'the main office area'}."
    else:
        situation = "Apologize and say you couldn't find that person in the directory, but offer to notify the administration team to assist them."

    return await _llm_reply(situation, state, user_query, client_id)


# 4. FIX THE "WHO DO YOU WANT TO MEET" LOOP
async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    """Modified to re-trigger Slack notifications if a visitor identity changes."""
    # Resolve host
    if state["meeting_with_raw"] and not state.get("meeting_with_resolved"):
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name
            state["meeting_with_raw"] = emp.name

    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    v_type = state.get("visitor_type")

    # THE GATE: Check if we should send Slack/Log to DB
    # Logic: Notify if NOT COMPLETED OR if we just detected an Identity Switch (Deepak)
    should_notify = (state["conv_state"] != State.COMPLETED) or state.get(
        "identity_updated"
    )

    if state.get("visitor_name") and host:
        if should_notify:
            _commit_checkin(state, client_id, user_query)
            state["conv_state"] = State.COMPLETED
            state["identity_updated"] = False  # Reset switch flag

        return await _llm_reply(
            f"Acknowledge arrival of the {v_type}. Confirm you notified {host} via Slack. Do not ask for names again.",
            state,
            user_query,
            client_id,
        )

    # Missing Info Prompts
    if not state.get("visitor_name"):
        return await _llm_reply(
            "Politely ask for their name.", state, user_query, client_id
        )

    if not host:
        return await _llm_reply(
            "Ask who they are here to see today.", state, user_query, client_id
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
