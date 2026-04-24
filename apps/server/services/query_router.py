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


# --- IN query_router.py ---


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], user_query: str, intent: str
) -> None:
    query_low = user_query.lower()

    # 1. ENHANCED VISITOR TYPE MAPPING (The Rajesh Fix)
    if any(k in query_low for k in ["employee", "staff", "intern", "work here"]):
        state["visitor_type"] = "Employee"
        state["is_employee"] = True
    elif "interview" in query_low:
        state["visitor_type"] = "Interviewee"
    elif any(k in query_low for k in ["zomato", "swiggy", "food", "lunch", "order"]):
        state["visitor_type"] = "Food Delivery"
    elif any(
        k in query_low
        for k in ["amazon", "flipkart", "parcel", "delivery", "courier", "package"]
    ):
        state["visitor_type"] = "Delivery"
    elif any(k in query_low for k in ["demo", "client", "hdfc", "bank"]):
        state["visitor_type"] = "Client"
    # RAJESH FIX: Added 'maintenance', 'leak', 'ac', 'fix' to Contractor/Vendor
    elif any(
        k in query_low
        for k in [
            "urban company",
            "fix",
            "maintenance",
            "vendor",
            "plumber",
            "electrician",
            "leak",
            "ac",
        ]
    ):
        state["visitor_type"] = "Contractor/Vendor"
    elif not state.get("visitor_type"):
        state["visitor_type"] = entities.get("visitor_type") or "Visitor/Guest"

    # 2. AUTO-RESOLVE HOST FOR SERVICE TASKS
    # If it's a vendor or delivery and they mention 'admin' or 'maintenance' or 'ac',
    # we stop asking 'Who are you meeting?' and point to Admin.
    if state["visitor_type"] in ["Contractor/Vendor", "Delivery", "Food Delivery"]:
        if any(
            k in query_low
            for k in ["admin", "reception", "desk", "maintenance", "fix", "ac", "leak"]
        ):
            state["meeting_with_resolved"] = "Administration Team"
            state["meeting_with_raw"] = "Administration Team"

    # 3. NAME & HOST LOCKING (Existing logic)
    v_name = entities.get("visitor_name")
    if v_name and v_name.lower() not in NAME_BLACKLIST:
        state["visitor_name"] = v_name.capitalize()

    new_host = entities.get("employee_name")
    if new_host and not _is_jarvis(new_host) and not state.get("meeting_with_resolved"):
        emp = _lookup_employee(new_host)
        if emp:
            state["meeting_with_resolved"] = emp.name
            state["meeting_with_raw"] = emp.name
        else:
            state["meeting_with_raw"] = new_host


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
    query_low = user_query.lower()

    # THE ATTENDEE LOOP KILLER
    if any(
        p in query_low for p in ["only me", "just us", "no one else", "nobody else"]
    ):
        state["attendees_finalized"] = True

    # Check for Host
    if not state.get("sched_employee_name"):
        emp = _lookup_employee(
            state.get("meeting_with_raw") or entities.get("employee_name")
        )
        if emp:
            state["sched_employee_name"] = emp.name
        else:
            return await _llm_reply(
                "Ask who they want to meet.", state, user_query, client_id
            )

    # If not finalized, ask ONCE
    if not state.get("attendees_finalized"):
        return await _llm_reply(
            f"Confirm meeting with {state['sched_employee_name']} and ask if anyone else is joining.",
            state,
            user_query,
            client_id,
        )

    # Proceed to Date/Time/Purpose (Trust collected entities)
    if not state.get("sched_date"):
        return await _llm_reply("Ask for the date.", state, user_query, client_id)
    if not state.get("sched_time"):
        return await _llm_reply("Ask for the time.", state, user_query, client_id)

    # 4. Final Confirmation
    if intent == "confirm" or "yes" in query_low or "confirm" in query_low:
        _commit_meeting_to_db(state)  # Your existing commit logic
        state["scheduling_active"] = False
        state["conv_state"] = State.COMPLETED
        return await _llm_reply(
            "Confirm the meeting is booked successfully.", state, user_query, client_id
        )

    return await _llm_reply(
        f"Summarize: Meeting with {state['sched_employee_name']} on {state['sched_date']} at {state['sched_time']} for {state['sched_purpose']}. Ask to confirm.",
        state,
        user_query,
        client_id,
    )


# --- IN query_router.py ---


async def route_query(client_id: str, user_query: str) -> str:
    state = get_session_state(client_id)
    query_clean = user_query.lower().strip()

    # 1. NLU Extraction
    llm = GroqProcessor.get_instance()
    extracted = await llm.extract_intent_and_entities(user_query)
    entities = extracted.get("entities", {})
    intent = extracted.get("intent", "general")

    # 2. DATA MERGING
    _merge_checkin_entities(state, entities, user_query, intent)

    # 3. CRITICAL FIX: COMMIT SCHEDULING BEFORE EXITING
    # If they confirm OR say 'thank you' while scheduling is active, SAVE IT.
    is_confirming = (intent == "confirm") or any(
        w in query_clean for w in ["yes", "correct", "thank you", "confirm"]
    )

    if state.get("scheduling_active") and is_confirming:
        if state.get("sched_employee_name") and state.get("sched_date"):
            # Commit to DB now
            success = _commit_meeting_to_db(state)
            if success:
                state["scheduling_active"] = False
                state["conv_state"] = State.COMPLETED
                logger.info(
                    f"Meeting committed for {state['visitor_name']} with {state['sched_employee_name']}"
                )

    # 4. TERMINAL HANDLING (Goodbye)
    if any(w in query_clean for w in ["bye", "goodbye"]) or (
        state["conv_state"] == State.COMPLETED and "thank you" in query_clean
    ):
        reply = await llm.get_raw_response(
            f"The visitor {state.get('visitor_name', '')} is leaving. Say a short goodbye.",
            client_id,
        )
        clear_session_state(client_id)
        return reply

    # 5. REMAINING FLOWS (Employee Lookup, Check-in, etc.)
    if intent == "employee_lookup":
        return await _handle_directory_lookup(client_id, user_query, state)

    if intent == "schedule_meeting" or state.get("scheduling_active"):
        state["scheduling_active"] = True
        return await _handle_scheduling(client_id, user_query, state, entities, intent)

    if intent == "check_in" or state.get("meeting_with_raw"):
        return await _advance_checkin(state, user_query, client_id)

    return await llm.get_response(client_id, user_query)


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
    # Resolve the Host Name for the Prompt
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    v_type = state.get("visitor_type")

    # THE GATE: If we have Name AND (Resolved Host or Admin Task)
    if state.get("visitor_name") and host:
        if state["conv_state"] != State.COMPLETED:
            _commit_checkin(state, client_id, user_query)
            state["conv_state"] = State.COMPLETED

        # GROUNDED REPLY: Use the specific visitor type in the greeting
        situation = f"Confirm that you have notified {host} about the {v_type}'s arrival. Do not ask for any more names or hosts."
        return await _llm_reply(situation, state, user_query, client_id)

    # Missing Info Handling...
    if not state.get("visitor_name"):
        return await _llm_reply(
            "Politely ask for their name.", state, user_query, client_id
        )

    if not host:
        # Special prompt for service staff who haven't named a person yet
        if v_type in ["Contractor/Vendor", "Delivery"]:
            return await _llm_reply(
                "Ask if they are here to see the Administration team or a specific person.",
                state,
                user_query,
                client_id,
            )

        return await _llm_reply(
            "Ask who they are here to see today.", state, user_query, client_id
        )

    return await _llm_reply("Assist with check-in.", state, user_query, client_id)


async def _complete_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    """Finalizing step for deliveries."""
    # --- FIXED LINE: Added client_id here ---
    _commit_checkin(state, client_id)
    state["conv_state"] = State.COMPLETED
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    return await _llm_reply(
        f"Tell them to leave the package. You will ping {host} immediately.",
        state,
        user_query,
        client_id,
    )
