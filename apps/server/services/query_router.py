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


def _commit_checkin(state: Dict[str, Any], client_id: str) -> bool:
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name") or "Guest"
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()

        host_raw = state.get("meeting_with_raw")
        emp = _lookup_employee(host_raw)

        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=emp.id if emp else None,
            person_type=state.get("visitor_type", "Visitor"),
            check_in_time=datetime.utcnow(),
            purpose=state.get("purpose") or "Office Visit",
            notes=f"Check-in for {host_raw}",
        )
        db.add(log)
        db.commit()
        if emp:
            send_slack_arrival(
                emp.name,
                v_name,
                state["visitor_type"],
                state.get("purpose") or "Meeting",
                state["session_id"],
            )
        return True
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        return False
    finally:
        db.close()


def _commit_meeting(
    emp_name, emp_email, date_str, time_str, purpose, org_name, org_email, session_id
) -> bool:
    res = schedule_meeting(org_name, "visitor", emp_name, date_str, time_str, purpose)
    if res:
        send_slack_arrival(emp_name, org_name, "Scheduled Visitor", purpose, session_id)
    return res is not None


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
    query_low = user_query.lower()

    # 1. PERSONA SWITCH: Stop 'Sudha' becoming 'Zomato'
    if any(k in query_low for k in DELIVERY_KEYWORDS):
        if state.get("is_employee"):
            state["visitor_name"] = None
            state["is_employee"] = False
        state["is_delivery"] = True
        state["visitor_type"] = "Delivery"

    # 2. NAME CAPTURE (Regex Fallback) - Captures name if NLU misses it
    if not state.get("visitor_name"):
        match = re.search(
            r"(?:i am|i'm|name is|this is|call me)\s+([a-zA-Z]+)", query_low
        )
        if match:
            extracted = match.group(1).strip().capitalize()
            if extracted.lower() not in NAME_BLACKLIST:
                state["visitor_name"] = extracted

    # 3. NLU IDENTITY MERGE
    v_name = entities.get("visitor_name")
    if v_name and not state["visitor_name"]:
        if v_name.lower() not in NAME_BLACKLIST:
            state["visitor_name"] = v_name.capitalize()

    # 4. EMPLOYEE DETECTION
    if any(
        k in query_low
        for k in ["intern", "employee", "work here", "staff", "director", "manager"]
    ):
        state["is_employee"] = True
        state["visitor_type"] = "Internal Staff"

    # 5. HOST & PURPOSE
    target = entities.get("employee_name") or entities.get("role")
    if target and not _is_jarvis(target):
        state["meeting_with_raw"] = target

    raw_p = _clean_entity(entities.get("purpose"))
    if raw_p:
        state["purpose"] = raw_p

    if not state.get("is_employee"):
        state["visitor_type"] = _determine_visitor_type(
            user_query, str(raw_p or ""), state["visitor_type"]
        )
        state["is_delivery"] = state["visitor_type"] in ["Delivery", "Food Delivery"]


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
        w in user_query.lower() for w in ["yes", "yeah", "correct", "confirm"]
    ):
        _commit_meeting(
            state["sched_employee_name"],
            None,
            state["sched_date"],
            state["sched_time"],
            state["sched_purpose"],
            state.get("visitor_name") or "Staff",
            None,
            state["session_id"],
        )
        state["scheduling_active"] = False
        state["conv_state"] = State.COMPLETED
        return await _llm_reply(
            "Confirm the booking is finished.", state, user_query, client_id
        )

    state["sched_pending_confirm"] = True
    return await _llm_reply(
        "Read back scheduling details and ask for confirmation.",
        state,
        user_query,
        client_id,
    )


async def route_query(client_id: str, user_query: str) -> str:
    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_clean = user_query.lower().strip()

    if user_query == "WAKE_WORD_TRIGGERED" or "hey jarvis" in query_clean:
        clear_session_state(client_id)
        return f"Welcome to {COMPANY_NAME}. I am {AI_NAME}, how can I assist you today?"

    extracted = await llm.extract_intent_and_entities(user_query)
    entities, intent = extracted.get("entities", {}), extracted.get("intent", "general")

    _merge_checkin_entities(state, entities, user_query, intent)

    if intent == "schedule_meeting" or state["scheduling_active"]:
        state["scheduling_active"] = True
        return await _handle_scheduling(client_id, user_query, state, entities, intent)

    if intent == "check_in" or state["meeting_with_raw"] or state["is_delivery"]:
        return await _advance_checkin(state, user_query, client_id)

    return await llm.get_response(
        client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
    )


async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    if state["meeting_with_raw"] and not state["meeting_with_resolved"]:
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name

    if not state.get("visitor_name"):
        return await _llm_reply(
            "Ask for their name politely.", state, user_query, client_id
        )
    if not state.get("meeting_with_raw"):
        return await _llm_reply(
            f"Ask who they are here to see.", state, user_query, client_id
        )

    _commit_checkin(state, client_id)
    state["conv_state"] = State.COMPLETED
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    return await _llm_reply(
        f"Confirm check-in is done and {host} notified.", state, user_query, client_id
    )


async def _complete_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    """Finalizing step for deliveries."""
    _commit_checkin(state, client_id)
    state["conv_state"] = State.COMPLETED
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    return await _llm_reply(
        f"Tell them to leave the package. You will ping {host} immediately.",
        state,
        user_query,
        client_id,
    )
