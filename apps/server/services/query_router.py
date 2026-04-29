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
from services.calendar_service import schedule_google_meeting_background

# Logger Configuration
logger = logging.getLogger(__name__)

# Constants
AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Private limited."
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
    "dunzo",
    "blinkit",
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
        "attendees_finalized": False,  # Restored from ROUTER1
        "identity_updated": False,  # Restored from ROUTER1
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

    # 1. PERSONA SWITCH DETECTION
    is_delivery_now = any(k in query_low for k in DELIVERY_KEYWORDS) or entities.get(
        "visitor_type"
    ) in ["Delivery", "Food Delivery"]
    if is_delivery_now and not state.get("is_delivery"):
        if state.get("is_employee"):
            state["visitor_name"] = None
            state["is_employee"] = False
        state["is_delivery"] = True
        state["visitor_type"] = "Delivery"

    # 2. IDENTITY LOCKING & UPDATE DETECTION (Restored from ROUTER1)
    v_name = entities.get("visitor_name")
    if v_name and v_name.lower() not in NAME_BLACKLIST:
        new_name = v_name.capitalize()
        if state.get("visitor_name") and state["visitor_name"] != new_name:
            state["visitor_name"] = new_name
            state["identity_updated"] = True
            state["conv_state"] = State.INIT
        elif not state.get("visitor_name"):
            state["visitor_name"] = new_name

    # 3. VISITOR TYPE & ROLE DETECTION
    v_type_extracted = entities.get("visitor_type")
    if v_type_extracted:
        state["visitor_type"] = v_type_extracted

    if "interview" in query_low:
        state["visitor_type"] = "Interviewee"
        state["is_employee"] = False
    elif any(k in query_low for k in ["i work here", "staff", "employee"]):
        state["visitor_type"] = "Internal Staff"
        state["is_employee"] = True

    # 4. ATTENDEE LOOP KILLER (Restored from ROUTER1)
    if any(p in query_low for p in ["only me", "just us", "no one else", "just me"]):
        state["attendees_finalized"] = True

    # 5. HOST & SCHEDULING DATA
    if entities.get("employee_name"):
        state["meeting_with_raw"] = entities["employee_name"]
        state["sched_employee_raw"] = entities["employee_name"]

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


def _commit_checkin(state: Dict[str, Any], user_query: str, client_id: str) -> bool:
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

        reason = state.get("purpose") or "Office Visit"
        v_type = state.get("visitor_type", "Visitor/Guest")
        db_notes = f"Visit Type: {v_type} | Reason: {reason}"
        if host_raw:
            db_notes += f" | Requested: {host_raw}"

        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=emp.id if emp else None,
            person_type=v_type,
            check_in_time=datetime.utcnow(),
            purpose=reason,
            notes=db_notes,
        )
        db.add(log)
        db.commit()

        # SLACK NOTIFICATION
        if emp:
            send_slack_arrival(
                emp.name, v_name, state["visitor_type"], reason, state["session_id"]
            )
        elif "admin" in user_query.lower() or "ac" in user_query.lower():
            send_slack_arrival(
                "Admin Team", v_name, "Inquiry", reason, state["session_id"]
            )

        return True
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        return False
    finally:
        db.close()


def _commit_meeting_to_db(state: Dict[str, Any]) -> bool:
    """Restored from ROUTER1, but upgraded with Slack session_id fix from router."""
    try:
        emp_name = state.get("sched_employee_name")
        emp_email = state.get("sched_employee_email")
        date_str = state.get("sched_date")
        time_str = state.get("sched_time")
        purpose = state.get("sched_purpose") or "Meeting"
        visitor_name = state.get("visitor_name") or "Guest"
        visitor_type = state.get("visitor_type", "Visitor/Guest")
        session_id = state.get("session_id")

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
    situation: str,
    state: Dict[str, Any],
    user_query: str = None,
    client_id: str = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    llm = GroqProcessor.get_instance()
    ctx = context if context else state

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
    1. If an item in 'KNOWLEDGE BASE' is NOT 'UNKNOWN' or 'NOT SET', NEVER ask the user for it.
    2. If the user is Staff/Employee, be respectful and skip the 'Welcome to Sharp Software' robotic intro.
    3. If 'TASK' asks for info you already have, ignore that part of the task.
    4. USER SAID: "{user_query}"
    5. GOAL: {situation}
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
            state["sched_employee_email"] = emp.email
        else:
            state["sched_employee_name"] = state["sched_employee_raw"]

    # ATTENDEE LOOP KILLER (Restored from ROUTER1)
    if not state.get("attendees_finalized"):
        return await _llm_reply(
            f"I've noted the meeting with {state['sched_employee_name']}. Will anyone else be joining you?",
            state,
            user_query,
            client_id,
        )

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
        if _commit_meeting_to_db(state):
            state["scheduling_active"] = False
            state["conv_state"] = State.COMPLETED
            return await _llm_reply(
                "Confirm the booking is successful.", state, user_query, client_id
            )
        else:
            return await _llm_reply(
                "Apologize and say there was an error booking the meeting.",
                state,
                user_query,
                client_id,
            )

    state["sched_pending_confirm"] = True
    return await _llm_reply(
        "Summarize details and ask for confirmation.", state, user_query, client_id
    )


async def route_query(client_id: str, user_query: str) -> str:
    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_clean = user_query.lower().strip()

    # 1. WAKE WORD TRIGGER
    if user_query == "WAKE_WORD_TRIGGERED" or any(
        x in query_clean for x in ["hey jarvis", "hi jarvis"]
    ):
        clear_session_state(client_id)
        state = get_session_state(client_id)
        return f"Welcome to {COMPANY_NAME}. I am {AI_NAME}, how can I assist you today?"

    # 2. TERMINAL HANDLING
    if any(
        w in query_clean
        for w in ["thank you", "thanks", "bye", "goodbye", "that's all"]
    ):
        reply = await llm.get_raw_response(
            f"The user said '{user_query}' after a successful interaction. Give a warm closing.",
            client_id,
        )
        clear_session_state(client_id)
        return reply

    # 3. NLU EXTRACTION & MERGING
    extracted = await llm.extract_intent_and_entities(user_query)
    entities = extracted.get("entities", {})
    intent = extracted.get("intent", "general")

    _merge_checkin_entities(state, entities, user_query, intent)

    # 4. FLOW ROUTING
    if intent == "employee_lookup" or any(
        x in query_clean for x in ["who is", "where is"]
    ):
        return await _handle_directory_lookup(client_id, user_query, state)

    if intent == "schedule_meeting" or state["scheduling_active"]:
        state["scheduling_active"] = True
        return await _handle_scheduling(client_id, user_query, state, entities, intent)

    if state["conv_state"] == State.COMPLETED:
        return await llm.get_response(
            client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
        )

    if intent == "check_in" or state["meeting_with_raw"] or state["is_delivery"]:
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

        host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
        return await _llm_reply(
            f"Acknowledge their arrival. Confirm you have notified {host} via Slack. Do not ask them who they are here to see again.",
            state,
            user_query,
            client_id,
        )

    if not state.get("visitor_name"):
        return await _llm_reply(
            "Ask for their name politely.", state, user_query, client_id
        )

    if not state.get("meeting_with_raw"):
        if any(k in user_query.lower() for k in ["ac", "temperature", "admin"]):
            state["meeting_with_raw"] = "Administration Team"
            _commit_checkin(state, user_query, client_id)
            state["conv_state"] = State.COMPLETED
            return await _llm_reply(
                "Tell them you have notified the admin team about their request.",
                state,
                user_query,
                client_id,
            )

        return await _llm_reply(
            "Ask who they are here to see today.", state, user_query, client_id
        )

    return await _llm_reply(
        "Assist the user with their check-in.", state, user_query, client_id
    )


async def _complete_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    """Finalizing step for deliveries."""
    # --- BUG FIXED: Added user_query and corrected argument order ---
    _commit_checkin(state, user_query, client_id)
    state["conv_state"] = State.COMPLETED
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    return await _llm_reply(
        f"Tell them to leave the package. You will ping {host} immediately.",
        state,
        user_query,
        client_id,
    )
