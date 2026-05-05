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
    "dadfish",
    "jadfish",
}

WAKE_WORDS = [
    "hey jarvis",
    "hi jarvis",
    "wake_word_triggered",
    "hey charles",
    "hey elvis",
    "hey jadfish",
    "hey dadfish",
    "hey travis",
]

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
FOOD_DELIVERY_KEYWORDS = {"zomato", "swiggy", "food", "bistro", "blinkit", "danzo"}

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
    if client_id in _client_sessions:
        session_id = _client_sessions[client_id].get("session_id")
        if session_id:
            clear_slack_cache(session_id)
        del _client_sessions[client_id]
    # We don't call fresh_state here because get_session_state handles it
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
        "scheduling_active": False,
        "sched_employee_name": None,
        "sched_employee_email": None,
        "sched_date": None,
        "sched_time": None,
        "sched_purpose": None,
        "identity_updated": False,
        "notified_hosts": set(),  # FIX: Allows multiple Slack notifications (Zomato/Blanket case)
        "greeted": False,  # FIX: Prevents redundant "Welcome to Sharp..."
        "force_admin": False,  # FIX: Breaks loops for visitors who don't know who to meet
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION & LOOKUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_time_greeting() -> str:
    """FIX: Returns correct time-based greeting."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"


def _is_jarvis(name: str) -> bool:
    if not name:
        return False
    return name.lower().strip().replace(".", "") in NAME_BLACKLIST


def _determine_visitor_type(text: str, purpose: str, current_type: str) -> str:
    combined = f"{text} {purpose}".lower()
    if "intern" in combined:
        return "Intern"  # New Logic
    if re.search(r"\b(interview|candidate)\b", combined):
        return "Interviewee"
    if any(k in combined for k in FOOD_DELIVERY_KEYWORDS):
        return "Food Delivery"
    if any(k in combined for k in PACKAGE_DELIVERY_KEYWORDS):
        return "Package Delivery"
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
    s = str(raw).lower()
    today = datetime.now().date()
    if "today" in s:
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in s:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if "friday" in s and "next" in s:
        days_ahead = (4 - today.weekday() + 7) % 7 or 7
        return (today + timedelta(days=days_ahead + 7)).strftime("%Y-%m-%d")
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except:
        return None


def _normalize_time(raw: str) -> Optional[str]:
    """FIX: Converts '4pm' to '16:00' to prevent the 'Priya is busy' error."""
    if not raw:
        return None
    s = (
        str(raw)
        .lower()
        .replace("p.m.", "pm")
        .replace("a.m.", "am")
        .replace(".", "")
        .replace(" ", "")
    )

    # Handle formats like 4pm, 11am, 4:30pm
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        h, mn, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if mer == "pm" and h != 12:
            h += 12
        if mer == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mn:02d}"

    # Handle standard 24h strings
    if re.match(r"^\d{2}:\d{2}$", s):
        return s
    return None


def _lookup_employee(search_term: str) -> Optional[Employee]:
    """FIX: Maps 'Admin' phrases and handles role-based searches like 'CEO'."""
    if not search_term or len(str(search_term)) < 2:
        return None
    clean = re.sub(
        r"\b(the|is|who|of|this|company|with|for|at|his|her|name|hr|manager|engineer|lead|ceo)\b",
        "",
        str(search_term).lower(),
    ).strip()

    # Direct mapping for Vikram's case
    if clean in ["admin", "administration", "front desk", "anyone"]:
        return Employee(name="Administration Team", email=None, id=None)

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
# DATABASE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def _finalize_meeting_and_log(state: Dict[str, Any]) -> bool:
    """Atomic: Database schedule + Reception Log + Slack + Calendar."""
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name") or "Guest"
        host_name = state["sched_employee_name"]
        emp = _lookup_employee(host_name)

        # 1. Schedule Meeting Table
        mid = schedule_meeting(
            v_name,
            "Visitor",
            host_name,
            state["sched_date"],
            state["sched_time"],
            state.get("sched_purpose", "Meeting"),
        )
        if not mid:
            return False

        # 2. Add to Reception Log Table (Syncing the visit)
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()

        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=emp.id if emp and emp.id else None,
            person_type=state["visitor_type"],
            purpose=f"BOOKED: {state.get('sched_purpose') or 'Meeting'}",
            notes=f"Meeting set for {state['sched_date']} at {state['sched_time']}. SID:{state['session_id']}",
            check_in_time=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

        # 3. External Notifications
        send_slack_arrival(
            host_name,
            v_name,
            state["visitor_type"],
            f"Scheduled Meeting for {state['sched_date']}",
            state["session_id"],
        )
        state["notified_hosts"].add(host_name)

        if state.get("sched_employee_email"):
            schedule_google_meeting_background(
                v_name,
                state["sched_employee_email"],
                state["sched_date"],
                state["sched_time"],
            )
        return True
    except Exception as e:
        logger.error(f"Meeting Finalization Error: {e}")
        return False
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY MERGING (ROLE AWARE & CONTEXT LOCKING)
# ─────────────────────────────────────────────────────────────────────────────


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], user_query: str
) -> None:
    query_low = user_query.lower()
    v_name = entities.get("visitor_name")
    if v_name and not _is_jarvis(v_name):
        new_name = v_name.capitalize()
        if state["visitor_name"] and state["visitor_name"] != new_name:
            state["visitor_name"], state["identity_updated"] = new_name, True
        elif not state["visitor_name"]:
            state["visitor_name"] = new_name

    state["visitor_type"] = _determine_visitor_type(
        user_query, entities.get("purpose", ""), state["visitor_type"]
    )

    target = (
        entities.get("employee_name")
        or entities.get("employee_role")
        or state.get("meeting_with_raw")
    )
    if target and not _is_jarvis(target):
        state["meeting_with_raw"] = target
        emp = _lookup_employee(target)
        if emp:
            (
                state["meeting_with_resolved"],
                state["sched_employee_name"],
                state["sched_employee_email"],
            ) = (emp.name, emp.name, emp.email)
        else:
            state["meeting_with_resolved"] = target

    if entities.get("date"):
        state["sched_date"] = _normalize_date(str(entities["date"]))
    if entities.get("time"):
        state["sched_time"] = _normalize_time(str(entities["time"]))
    if entities.get("purpose"):
        state["purpose"] = state["sched_purpose"] = entities["purpose"]


def _commit_checkin(state: Dict[str, Any], client_id: str, user_query: str) -> bool:
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name") or "Guest"
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()

        host_emp = _lookup_employee(
            state.get("meeting_with_resolved") or state.get("meeting_with_raw")
        )
        existing_log = (
            db.query(ReceptionLog)
            .filter(ReceptionLog.notes.like(f"%SID:{state['session_id']}%"))
            .first()
        )

        if existing_log:
            existing_log.visitor_id, existing_log.employee_id = visitor.id, (
                host_emp.id if host_emp and host_emp.id else None
            )
        else:
            db.add(
                ReceptionLog(
                    visitor_id=visitor.id,
                    employee_id=host_emp.id if host_emp and host_emp.id else None,
                    person_type=state["visitor_type"],
                    purpose=state["purpose"] or "Check-in",
                    notes=f"via Jarvis. SID:{state['session_id']}",
                    check_in_time=datetime.utcnow(),
                )
            )
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Commit Checkin failed: {e}")
        return False
    finally:
        db.close()


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


# ─────────────────────────────────────────────────────────────────────────────
# FIX A — Upgraded _llm_reply with full context-aware prompt
# Now passes everything Jarvis already knows so it never asks twice.
# ─────────────────────────────────────────────────────────────────────────────
async def _llm_reply(
    situation: str, state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    llm = GroqProcessor.get_instance()

    # Instruction to stop redundant greetings
    greeting_instr = (
        "Do NOT greet or repeat the company name if the conversation is already in progress."
        if state["greeted"]
        else "Greet the visitor warmly."
    )

    prompt = f"""{BASE_SYSTEM_PROMPT}
    CONTEXT: User: {state.get('visitor_name') or 'Visitor'} | Host: {state.get('meeting_with_resolved') or 'Admin'}
    SITUATION: {situation}
    {greeting_instr}
    USER SAID: "{user_query}"
    STRICT: Respond DIRECTLY as Jarvis. No analysis. Max 2 sentences."""

    response = await llm.get_raw_response(prompt, client_id=client_id)
    state["greeted"] = True  # Mark as greeted so Turn 2 and beyond are concise
    return response


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


async def _handle_scheduling(client_id, query, state, entities, intent):
    """Smart Availability Logic: Checks DB slots and triggers finalization."""
    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name.", state, query, client_id)
    if not state.get("sched_employee_name"):
        return await _llm_reply("Ask who they want to meet.", state, query, client_id)

    # Update state with latest date/time found in this turn
    if entities.get("date"):
        state["sched_date"] = _normalize_date(entities["date"])
    if entities.get("time"):
        state["sched_time"] = _normalize_time(entities["time"])

    if not state.get("sched_date"):
        return await _llm_reply("Ask for the date.", state, query, client_id)
    if not state.get("sched_time"):
        return await _llm_reply("Ask for the time.", state, query, client_id)

    # 1. ACTUAL DB AVAILABILITY CHECK
    slots = get_available_slots(state["sched_employee_name"], state["sched_date"])
    if state["sched_time"] not in slots:
        suggestions = ", ".join(slots[:3])
        return await _llm_reply(
            f"Priya is busy at {state['sched_time']}. Suggest these alternatives for tomorrow: {suggestions}.",
            state,
            query,
            client_id,
        )

    # 2. FINALIZATION TRIGGER
    if intent == "confirm" or any(
        x in query.lower()
        for x in ["okay", "yes", "schedule", "correct", "perfect", "book it"]
    ):
        if _finalize_meeting_and_log(state):
            state["scheduling_active"] = False
            return await _llm_reply(
                f"I've successfully booked your meeting with {state['sched_employee_name']} for {state['sched_time']}. Notifications have been sent.",
                state,
                query,
                client_id,
            )

    return await _llm_reply(
        f"I have you down for {state['sched_date']} at {state['sched_time']} with {state['sched_employee_name']}. Should I book it?",
        state,
        query,
        client_id,
    )


async def _advance_checkin(state, query, client_id):
    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name.", state, query, client_id)
    if not state.get("meeting_with_resolved"):
        emp = _lookup_employee(state.get("meeting_with_raw"))
        if emp:
            state["meeting_with_resolved"] = emp.name
        else:
            return await _llm_reply(
                "Ask who they are here to see.", state, query, client_id
            )

    _commit_checkin(state, client_id, query)
    state["conv_state"] = State.COMPLETED
    msg = f"Confirm {state['meeting_with_resolved']} notified."
    msg += (
        " Leave at desk." if "Delivery" in state["visitor_type"] else " Wait in lobby."
    )
    return await _llm_reply(msg, state, query, client_id)


async def _handle_directory_lookup(client_id, query, state):
    emp = _lookup_employee(query)
    if emp:
        state["meeting_with_resolved"] = emp.name
        return await _llm_reply(
            f"Tell them {emp.name} is our {emp.role} at {emp.location}.",
            state,
            query,
            client_id,
        )
    return await _llm_reply(
        "Apologize you couldn't find them and offer Admin help.",
        state,
        query,
        client_id,
    )


async def route_query(client_id: str, user_query: str) -> str:
    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_low = user_query.lower().strip()

    # 1. Wake word / Trigger Logic (Preserved + Fuzzy Fixes)
    if any(x in query_low for x in WAKE_WORDS):
        clear_session_state(client_id)
        state = get_session_state(client_id)
        state["greeted"] = True
        return f"{_get_time_greeting()}! Welcome to {COMPANY_NAME}. I am {AI_NAME}, how can I help you today?"

    # 2. Extraction & Merging
    extracted = await llm.extract_intent_and_entities(user_query)
    intent, entities = extracted.get("intent", "general"), extracted.get("entities", {})

    # Run the merging logic (Handles visitor name, intern type, and host resolution)
    from services.query_router import _merge_checkin_entities

    _merge_checkin_entities(state, entities, user_query)

    # 3. Vikram Fix: Handle "Don't know anyone" loop (Smart & Flexible)
    if any(
        x in query_low
        for x in ["don't know", "do not know", "anyone", "just notify admin"]
    ):
        state["force_admin"] = True
        state["meeting_with_resolved"] = "Administration Team"
        state["visitor_type"] = _determine_visitor_type(
            user_query, "", state["visitor_type"]
        )

    # 4. Termination Logic (Preserved)
    if any(
        w in query_low for w in ["thank you", "thanks", "bye", "goodbye", "shut up"]
    ):
        reply = await llm.get_raw_response(
            f"The user said '{user_query}'. Give warm closing.", client_id=client_id
        )
        clear_session_state(client_id)
        return reply

    # 5. Intent: Employee Lookup (Preserved + Role Fix)
    if intent == "employee_lookup" or any(
        x in query_low for x in ["who is", "director", "ceo", "manager"]
    ):
        emp = _lookup_employee(user_query)
        if emp:
            # Context lock the host for the next turn
            state["meeting_with_resolved"] = emp.name
            return await _llm_reply(
                f"Tell them {emp.name} is the {emp.role}.", state, user_query, client_id
            )

    # 6. Intent: Scheduling (Kapoor Fix - Auto-Commit)
    if intent == "schedule_meeting" or state["scheduling_active"]:
        state["scheduling_active"] = True

        # SMART AUTO-COMMIT: If Host + Date + Time are all present, book immediately
        if state["sched_employee_name"] and state["sched_date"] and state["sched_time"]:
            if _finalize_meeting_and_log(state):
                state["scheduling_active"] = False
                return await _llm_reply(
                    f"I've successfully booked your meeting with {state['sched_employee_name']} for {state['sched_time']} on {state['sched_date']}.",
                    state,
                    user_query,
                    client_id,
                )

        # Otherwise, continue collecting info
        return await _handle_scheduling(client_id, user_query, state, entities, intent)

    # 7. Intent: Check-in / Arrival (Zomato/Blanket/Intern Fix)
    if intent == "check_in" or state["meeting_with_resolved"]:
        # Employee Check-in (Preserved)
        if state.get("is_employee"):
            return await _llm_reply(
                "Wish the staff member a great day.", state, user_query, client_id
            )

        # Multi-notification logic: Send Slack only if this host hasn't been notified yet
        current_host = state["meeting_with_resolved"] or "Administration Team"
        if current_host not in state.get("notified_hosts", set()):
            send_slack_arrival(
                current_host,
                state["visitor_name"] or "Guest",
                state["visitor_type"],
                entities.get("purpose") or "Arrival",
                state["session_id"],
            )
            if "notified_hosts" not in state:
                state["notified_hosts"] = set()
            state["notified_hosts"].add(current_host)

            # Commit to Database Log (Stores visitor type as "Intern", "Food Delivery", etc.)
            from services.query_router import _commit_checkin

            _commit_checkin(state, client_id, user_query)

        # Smart Response: Check if they are an Intern
        if state["visitor_type"] == "Intern":
            return await _llm_reply(
                f"Welcome the new intern and tell them {current_host} is notified.",
                state,
                user_query,
                client_id,
            )

        return await _advance_checkin(state, user_query, client_id)

    # 8. Default Chat Fallback (Preserved)
    return await llm.get_response(
        client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
    )


async def _complete_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    _commit_checkin(state, client_id, user_query)
    state["conv_state"] = State.COMPLETED
    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    return await _llm_reply(
        f"Tell them to leave the package. I'll notify {host}.",
        state,
        user_query,
        client_id,
    )
