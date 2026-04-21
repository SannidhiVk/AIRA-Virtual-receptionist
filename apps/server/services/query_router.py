import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from sqlalchemy import or_, and_

from receptionist.database import SessionLocal, get_company_details
from receptionist.models import Employee, Visitor, Meeting, ReceptionLog
from models.groq_processor import GroqProcessor

logger = logging.getLogger(__name__)

AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Pvt. Ltd."

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


class State:
    INIT = "INIT"
    COLLECTING_NAME = "COLLECTING_NAME"
    COLLECTING_HOST = "COLLECTING_HOST"
    COLLECTING_PURPOSE = "COLLECTING_PURPOSE"
    COMPLETED = "COMPLETED"


_client_sessions: Dict[str, Dict[str, Any]] = {}


def get_session_state(client_id: str) -> Dict[str, Any]:
    if client_id not in _client_sessions:
        _client_sessions[client_id] = _fresh_state()
    return _client_sessions[client_id]


def clear_session_state(client_id: str, retain_name=False) -> None:
    # 1. DELETE the key entirely to ensure NO data persists
    if client_id in _client_sessions:
        del _client_sessions[client_id]

    # 2. Re-initialize a truly empty state
    _client_sessions[client_id] = _fresh_state()

    try:
        # 3. Wipe Groq History
        GroqProcessor.get_instance().reset_history(client_id)
        # 4. Wipe external pronoun context
        from client_context import clear_context

        clear_context(client_id)
        logger.info(
            f"HARD RESET: All data for {client_id} has been physically deleted."
        )
    except Exception as e:
        logger.error(f"Reset failed: {e}")


def _is_jarvis(name: str) -> bool:
    """Detects if a name is actually the AI's own name (Jarvis, Davis, Darwis, etc.)"""
    if not name:
        return False
    # Common mishearings from Whisper
    blacklist = {"jarvis", "davis", "darwis", "darvis", "jarves", "dervis", "jarvis"}
    return name.lower().strip().replace(".", "") in blacklist


def _is_self_reference(name: str) -> bool:
    """Returns True if the name sounds like the AI's own name."""
    if not name:
        return False
    self_names = {AI_NAME.lower(), "jarvis", "darwis", "darvis", "jarvis", "jarves"}
    return name.lower() in self_names


import uuid


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
    }


def _determine_visitor_type(text: str, purpose: str, current_type: str) -> str:
    combined = f"{text} {purpose}".lower()
    if re.search(r"\b(interview|candidate)\b", combined):
        return "Interviewee"
    if re.search(r"\b(swiggy|zomato|food|lunch|food delivery)\b", combined):
        return "Food Delivery"
    if re.search(r"\b(amazon|flipkart|delivery|courier|package|parcel)\b", combined):
        return "Delivery"
    if re.search(r"\b(vendor|electrician|plumber|maintenance)\b", combined):
        return "Contractor/Vendor"
    if re.search(r"\b(client|customer|demo)\b", combined):
        return "Client"
    return current_type or "Visitor/Guest"


def _is_time_close_to_now(time_str: str) -> bool:
    """Checks if a mentioned time is within 30 minutes of the current time."""
    norm = _normalize_time(time_str)
    if not norm:
        return False
    try:
        now = datetime.now()
        target_time = datetime.strptime(norm, "%H:%M").time()
        target_dt = datetime.combine(now.date(), target_time)
        # If difference is less than 30 mins (1800 seconds)
        return abs((target_dt - now).total_seconds()) < 1800
    except:
        return False


def _normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    today = datetime.now().date()
    if s in ("today", "now"):
        return today.strftime("%Y-%m-%d")
    if s == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except ValueError:
        pass
    return None


def _normalize_time(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower().replace("p.m.", "pm").replace("a.m.", "am").replace(" ", "")
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
        if emp:
            return emp
        if "manager" in clean:
            dept = clean.replace("manager", "").strip()
            return (
                db.query(Employee)
                .filter(
                    and_(
                        Employee.department.ilike(f"%{dept}%"),
                        Employee.role.ilike(f"%manager%"),
                    )
                )
                .first()
                if dept
                else None
            )
        return None
    finally:
        db.close()


def _commit_checkin(state: Dict[str, Any], client_id: str) -> bool:
    db = SessionLocal()
    try:
        # 1. Resolve Visitor (Handle cases where name might be missing for quick deliveries)
        v_name = state.get("visitor_name") or "Delivery Personnel"
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()

        # 2. Resolve Employee (The Host)
        host_raw = state.get("meeting_with_raw")
        emp = _lookup_employee(host_raw) if host_raw else None

        # 3. Determine Dynamic Notes and Purpose
        # If it's a delivery, we change the wording in the DB notes
        if state.get("is_delivery"):
            prefix = (
                "Food delivery for"
                if state.get("visitor_type") == "Food Delivery"
                else "Delivery for"
            )
            note_content = f"{prefix}: {host_raw}"
            # For deliveries, if purpose is empty, use the visitor type (e.g., "Food Delivery")
            db_purpose = state.get("purpose") or state.get("visitor_type")
        else:
            note_content = f"Meeting with: {host_raw}"
            db_purpose = state.get("purpose")

        # 4. Create the Reception Log
        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=emp.id if emp else None,
            person_type=state.get("visitor_type", "Visitor/Guest"),
            check_in_time=datetime.utcnow(),
            purpose=db_purpose,
            notes=note_content,
        )
        db.add(log)
        db.commit()

        # 5. Fire Slack/Teams Notification
        if emp:
            try:
                from services.notify_slack import send_slack_arrival

                send_slack_arrival(
                    emp.name,
                    v_name,
                    state.get("visitor_type", "Visitor"),
                    db_purpose or "General Visit",
                    session_id=state["session_id"],
                )
            except Exception as slack_err:
                logger.warning(f"Slack notification failed: {slack_err}")

        return True

    except Exception as e:
        logger.error(f"Check-in failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def _commit_meeting(
    emp_name: str,
    emp_email: Optional[str],
    date_str: str,
    time_str: str,
    purpose: str,
    org_name: str,
    org_email: str,
) -> bool:
    from receptionist.database import schedule_meeting

    # 1. Save the meeting to your local SQLite Database
    res = schedule_meeting(
        organizer_name=org_name,
        organizer_type="visitor",
        employee_name=emp_name,
        meeting_date=date_str,
        meeting_time=time_str,
        purpose=purpose,
    )

    # 2. Trigger the Google Calendar API
    if res and emp_email:
        from services.calendar_service import send_calendar_invite

        try:
            meeting_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            send_calendar_invite(
                visitor_name=org_name,
                employee_email=emp_email,
                dt=meeting_dt,
            )
        except Exception as e:
            logger.error("Failed to send calendar invite: %s", e)

        import asyncio
        from services.notification_service import send_meeting_notification

        asyncio.create_task(
            send_meeting_notification(
                employee_name=emp_name,
                employee_email=emp_email,
                organizer_name=org_name,
                meeting_date=date_str,
                meeting_time=time_str,
                purpose=purpose or "Not specified",
            )
        )

    return res is not None


def _clean_entity(val: Any) -> Optional[str]:
    s = str(val).strip() if val else ""
    return (
        s
        if s and s.lower() not in ("null", "none", "") and s.lower() not in PRONOUNS
        else None
    )


DELIVERY_SERVICES = {
    "swiggy",
    "zomato",
    "amazon",
    "flipkart",
    "courier",
    "fedex",
    "dhl",
    "uber eats",
    "dunzo",
}


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], client_id: str, user_query: str
) -> None:
    # 1. Detect if the speaker is an employee
    query_low = user_query.lower()
    employee_keywords = {
        "manager",
        "director",
        "employee",
        "staff",
        "intern",
        "i work here",
    }
    if any(k in query_low for k in employee_keywords) and "i am" in query_low:
        state["is_employee"] = True
        state["visitor_type"] = "Internal Staff"

    # 2. Update Name (Filter out misheard 'Travis/Jarvis')
    v_name = _clean_entity(entities.get("visitor_name"))
    if v_name and not _is_jarvis(v_name):
        state["visitor_name"] = v_name.capitalize()

    # 3. Update Host (Filter out 'Travis/Davis')
    target = _clean_entity(entities.get("employee_name")) or _clean_entity(
        entities.get("role")
    )
    if target and not _is_jarvis(target) and target.lower() not in PRONOUNS:
        state["meeting_with_raw"] = target
        state["meeting_with_resolved"] = None

    # 4. Delivery/Visitor Reset
    raw_p = _clean_entity(entities.get("purpose"))
    if not state.get("is_employee"):
        state["visitor_type"] = _determine_visitor_type(
            user_query, str(raw_p or ""), state["visitor_type"]
        )
        state["is_delivery"] = (
            True if state["visitor_type"] in ["Delivery", "Food Delivery"] else False
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX A — Upgraded _llm_reply with full context-aware prompt
# Now passes everything Jarvis already knows so it never asks twice.
# ─────────────────────────────────────────────────────────────────────────────
async def _llm_reply(
    situation: str,
    state: dict[str, Any],
    context: dict,
    user_query: str = None,
    client_id: str = None,
) -> str:

    llm = GroqProcessor.get_instance()

    is_emp = "EMPLOYEE" if state.get("is_employee") else "VISITOR"
    current_time = datetime.now().strftime("%I:%M %p")

    # 1. Prepare "Known Info" block for the AI
    visitor_name = context.get("visitor_name")
    host_name = context.get("employee_name")
    purpose = context.get("purpose")

    known_info = []
    if visitor_name:
        known_info.append(f"- Visitor Name: {visitor_name}")
    if host_name:
        known_info.append(f"- Meeting With: {host_name}")
    if purpose:
        known_info.append(f"- Purpose: {purpose}")
    if state.get("is_employee"):
        known_info.append("- User Type: Employee/Staff")

    info_block = "\n".join(known_info) if known_info else "- No info yet."

    prompt = f"""
    You are {AI_NAME}, the professional AI receptionist at {COMPANY_NAME}.
    Current Time: {current_time}

    KNOWLEDGE:
    {info_block}

    TASK:
    {situation}

    STRICT OPERATIONAL RULES:
    1. NEVER address the user as 'Visitor'. If name is unknown, use no title at all.
    2. NEVER start with 'Welcome to Sharp Software' unless this is the very first greeting.
    3. Keep responses warm but concise. No filler like 'Certainly' or 'Absolutely'.
    4. GOODBYE LOGIC: Only say goodbye or 'safe trip home' if the visitor is clearly EXITING. 
    5. FACILITY LOGIC: If a visitor asks for a washroom or water, give directions and wait for them to finish. Do NOT end the conversation.
    6. DELIVERY LOGIC: If you do not see a 'Meeting With' name in the knowledge block above, you MUST ask 'Who is the delivery for?'. Do NOT guess or assume the host is Jarvis or Davis.
    7. MISPRONUNCIATION: If the visitor says 'Darwis', 'Davis', or 'Jarves', they are talking to YOU. Ignore it and do not treat those names as the host.
    8. NO REPETITION: Do not ask for information that is already listed in the 'Knowledge' block above.

    THE VISITOR'S LAST UTTERANCE:
    "{user_query if user_query else 'Just walked up'}"
    """

    try:
        response = await llm.get_raw_response(prompt, client_id=client_id)
        return response.strip().replace('"', "")

    except Exception as e:
        logger.error(f"Error generating LLM reply: {e}")
        return (
            "I'm sorry, I'm having a bit of trouble. Could you please say that again?"
        )


async def _handle_scheduling(
    client_id: str,
    user_query: str,
    state: Dict[str, Any],
    entities: Dict[str, Any],
    intent: str,
) -> str:
    # ... (Keep existing entity capture logic at the top of this function) ...
    target = _clean_entity(entities.get("employee_name")) or _clean_entity(
        entities.get("role")
    )
    if target and not state["sched_employee_raw"]:
        state["sched_employee_raw"] = target
    if entities.get("date") and not state["sched_date"]:
        state["sched_date"] = _normalize_date(str(entities["date"]))
    if entities.get("time") and not state["sched_time"]:
        state["sched_time"] = _normalize_time(str(entities["time"]))
    if _clean_entity(entities.get("purpose")) and not state["sched_purpose"]:
        state["sched_purpose"] = _clean_entity(entities["purpose"])

    # ... (Keep existing missing info checks for name, host, date, time, purpose) ...
    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name.", state, user_query, client_id)
    if not state["sched_employee_raw"]:
        return await _llm_reply(
            "Ask who they want to meet.", state, user_query, client_id
        )

    if not state["sched_employee_name"]:
        emp = _lookup_employee(state["sched_employee_raw"])
        if not emp:
            state["sched_employee_raw"] = None
            return await _llm_reply(
                "Tell them host not found.", state, user_query, client_id
            )
        state["sched_employee_name"], state["sched_employee_email"] = emp.name, getattr(
            emp, "email", None
        )

    if not state["sched_date"]:
        return await _llm_reply("Ask for date.", state, user_query, client_id)
    if not state["sched_time"]:
        return await _llm_reply("Ask for time.", state, user_query, client_id)
    if not state["sched_purpose"]:
        return await _llm_reply("Ask for purpose.", state, user_query, client_id)

    # --- UPDATED CONFIRMATION LOGIC ---
    is_yes = intent == "confirm" or any(
        w in user_query.lower().split()
        for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "confirm"]
    )

    # If they already said yes/confirm while we were waiting for confirmation
    if is_yes and state["sched_pending_confirm"]:
        success = _commit_meeting(
            state["sched_employee_name"],
            state["sched_employee_email"],
            state["sched_date"],
            state["sched_time"],
            state["sched_purpose"],
            state.get("visitor_name") or "Visitor",
            state.get("visitor_email"),
        )
        state["scheduling_active"] = False
        state["conv_state"] = State.COMPLETED
        return await _llm_reply(
            "Confirm the meeting is scheduled.",
            {
                "visitor_name": state.get("visitor_name"),
                "employee_name": state["sched_employee_name"],
            },
            state,
            user_query,
            client_id,
        )

    if not state["sched_pending_confirm"]:
        state["sched_pending_confirm"] = True
        return await _llm_reply(
            "Read back details and ask to confirm.",
            {
                "visitor_name": state.get("visitor_name"),
                "employee_name": state["sched_employee_name"],
                "date": state["sched_date"],
                "time": state["sched_time"],
                "purpose": state["sched_purpose"],
            },
            state,
            user_query,
            client_id,
        )

    # Fallback for "No" or "Cancel"
    if intent == "cancel" or any(
        w in user_query.lower().split() for w in ["no", "cancel", "nevermind"]
    ):
        state["scheduling_active"] = False
        return await _llm_reply(
            "Tell them you cancelled the request.",
            {"visitor_name": state.get("visitor_name")},
            state,
            user_query,
            client_id,
        )

    return await _llm_reply(
        "Ask clearly if you should log the meeting.",
        {"visitor_name": state.get("visitor_name")},
        state,
        user_query,
        client_id,
    )


def _get_full_company_info(state: Dict[str, Any]) -> Dict[str, Any]:
    db_details = get_company_details()

    # Fallback to the hardcoded name if the database is empty
    return {
        "company_name": db_details.get("name") or COMPANY_NAME,
        "company_address": db_details.get("address"),
        "company_phone": db_details.get("phone"),
        "company_email": db_details.get("email"),
        "company_website": db_details.get("website"),
        "visitor_name": state.get("visitor_name") or "Visitor",
    }


async def route_query(client_id: str, user_query: str) -> str:
    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_clean = user_query.lower().strip()

    # 1. WAKE WORD RESET (Travis/Davis/Darwis)
    if (user_query == "WAKE_WORD_TRIGGERED") or re.match(
        r"^(hey jarvis|hi jarvis|jarvis|hey travis|hey davis)\b", query_clean
    ):
        clear_session_state(client_id)
        state = get_session_state(client_id)
        return f"Welcome to {COMPANY_NAME}. I am {AI_NAME}, how can I assist you today?"

    # 2. INTENT EXTRACTION
    extracted = await llm.extract_intent_and_entities(user_query)
    entities, intent = extracted.get("entities", {}), extracted.get(
        "intent", "general_conversation"
    )
    _merge_checkin_entities(state, entities, client_id, user_query)

    # 3. EMPLOYEE FLOW (Priya doesn't need to 'check in')
    if state.get("is_employee"):
        if intent == "schedule_meeting" or state.get("scheduling_active"):
            state["scheduling_active"] = True
            return await _handle_scheduling(
                client_id, user_query, state, entities, intent
            )
        # If an employee asks for deliveries
        if "delivery" in query_clean or "parcel" in query_clean:
            return await _llm_reply(
                "Tell them you are checking the delivery log for their name.",
                state,
                user_query,
                client_id,
            )

    # 4. ABUSE FILTER (Stay professional, then stop)
    curse_words = {"bitch", "dumbass", "gay", "stupid", "shut up"}
    if any(w in query_clean for w in curse_words):
        return "I am programmed to be a professional assistant. Please let me know if you have a business-related request, otherwise, I will end this interaction."

    # 5. VISITOR FLOW
    if intent == "check_in" and not state.get("is_employee"):
        return await _advance_checkin(state, user_query, client_id)

    # 6. DEFAULT
    return await llm.get_response(
        client_id, user_query, company_info=_get_full_company_info(state)
    )


async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    if state["meeting_with_raw"] and _is_jarvis(state["meeting_with_raw"]):
        state["meeting_with_raw"] = None
    # Resolve host if we have a name
    if state["meeting_with_raw"] and not state["meeting_with_resolved"]:
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name

    # 1. Delivery Flow
    if state.get("is_delivery"):
        if not state.get("meeting_with_raw"):
            return await _llm_reply(
                "Ask who the delivery is for. Do not guess a name.",
                state,
                user_query,
                client_id,
            )
        return await _complete_checkin(state, user_query, client_id)

    # 2. Standard Visitor Flow
    if not state.get("visitor_name"):
        return await _llm_reply(
            "Ask for the visitor's name.", state, user_query, client_id
        )

    if not state.get("meeting_with_raw"):
        return await _llm_reply(
            "Ask who they are here to see.", state, user_query, client_id
        )

    return await _complete_checkin(state, user_query, client_id)


async def _complete_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    # Fire the database log
    _commit_checkin(state, client_id)
    state["conv_state"] = State.COMPLETED

    host = state.get("meeting_with_resolved") or state.get("meeting_with_raw")

    if state.get("is_delivery"):
        return await _llm_reply(
            f"Tell them to leave the package. You will ping {host}.",
            state,
            user_query,
            client_id,
        )

    return await _llm_reply(
        f"Confirm check-in. Notify {host}. Direct them to the correct floor.",
        state,
        user_query,
        client_id,
    )
