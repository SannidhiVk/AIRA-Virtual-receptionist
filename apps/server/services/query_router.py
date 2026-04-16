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
    old_state = _client_sessions.get(client_id, {})
    
    # 1. Reset the local Python state
    _client_sessions[client_id] = _fresh_state()
    
    if retain_name:
        # SOFT RESET: Keep the person's identity but clear the task
        _client_sessions[client_id]["visitor_name"] = old_state.get("visitor_name")
        _client_sessions[client_id]["visitor_email"] = old_state.get("visitor_email")
        _client_sessions[client_id]["visitor_type"] = old_state.get("visitor_type")
        _client_sessions[client_id]["meeting_with_raw"] = old_state.get("meeting_with_raw")
        _client_sessions[client_id]["meeting_with_resolved"] = old_state.get("meeting_with_resolved")
    else:
        # HARD RESET: Wipe everything
        try:
            # A. Wipe LLM History
            GroqProcessor.get_instance().reset_history(client_id)
            
            # B. WIPE CLIENT CONTEXT (This stops the "Delhi engineers" leak)
            from client_context import set_last_employee
            # We overwrite the context with None/Empty values
            set_last_employee(client_id, name=None, role=None, email=None, cabin=None, department=None)
            
            logger.info(f"Hard reset: Cleared session, LLM history, and Client Context for {client_id}")
        except Exception as e:
            logger.error(f"Error during hard reset: {e}")

def _fresh_state() -> Dict[str, Any]:
    return {
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


def _commit_checkin(state: Dict[str, Any]) -> bool:
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
        from services.calendar_service import schedule_google_meeting_background

        schedule_google_meeting_background(
            visitor_name=org_name,
            employee_email=emp_email,
            date_str=date_str,
            time_str=time_str,
        )

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
    name, email = _clean_entity(entities.get("visitor_name")), _clean_entity(
        entities.get("email")
    )
    if name:
        state["visitor_name"] = name.capitalize()
    if email:
        state["visitor_email"] = email

    query_lower = user_query.lower()
    for service in DELIVERY_SERVICES:
        if service in query_lower and not state.get("visitor_name"):
            state["visitor_name"] = service.capitalize()

    target = _clean_entity(entities.get("employee_name")) or _clean_entity(
        entities.get("role")
    )
    if target and not state["meeting_with_raw"]:
        state["meeting_with_raw"] = target

    # --- AUTO-PURPOSE LOGIC ---
    raw_purpose = _clean_entity(entities.get("purpose"))
    state["visitor_type"] = _determine_visitor_type(
        user_query, str(raw_purpose or ""), state["visitor_type"]
    )

    if not state.get("purpose"):
        if state["visitor_type"] == "Interviewee":
            state["purpose"] = "Job Interview"
        elif state["visitor_type"] == "Contractor/Vendor":
            state["purpose"] = raw_purpose or "Maintenance/Repair Work"
        elif state["visitor_type"] in ["Delivery", "Food Delivery"]:
            state["purpose"] = state["visitor_type"]
        elif raw_purpose:
            state["purpose"] = raw_purpose.capitalize()
    # --------------------------

    if state["visitor_type"] in ["Delivery", "Food Delivery"]:
        state["is_delivery"] = True

    if (
        str(entities.get("employee_name") or "").strip().lower() in PRONOUNS
        and client_id
    ):
        from client_context import get_last_employee_name

        resolved = get_last_employee_name(client_id)
        if resolved and not state["meeting_with_raw"]:
            state["meeting_with_raw"] = resolved


# ─────────────────────────────────────────────────────────────────────────────
# FIX A — Upgraded _llm_reply with full context-aware prompt
# Now passes everything Jarvis already knows so it never asks twice.
# ─────────────────────────────────────────────────────────────────────────────
async def _llm_reply(
    situation: str, context: dict, user_query: str = None, client_id: str = None
) -> str:
    llm = GroqProcessor.get_instance()
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    known_info = []
    for k, v in context.items():
        if v:
            known_info.append(f"- {k.replace('_', ' ').capitalize()}: {v}")

    known_block = "\n".join(known_info) if known_info else "- None collected yet"

    prompt = f"""You are {AI_NAME}, the intelligent AI receptionist at {COMPANY_NAME}.
CURRENT DATE & TIME: {current_time}

WHAT YOU ALREADY KNOW ABOUT THIS VISITOR:
{known_block}

YOUR TASK RIGHT NOW:
{situation}

RULES:
1. NEVER start with a robotic 'Welcome to Sharp Software Development India'. 
2. NEVER address the person as 'Visitor'. If you don't know their name, don't use a label.
3. Keep greetings natural like a human (e.g., 'Hi there!', 'Hello! How can I help you?').
4. If a visitor is frustrated, prioritize finishing the task over asking questions.
5. Do not use filler words like "Certainly!", "Absolutely!", "Of course!" — just respond naturally.
6. If the visitor is leaving (saying bye, thanks, see you), say goodbye warmly and do not ask anything else.
7. If the visitor refuses to give their name or is getting frustrated (e.g., "Shut up", "doesn't matter"), 
   immediately stop asking questions and provide the final instruction (like leaving the package).
8. If the visitor already mentioned a name (e.g., "Priya"), DO NOT ask for that name again.
"""
    if user_query:
        prompt += f'\nTHE VISITOR JUST SAID:\n"{user_query}"\n'

    return await llm.get_raw_response(prompt, client_id=client_id)


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
        return await _llm_reply(
            "Ask the visitor for their name.", {}, user_query, client_id
        )
    if not state["sched_employee_raw"]:
        return await _llm_reply(
            "Ask who they want to meet.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id,
        )
    if not state["sched_employee_name"]:
        emp = _lookup_employee(state["sched_employee_raw"])
        if not emp:
            state["sched_employee_raw"] = None
            return await _llm_reply(
                "Tell them host not found.",
                {"visitor_name": state.get("visitor_name")},
                user_query,
                client_id,
            )
        state["sched_employee_name"], state["sched_employee_email"] = emp.name, getattr(
            emp, "email", None
        )

    if not state["sched_date"]:
        return await _llm_reply(
            "Ask for date.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id,
        )
    if not state["sched_time"]:
        return await _llm_reply(
            "Ask for time.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id,
        )
    if not state["sched_purpose"]:
        return await _llm_reply(
            "Ask for purpose.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id,
        )

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
            user_query,
            client_id,
        )

    return await _llm_reply(
        "Ask clearly if you should log the meeting.",
        {"visitor_name": state.get("visitor_name")},
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

    # 1. THE "HEY JARVIS" HARD RESET
    # Matches "Jarvis", "Hey Jarvis", "Hi Jarvis", etc., at the start of the string
    wake_word_match = re.match(r"^(hey jarvis|hi jarvis|hello jarvis|jarvis)\b\s*,? ?(.*)", query_clean)
    
    if wake_word_match:
        # Perform a Hard Reset (Wipe session state AND LLM history)
        clear_session_state(client_id, retain_name=False) 
        # Note: clear_session_state inside your file calls GroqProcessor.reset_history
        state = get_session_state(client_id) 
        
        # Capture text after the wake word (e.g., "I'm Sudha")
        remaining_text = wake_word_match.group(2).strip()
        
        # If the user ONLY said the wake word, give a fresh greeting
        if not remaining_text:
            return await _llm_reply(
                "The user just said your wake word. Greet them warmly and naturally as if they just walked up. "
                "Do NOT use robotic greetings like 'Welcome to Sharp Software' and NEVER call them 'Visitor'.",
                {}, None, client_id
            )
        # Otherwise, replace the query with the remaining text and continue
        user_query = remaining_text
        query_clean = user_query.lower().strip()

    # 2. SMART TIMEOUT LOGIC
    time_since_last = (datetime.utcnow() - state["last_active"]).total_seconds()
    timeout_limit = 300 if state["conv_state"] == State.COMPLETED else 60
    if time_since_last > timeout_limit:
        clear_session_state(client_id, retain_name=False)
        state = get_session_state(client_id)
    state["last_active"] = datetime.utcnow()

    # 3. INTENT & ENTITY EXTRACTION
    extracted = await llm.extract_intent_and_entities(user_query)
    entities, intent = extracted.get("entities", {}), extracted.get("intent", "general_conversation")

    # 4. 30-MINUTE ARRIVAL PRIORITY
    # If they mention a time (e.g., "I'm here for my 3pm"), check if it's close to now.
    time_val = entities.get("time")
    if time_val:
        if _is_time_close_to_now(str(time_val)):
            intent = "check_in"  # They are likely arriving for a scheduled meeting
        else:
            intent = "schedule_meeting"  # They are likely booking for the future
            state["scheduling_active"] = True

    # 5. THE GOODBYE SOFT RESET
    # We catch goodbyes here. We don't wipe the memory yet so that if they 
    # follow up immediately (like Sudha did), Jarvis still knows who they are.
    is_goodbye = re.search(
        r"\b(bye|goodbye|thank you|thanks|no thanks|see you|see ya|take care|that'?s all|i'?m done|have a good day)\b",
        query_clean
    )
    if is_goodbye:
        response = await _llm_reply(
            "The visitor is leaving or finished. Say a warm, natural goodbye. "
            "Do not ask any follow-up questions.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id
        )
        state["conv_state"] = State.COMPLETED
        return response

    # 6. RE-START LOGIC
    # If they are starting a brand new check-in/schedule but we already have a name,
    # we treat it as a continuation or a fresh task for the same person.
    if state["conv_state"] == State.COMPLETED and intent in ["check_in", "schedule_meeting"]:
        clear_session_state(client_id, retain_name=True)
        state = get_session_state(client_id)

    # 7. MERGE DATA & ROUTE
    _merge_checkin_entities(state, entities, client_id, user_query)

    if intent == "facility_request":
        return await _llm_reply("Assure them you are pinging administration.", {"visitor_name": state.get("visitor_name")}, user_query, client_id)

    if intent == "schedule_meeting":
        state["scheduling_active"] = True

    if state["scheduling_active"]:
        return await _handle_scheduling(client_id, user_query, state, entities, intent)

    if intent == "employee_lookup":
        target = _clean_entity(entities.get("employee_name")) or _clean_entity(entities.get("role"))
        emp = _lookup_employee(target) if target else None
        if emp:
            from client_context import set_last_employee
            set_last_employee(client_id, name=emp.name, role=emp.role, cabin=getattr(emp, "location", None))
            return await llm.generate_grounded_response({"employee": {"name": emp.name, "role": emp.role, "cabin_number": getattr(emp, "location", "")}, "visitor_name": state.get("visitor_name")}, user_query)
        return await _llm_reply(f"Apologize that you couldn't find '{target}'.", {"visitor_name": state.get("visitor_name")}, user_query, client_id)

    if intent == "check_in" and state["conv_state"] != State.COMPLETED:
        return await _advance_checkin(state, user_query, client_id)

    # 8. GENERAL CONVERSATION FALLBACK
    # If state is COMPLETED, we still answer using the company info but don't re-trigger check-in
    info = _get_full_company_info(state)
    if state["conv_state"] == State.COMPLETED:
        info["status"] = "Interaction finished, visitor is likely waiting or leaving."
        
    return await llm.get_response(client_id, user_query, company_info=info)

async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    # Step 1: Attempt to resolve the host immediately if we have a name
    if state["meeting_with_raw"] and not state["meeting_with_resolved"]:
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name

    # Step 2: Specialized Delivery Flow
    if state.get("is_delivery"):
        # If we don't know who it's for, that's the only mandatory question
        if not state.get("meeting_with_raw"):
            state["conv_state"] = State.COLLECTING_HOST
            situation = "Ask who the delivery is for."
            return await _llm_reply(situation, {}, user_query, client_id=client_id)

        # If we know it's for Priya, don't nag for a driver name.
        # Just auto-fill if missing and complete.
        if not state.get("visitor_name"):
            state["visitor_name"] = f"{state['visitor_type']} Personnel"

        return await _complete_checkin(state, user_query, client_id)

    # Step 3: Standard Visitor Flow
    if not state.get("visitor_name"):
        state["conv_state"] = State.COLLECTING_NAME
        return await _llm_reply(
            "Ask for the visitor's name.", {}, user_query, client_id=client_id
        )

    if not state.get("meeting_with_raw"):
        state["conv_state"] = State.COLLECTING_HOST
        return await _llm_reply(
            "Ask who they are here to see.",
            {"visitor_name": state["visitor_name"]},
            user_query,
            client_id=client_id,
        )

    if not state.get("purpose"):
        state["conv_state"] = State.COLLECTING_PURPOSE
        return await _llm_reply(
            "Ask the purpose of their visit.",
            {
                "visitor_name": state["visitor_name"],
                "employee_name": state["meeting_with_raw"],
            },
            user_query,
            client_id=client_id,
        )

    return await _complete_checkin(state, user_query, client_id)


async def _complete_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    emp = _lookup_employee(state["meeting_with_resolved"] or state["meeting_with_raw"])
    if emp:
        state["meeting_with_resolved"] = emp.name

    success = _commit_checkin(state)
    state["conv_state"] = State.COMPLETED

    if state.get("is_delivery"):
        target = emp.name if emp else state["meeting_with_raw"]
        return await _llm_reply(
            f"Thank the delivery person. Tell them to leave the package at the front desk, you will ping {target}.",
            {"visitor_name": state["visitor_name"], "employee_name": target},
            user_query,
            client_id=client_id,
        )

    if emp:
        return await _llm_reply(
            f"Tell the visitor they are checked in. You are notifying {emp.name}. Direct them to {getattr(emp, 'location', 'their workspace')}.",
            {"visitor_name": state["visitor_name"], "employee_name": emp.name},
            user_query,
            client_id=client_id,
        )

    return await _llm_reply(
        f"Tell the visitor they are checked in. You are notifying the {state['meeting_with_raw']} team.",
        {
            "visitor_name": state["visitor_name"],
            "employee_name": state["meeting_with_raw"],
        },
        user_query,
        client_id=client_id,
    )
