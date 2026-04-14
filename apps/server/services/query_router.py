import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from sqlalchemy import or_, and_

from receptionist.database import SessionLocal, get_company_details
from receptionist.models import Employee, Visitor, Meeting, ReceptionLog
from models.groq_processor import GroqProcessor

logger = logging.getLogger(__name__)

AI_NAME = "Jarvis"  # Changed from Sannika to Jarvis to match your request
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
    _client_sessions[client_id] = _fresh_state()
    if retain_name and old_state.get("visitor_name"):
        _client_sessions[client_id]["visitor_name"] = old_state["visitor_name"]
        _client_sessions[client_id]["visitor_email"] = old_state["visitor_email"]
    else:
        try:
            GroqProcessor.get_instance().reset_history(client_id)
        except Exception:
            pass


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
    if re.search(r"\b(swiggy|zomato|food|lunch)\b", combined):
        return "Food Delivery"
    if re.search(r"\b(amazon|flipkart|delivery|courier|package)\b", combined):
        return "Delivery"
    if re.search(r"\b(vendor|electrician|plumber|maintenance)\b", combined):
        return "Contractor/Vendor"
    if re.search(r"\b(client|customer|demo)\b", combined):
        return "Client"
    return current_type or "Visitor/Guest"


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
        visitor = (
            db.query(Visitor).filter(Visitor.name.ilike(state["visitor_name"])).first()
        )
        if not visitor:
            visitor = Visitor(name=state["visitor_name"])
            db.add(visitor)
            db.flush()

        emp = (
            _lookup_employee(state["meeting_with_raw"])
            if state.get("meeting_with_raw")
            else None
        )

        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=emp.id if emp else None,
            person_type=state["visitor_type"],
            check_in_time=datetime.utcnow(),
            purpose=state["purpose"],
            notes=f"Meeting with: {state.get('meeting_with_raw')}",
        )
        db.add(log)
        db.commit()

        # FIRE TEAMS (Email can be added here if needed)
        if emp:
            from services.notify_slack import send_slack_arrival

            send_slack_arrival(
                emp.name,
                state.get("visitor_name", "Unknown"),
                state["visitor_type"],
                state["purpose"],
            )
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

    return res is not None


def _clean_entity(val: Any) -> Optional[str]:
    s = str(val).strip() if val else ""
    return (
        s
        if s and s.lower() not in ("null", "none", "") and s.lower() not in PRONOUNS
        else None
    )


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], client_id: str, user_query: str
) -> None:
    name, email = _clean_entity(entities.get("visitor_name")), _clean_entity(
        entities.get("email")
    )
    if name:
        state["visitor_name"] = name.capitalize()  # Overrides and trusts user
    if email:
        state["visitor_email"] = email

    target = _clean_entity(entities.get("employee_name")) or _clean_entity(
        entities.get("role")
    )
    if target and not state["meeting_with_raw"]:
        state["meeting_with_raw"] = target

    purpose = _clean_entity(entities.get("purpose"))
    if purpose and len(purpose) > 3:
        state["purpose"] = purpose.capitalize()

    state["visitor_type"] = _determine_visitor_type(
        user_query, str(purpose or ""), _clean_entity(entities.get("visitor_type"))
    )

    if (
        str(entities.get("employee_name") or "").strip().lower() in PRONOUNS
        and client_id
    ):
        from client_context import get_last_employee_name

        resolved = get_last_employee_name(client_id)
        if resolved and not state["meeting_with_raw"]:
            state["meeting_with_raw"] = resolved


async def _llm_reply(
    situation: str, context: dict, user_query: str = None, client_id: str = None
) -> str:
    llm = GroqProcessor.get_instance()
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    lines = [f"Your name: {AI_NAME}"]
    for k, v in context.items():
        if v:
            lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")

    prompt = f"""You are {AI_NAME}, an intelligent corporate receptionist.
CURRENT DATE & TIME: {current_time}
FACTS:
{chr(10).join(lines)}

GOAL:
{situation}

RULES:
1. Answer intelligently in 1-3 sentences.
2. If the visitor's name is a company/job (Electrician, Swiggy, Amazon), DO NOT address them by that name.
3. NEVER correct the visitor's greeting (if they say Good Morning in the evening, ignore it).
4. NEVER argue about their name. Trust what they say.
"""
    if user_query:
        prompt += f'\nTHE VISITOR SAID:\n"{user_query}"\n'
    return await llm.get_raw_response(prompt, client_id=client_id)


async def _handle_scheduling(
    client_id: str,
    user_query: str,
    state: Dict[str, Any],
    entities: Dict[str, Any],
    intent: str,
) -> str:
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

    if not state.get("visitor_name"):
        return await _llm_reply(
            "Ask the visitor for their name.", {}, user_query, client_id=client_id
        )
    if not state["sched_employee_raw"]:
        return await _llm_reply(
            "Ask who they would like to schedule the meeting with.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )

    if not state["sched_employee_name"]:
        emp = _lookup_employee(state["sched_employee_raw"])
        if not emp:
            bad = state["sched_employee_raw"]
            state["sched_employee_raw"] = None
            return await _llm_reply(
                f"Tell the visitor '{bad}' was not found. Clarify who they want.",
                {"visitor_name": state.get("visitor_name")},
                user_query,
                client_id=client_id,
            )
        state["sched_employee_name"], state["sched_employee_email"] = emp.name, getattr(
            emp, "email", None
        )

    emp_display = state["sched_employee_name"]
    if not state["sched_date"]:
        return await _llm_reply(
            f"Ask what date to schedule with {emp_display}.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )
    if not state["sched_time"]:
        return await _llm_reply(
            "Ask what time to schedule.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )
    if not state["sched_purpose"]:
        return await _llm_reply(
            "Ask the purpose of the meeting.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )

    if not state["sched_pending_confirm"]:
        state["sched_pending_confirm"] = True
        return await _llm_reply(
            "Read back the meeting details and ask if you should confirm.",
            {
                "visitor_name": state.get("visitor_name"),
                "employee_name": emp_display,
                "date": state["sched_date"],
                "time": state["sched_time"],
                "purpose": state["sched_purpose"],
            },
            user_query,
            client_id=client_id,
        )

    is_yes = intent == "confirm" or any(
        w in user_query.lower().split()
        for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "please"]
    )
    is_no = intent == "cancel" or any(
        w in user_query.lower().split() for w in ["no", "cancel", "nevermind"]
    )

    if is_no:
        state["scheduling_active"] = False
        return await _llm_reply(
            "Tell them you cancelled the request.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )

    if is_yes:
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
        if success:
            state["conv_state"] = State.COMPLETED
            return await _llm_reply(
                "Confirm the meeting is scheduled.",
                {
                    "visitor_name": state.get("visitor_name"),
                    "employee_name": emp_display,
                },
                user_query,
                client_id=client_id,
            )
        return await _llm_reply(
            "Apologize for a technical issue saving the meeting.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )

    return await _llm_reply(
        "Ask clearly if you should log the meeting.",
        {"visitor_name": state.get("visitor_name")},
        user_query,
        client_id=client_id,
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

    # --- SMART TIMEOUT LOGIC ---
    time_since_last = (datetime.utcnow() - state["last_active"]).total_seconds()
    timeout_limit = 300 if state["conv_state"] == State.COMPLETED else 60
    if time_since_last > timeout_limit:
        clear_session_state(client_id, retain_name=False)
        state = get_session_state(client_id)
    state["last_active"] = datetime.utcnow()
    # ---------------------------

    is_greeting = re.search(
        r"\b(hello|hi|hey|good morning|good afternoon)\b", user_query.lower()
    )
    is_goodbye = re.search(
        r"\b(bye|goodbye|thank you|thanks|no thanks)\b", user_query.lower()
    )

    active_states = [
        State.COLLECTING_NAME,
        State.COLLECTING_HOST,
        State.COLLECTING_PURPOSE,
    ]
    if (
        is_greeting
        and state["conv_state"] not in active_states
        and not state.get("scheduling_active")
    ):
        clear_session_state(client_id, retain_name=False)
        state = get_session_state(client_id)

    extracted = await llm.extract_intent_and_entities(user_query)
    entities, intent = extracted.get("entities", {}), extracted.get(
        "intent", "general_conversation"
    )

    if is_goodbye and state["conv_state"] in [State.COMPLETED, State.INIT]:
        clear_session_state(client_id, retain_name=False)
        return await _llm_reply(
            "Warmly say goodbye and wish them a great day. DO NOT ask if they need help.",
            {},
            user_query,
            client_id=client_id,
        )

    if state["conv_state"] == State.COMPLETED and intent in [
        "check_in",
        "schedule_meeting",
    ]:
        clear_session_state(client_id, retain_name=True)
        state = get_session_state(client_id)

    _merge_checkin_entities(state, entities, client_id, user_query)

    if intent == "facility_request":
        return await _llm_reply(
            "Assure them you are pinging administration.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )
    if intent == "schedule_meeting":
        state["scheduling_active"] = True
    if state["scheduling_active"]:
        return await _handle_scheduling(client_id, user_query, state, entities, intent)
    if state["visitor_type"] in ["Delivery", "Food Delivery"]:
        state["is_delivery"] = True

    if intent == "employee_lookup":
        target = (
            _clean_entity(entities.get("employee_name"))
            or _clean_entity(entities.get("role"))
            or state.get("meeting_with_resolved")
        )
        emp = _lookup_employee(target) if target else None
        if emp:
            from client_context import set_last_employee

            set_last_employee(
                client_id,
                name=emp.name,
                email=getattr(emp, "email", None),
                cabin=getattr(emp, "location", None),
                role=emp.role,
                department=getattr(emp, "department", None),
            )
            return await llm.generate_grounded_response(
                {
                    "employee": {
                        "name": emp.name,
                        "role": emp.role,
                        "cabin_number": getattr(emp, "location", ""),
                    },
                    "visitor_name": state.get("visitor_name"),
                },
                user_query,
            )
        return await _llm_reply(
            f"Apologize warmly that you couldn't find '{target}'.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
            client_id=client_id,
        )

    if intent == "check_in" and state["conv_state"] != State.COMPLETED:
        return await _advance_checkin(state, user_query, client_id)

    if intent == "general_conversation" and state["conv_state"] != State.COMPLETED:
        info = _get_full_company_info(state)
        return await llm.get_response(client_id, user_query, company_info=info)

    emp = _lookup_employee(
        state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    )
    info = _get_full_company_info(state)
    info["status"] = "Visitor is checked in and waiting."
    if emp:
        info["dynamic_employee"] = (
            f"Name: {emp.name} | Role: {emp.role} | Location: {getattr(emp, 'location', 'N/A')}"
        )
    return await llm.get_response(client_id, user_query, company_info=info)


async def _advance_checkin(
    state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    if state["meeting_with_raw"] and not state["meeting_with_resolved"]:
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name
            from client_context import set_last_employee

            set_last_employee(
                client_id,
                name=emp.name,
                role=emp.role,
                department=getattr(emp, "department", None),
            )

    if state.get("is_delivery") and not state.get("meeting_with_raw"):
        state["conv_state"] = State.COLLECTING_HOST
        return await _llm_reply(
            "Ask the delivery person who the package is for.",
            {},
            user_query,
            client_id=client_id,
        )

    if not state.get("visitor_name"):
        state["conv_state"] = State.COLLECTING_NAME
        return await _llm_reply(
            f"Welcome the visitor. Ask for their name.",
            {},
            user_query,
            client_id=client_id,
        )

    if not state.get("meeting_with_raw"):
        state["host_ask_count"] = state.get("host_ask_count", 0) + 1
        state["conv_state"] = State.COLLECTING_HOST
        if state["host_ask_count"] >= 2:
            state["meeting_with_raw"] = "HR / Administration"
            state["host_ask_count"] = 0
            return await _llm_reply(
                "Tell them you've notified HR to assist them. Ask them to take a seat.",
                {"visitor_name": state["visitor_name"]},
                user_query,
                client_id=client_id,
            )
        return await _llm_reply(
            "Ask who they are here to see.",
            {"visitor_name": state["visitor_name"]},
            user_query,
            client_id=client_id,
        )

    if not state.get("purpose"):
        state["conv_state"] = State.COLLECTING_PURPOSE
        emp_display = state["meeting_with_resolved"] or state["meeting_with_raw"]
        return await _llm_reply(
            f"Ask the purpose of their visit with {emp_display}.",
            {"visitor_name": state["visitor_name"], "employee_name": emp_display},
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
