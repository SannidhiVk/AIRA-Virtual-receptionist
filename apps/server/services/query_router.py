"""
query_router.py
───────────────
Conversation state machine for Sannika, the virtual receptionist.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from sqlalchemy import or_

from receptionist.database import SessionLocal
from receptionist.models import Employee, Visitor, Meeting, ReceptionLog
from models.groq_processor import GroqProcessor

logger = logging.getLogger(__name__)

COMPANY_NAME = "Sharp Software"
AI_NAME = "Sannika"

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


def clear_session_state(client_id: str) -> None:
    _client_sessions[client_id] = _fresh_state()
    try:
        GroqProcessor.get_instance().reset_history(client_id)
    except Exception:
        pass


def _fresh_state() -> Dict[str, Any]:
    return {
        "conv_state": State.INIT,
        "visitor_name": None,
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
    }


# ─────────────────────────────────────────────
# DATE/TIME FORMATTING & DB
# ─────────────────────────────────────────────


def _normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    today = datetime.now().date()
    if s in ("today", "now"):
        return today.strftime("%Y-%m-%d")
    if s == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    m = re.match(r"^in\s+(\d+)\s+days?$", s)
    if m:
        return (today + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
    day_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    m2 = re.match(
        r"^(?:next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s
    )
    if m2:
        target = day_map[m2.group(1)]
        days_ahead = (target - today.weekday() + 7) % 7 or 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except ValueError:
        pass
    return None


def _normalize_time(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower().replace("p.m.", "pm").replace("a.m.", "am").replace(" ", "")
    if re.match(r"^\d{2}:\d{2}$", s):
        return s
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if mer == "pm" and hour != 12:
            hour += 12
        if mer == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    m2 = re.match(r"^(\d{1,2})(?::(\d{2}))?$", s)
    if m2:
        hour, minute = int(m2.group(1)), int(m2.group(2) or 0)
        if 1 <= hour <= 7:
            hour += 12
        return f"{hour:02d}:{minute:02d}"
    return None


def _fmt_date_str(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%A, %B ") + str(dt.day)
    except Exception:
        return date_str


def _fmt_time_str(time_str: str) -> str:
    try:
        h, m = map(int, time_str.split(":"))
        return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"
    except Exception:
        return time_str


def _lookup_employee(search_term: str) -> Optional[Any]:
    if not search_term or not search_term.strip():
        return None
    clean = search_term.strip()
    db = SessionLocal()
    try:
        return (
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
    finally:
        db.close()


def _commit_checkin(state: Dict[str, Any]) -> bool:
    db = SessionLocal()
    try:
        v = Visitor(
            name=state["visitor_name"],
            meeting_with=state["meeting_with_resolved"] or state["meeting_with_raw"],
            purpose=state["purpose"],
            checkin_time=datetime.utcnow(),
        )
        db.add(v)
        db.flush()
        log = ReceptionLog(
            person_name=state["visitor_name"],
            person_type="DELIVERY" if state.get("is_delivery") else "VISITOR",
            linked_visitor_id=v.id,
            check_in_time=datetime.utcnow().isoformat(),
            notes=(
                f"Meeting with: {v.meeting_with}" if v.meeting_with else "General visit"
            ),
        )
        db.add(log)
        db.commit()
        return True
    except Exception:
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
    organizer: str,
) -> bool:
    db = SessionLocal()
    try:
        meeting = Meeting(
            organizer_name=organizer,
            organizer_type="visitor",
            visitor_name=organizer,
            employee_name=emp_name,
            employee_email=emp_email,
            meeting_date=date_str,
            meeting_time=time_str,
            purpose=purpose,
            status="scheduled",
            created_at=datetime.utcnow().isoformat(),
        )
        db.add(meeting)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def _clean_entity(val: Any) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip()
    if s.lower() in ("null", "none", "") or s.lower() in PRONOUNS:
        return None
    return s


def _merge_checkin_entities(state: Dict[str, Any], entities: Dict[str, Any]) -> None:
    name = _clean_entity(entities.get("visitor_name"))
    if name and not state["visitor_name"]:
        state["visitor_name"] = name.capitalize()

    target = _clean_entity(entities.get("employee_name")) or _clean_entity(
        entities.get("role")
    )
    if target and not state["meeting_with_raw"]:
        state["meeting_with_raw"] = target

    purpose = _clean_entity(entities.get("purpose"))
    if purpose and not state["purpose"] and len(purpose) > 3:
        state["purpose"] = purpose.capitalize()


# ─────────────────────────────────────────────
# LLM REPLY GENERATOR
# ─────────────────────────────────────────────


async def _llm_reply(situation: str, context: dict, user_query: str = None) -> str:
    llm = GroqProcessor.get_instance()
    lines = [f"Your name: {AI_NAME}"]
    for k, v in context.items():
        if v:
            lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")

    prompt = f"""You are {AI_NAME}, a highly intelligent, proactive, and conversational corporate receptionist.
VERIFIED FACTS:
{chr(10).join(lines)}
"""
    if user_query:
        prompt += f'\nTHE VISITOR JUST SAID:\n"{user_query}"\n'

    prompt += f"""
YOUR GOAL FOR THIS RESPONSE:
{situation}

RULES:
1. If the visitor asked a question, answer it intelligently FIRST.
2. Respond naturally in 1 to 3 sentences. No robotic scripts. Use natural connectors ("Got it", "Sure", "Thanks").
3. Use ONLY facts listed above.
4. Never say you are an AI.
"""
    return await llm.get_raw_response(prompt)


# ─────────────────────────────────────────────
# SCHEDULING FLOW
# ─────────────────────────────────────────────


async def _handle_scheduling(
    client_id: str, user_query: str, state: Dict[str, Any], entities: Dict[str, Any]
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

    # Pull name from check-in state if available
    if not state.get("visitor_name") and entities.get("visitor_name"):
        state["visitor_name"] = entities.get("visitor_name").capitalize()

    if not state.get("visitor_name"):
        return await _llm_reply(
            "Ask the visitor for their name so you can schedule the meeting for them.",
            {},
            user_query,
        )

    if not state["sched_employee_raw"]:
        return await _llm_reply(
            "Ask the visitor who they would like to schedule the meeting with.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
        )

    if not state["sched_employee_name"]:
        emp = _lookup_employee(state["sched_employee_raw"])
        if not emp:
            bad = state["sched_employee_raw"]
            state["sched_employee_raw"] = None
            return await _llm_reply(
                f"Tell the visitor that '{bad}' was not found in the directory and ask to clarify who they want to meet.",
                {"visitor_name": state.get("visitor_name")},
                user_query,
            )
        state["sched_employee_name"] = emp.name
        state["sched_employee_email"] = getattr(emp, "email", None)

    emp_display = state["sched_employee_name"]

    if not state["sched_date"]:
        return await _llm_reply(
            f"Ask the visitor what date they want to schedule the meeting with {emp_display}.",
            {"visitor_name": state.get("visitor_name"), "employee_name": emp_display},
            user_query,
        )

    if not state["sched_time"]:
        return await _llm_reply(
            f"Ask the visitor what time they want to schedule the meeting.",
            {"visitor_name": state.get("visitor_name"), "employee_name": emp_display},
            user_query,
        )

    if not state["sched_purpose"]:
        return await _llm_reply(
            f"Ask the visitor what the purpose of the meeting is.",
            {"visitor_name": state.get("visitor_name"), "employee_name": emp_display},
            user_query,
        )

    date_str, time_str = _fmt_date_str(state["sched_date"]), _fmt_time_str(
        state["sched_time"]
    )

    if not state["sched_pending_confirm"]:
        state["sched_pending_confirm"] = True
        return await _llm_reply(
            "Read back the meeting details to the visitor and ask them if you should go ahead and confirm it.",
            {
                "visitor_name": state.get("visitor_name"),
                "employee_name": emp_display,
                "date_str": date_str,
                "time_str": time_str,
                "purpose": state["sched_purpose"],
            },
            user_query,
        )

    q = user_query.lower()
    is_yes = entities.get("intent") == "confirm" or any(
        w in q
        for w in [
            "yes",
            "yeah",
            "confirm",
            "proceed",
            "go ahead",
            "sure",
            "okay",
            "ok",
            "please do",
        ]
    )
    is_no = entities.get("intent") == "cancel" or any(
        w in q for w in ["no", "nope", "cancel", "stop", "never mind", "abort"]
    )

    if is_no:
        state["scheduling_active"] = False
        return await _llm_reply(
            "Tell the visitor you have cancelled the meeting request. Ask if they need help with anything else.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
        )

    if is_yes:
        success = _commit_meeting(
            state["sched_employee_name"],
            state["sched_employee_email"],
            state["sched_date"],
            state["sched_time"],
            state["sched_purpose"],
            state.get("visitor_name") or "Visitor",
        )
        state["scheduling_active"] = False
        if success:
            return await _llm_reply(
                f"Tell the visitor the meeting is successfully confirmed for {date_str} at {time_str}. Let them know they do not need to head to the cabin now, just wait until the scheduled time.",
                {
                    "visitor_name": state.get("visitor_name"),
                    "employee_name": emp_display,
                },
                user_query,
            )
        else:
            return await _llm_reply(
                "Apologize and say there was a technical issue saving the meeting to the system.",
                {"visitor_name": state.get("visitor_name")},
                user_query,
            )

    return await _llm_reply(
        "You didn't catch if they confirmed or not. Ask them clearly if you should log the meeting in the calendar.",
        {"visitor_name": state.get("visitor_name")},
        user_query,
    )


# ─────────────────────────────────────────────
# CORE ROUTER
# ─────────────────────────────────────────────


async def route_query(client_id: str, user_query: str) -> str:
    llm = GroqProcessor.get_instance()
    state = get_session_state(client_id)
    q_lower = user_query.strip().lower()

    logger.info(
        "[%s] State=%s | Input='%s'", client_id, state["conv_state"], user_query
    )

    extracted = await llm.extract_intent_and_entities(user_query)
    entities, intent = extracted.get("entities", {}), extracted.get(
        "intent", "general_conversation"
    )
    logger.info("[%s] Intent=%s | Entities=%s", client_id, intent, entities)

    _merge_checkin_entities(state, entities)

    # ── SCHEDULING DETECTION ─────────────────────────────────────────────
    # If the user explicitly asks to schedule/book an appointment, activate the scheduling flow.
    wants_scheduling = intent == "schedule_meeting" or (
        any(w in q_lower for w in ["schedule", "book", "appointment"])
        and not any(w in q_lower for w in ["now", "arrived", "here for", "where is"])
    )
    if wants_scheduling:
        state["scheduling_active"] = True

    if state["scheduling_active"]:
        return await _handle_scheduling(client_id, user_query, state, entities)

    # ── SMART HOST FALLBACK (If they agree to contact HR/Admin) ─────────
    if state["conv_state"] == State.COLLECTING_HOST:
        if intent == "confirm" or any(
            w in q_lower
            for w in [
                "yes",
                "sure",
                "hr",
                "admin",
                "go ahead",
                "please do",
                "okay",
                "yeah",
            ]
        ):
            if not state["meeting_with_raw"]:
                state["meeting_with_raw"] = "HR / Administration"

    # ── FACILITY & MAINTENANCE REQUESTS ──────────────────────────────────
    is_facility = intent == "facility_request" or any(
        w in q_lower
        for w in [
            "ac",
            "temperature",
            "aircon",
            "too cold",
            "too hot",
            "light",
            "clean",
            "spill",
            "maintenance",
            "wifi",
            "internet",
        ]
    )
    if is_facility:
        return await _llm_reply(
            "Empathize warmly with their request and assure them you are pinging the administration team to handle it right away.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
        )

    # ── SMART DELIVERY DETECTION ─────────────────────────────────────────
    purpose_str = (state.get("purpose") or "").lower()
    name_str = (state.get("visitor_name") or "").lower()
    if (
        any(w in purpose_str for w in ["deliver", "drop", "package", "courier"])
        or any(w in name_str for w in ["deliver", "courier"])
        or any(w in q_lower for w in ["delivery", "drop off", "package", "courier"])
    ):
        state["is_delivery"] = True

    if state.get("is_delivery"):
        if not state.get("visitor_name"):
            state["visitor_name"] = "Delivery"
        if not state.get("purpose"):
            state["purpose"] = "Drop off delivery"

    # ── EMPLOYEE / DEPARTMENT LOOKUP ──────────────────────────────────────
    is_lookup = intent == "employee_lookup" or any(
        k in q_lower
        for k in ["who is", "where is", "which cabin", "which floor", "what floor"]
    )
    if is_lookup:
        target = (
            _clean_entity(entities.get("employee_name"))
            or _clean_entity(entities.get("role"))
            or state.get("meeting_with_resolved")
            or state.get("meeting_with_raw")
        )
        emp = _lookup_employee(target) if target else None
        if emp:
            context = {
                "employee": {
                    "name": emp.name,
                    "role": emp.role,
                    "cabin_number": emp.cabin_number,
                    "department": emp.department,
                    "floor": emp.floor,
                },
                "visitor_name": state.get("visitor_name"),
            }
            return await llm.generate_grounded_response(
                context=context, question=user_query
            )

        return await _llm_reply(
            f"Apologize warmly that you couldn't find '{target}' in the directory. Suggest they grab a seat in the lobby while you message administration to figure it out for them.",
            {"visitor_name": state.get("visitor_name")},
            user_query,
        )

    # ── GENERAL CONVERSATION (PRE-CHECK-IN) ──────────────────────────────
    if intent == "general_conversation" and state["conv_state"] != State.COMPLETED:
        has_new_info = any(
            _clean_entity(entities.get(k))
            for k in ["visitor_name", "employee_name", "role", "purpose"]
        )
        if not has_new_info and not state.get("is_delivery"):
            company_info = {
                "company_name": COMPANY_NAME,
                "visitor_name": state.get("visitor_name") or "Visitor",
            }
            return await llm.get_response(
                client_id, user_query, company_info=company_info
            )

    # ── EMPLOYEE ARRIVAL DETECTION ────────────────────────────────────────
    if (
        intent == "check_in"
        and state["conv_state"] in (State.INIT, State.COLLECTING_NAME)
        and not state.get("is_delivery")
    ):
        name_ent = _clean_entity(entities.get("visitor_name"))
        host_ent = _clean_entity(entities.get("employee_name")) or _clean_entity(
            entities.get("role")
        )
        if name_ent and not host_ent:
            emp_match = _lookup_employee(name_ent)
            if emp_match:
                db = SessionLocal()
                try:
                    log = ReceptionLog(
                        person_name=emp_match.name,
                        person_type="EMPLOYEE",
                        linked_employee_id=emp_match.id,
                        check_in_time=datetime.utcnow().isoformat(),
                        notes="Employee arrived",
                    )
                    db.add(log)
                    db.commit()
                except Exception as e:
                    logger.error("Employee check-in DB error: %s", e)
                finally:
                    db.close()
                clear_session_state(client_id)
                return await _llm_reply(
                    f"Warmly welcome {emp_match.name} back to the office. Let them know you've logged their arrival for the day.",
                    {"employee_name": emp_match.name},
                    user_query,
                )

    # ── CHECK-IN FLOW ─────────────────────────────────────────────────────
    if state["conv_state"] != State.COMPLETED:
        return await _advance_checkin(state, user_query)

    # ── POST CHECK-IN FALLBACK ────────────────────────────────────────────
    emp = _lookup_employee(
        state.get("meeting_with_resolved") or state.get("meeting_with_raw")
    )
    company_info = {
        "company_name": COMPANY_NAME,
        "visitor_name": state.get("visitor_name") or "Visitor",
    }
    if emp:
        company_info["dynamic_employee"] = (
            f"Name: {emp.name} | Role: {emp.role} | Floor: {emp.floor} | Cabin: {emp.cabin_number}"
        )
    return await llm.get_response(client_id, user_query, company_info=company_info)


# ─────────────────────────────────────────────
# CHECK-IN FLOW HELPERS
# ─────────────────────────────────────────────


async def _advance_checkin(state: Dict[str, Any], user_query: str) -> str:
    if state["meeting_with_raw"] and not state["meeting_with_resolved"]:
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name

    has_name = bool(state["visitor_name"])
    has_host = bool(state["meeting_with_raw"])
    has_purpose = bool(state["purpose"])

    if has_name and has_host and has_purpose:
        return await _complete_checkin(state, user_query)

    if state.get("is_delivery") and not has_host:
        state["conv_state"] = State.COLLECTING_HOST
        return await _llm_reply(
            "Warmly ask the delivery person who the package or delivery is for.",
            {},
            user_query,
        )

    if not has_name:
        state["conv_state"] = State.COLLECTING_NAME
        return await _llm_reply(
            f"Warmly welcome the visitor. Introduce yourself as {AI_NAME} and ask for their name.",
            {},
            user_query,
        )

    if not has_host:
        state["conv_state"] = State.COLLECTING_HOST
        return await _llm_reply(
            "Acknowledge the visitor naturally and ask who they are here to see today. If they don't know, suggest pinging HR or Administration.",
            {"visitor_name": state["visitor_name"]},
            user_query,
        )

    if not has_purpose:
        state["conv_state"] = State.COLLECTING_PURPOSE
        emp_display = state["meeting_with_resolved"] or state["meeting_with_raw"]
        return await _llm_reply(
            f"Politely ask the visitor the purpose of their visit with {emp_display}.",
            {"visitor_name": state["visitor_name"], "employee_name": emp_display},
            user_query,
        )

    return await _complete_checkin(state, user_query)


async def _complete_checkin(state: Dict[str, Any], user_query: str) -> str:
    emp = None
    if state["meeting_with_raw"] and state["meeting_with_raw"].lower() not in [
        "front desk",
        "reception",
    ]:
        emp = _lookup_employee(
            state["meeting_with_resolved"] or state["meeting_with_raw"]
        )
        if emp:
            state["meeting_with_resolved"] = emp.name

    success = _commit_checkin(state)
    state["conv_state"] = State.COMPLETED

    if not success:
        return (
            "Got it. Your visit has been noted. Please have a seat in the lobby area."
        )

    if state.get("is_delivery"):
        target = emp.name if emp else state["meeting_with_raw"]
        return await _llm_reply(
            f"Thank the delivery person. Tell them to leave the package at the front desk, and assure them you will ping {target} to pick it up.",
            {"visitor_name": state["visitor_name"], "employee_name": target},
            user_query,
        )

    if emp:
        return await _llm_reply(
            f"Tell the visitor they are fully checked in. Let them know you're sending a message to {emp.name}. Direct them to floor {emp.floor}, cabin {emp.cabin_number}.",
            {
                "visitor_name": state["visitor_name"],
                "employee_name": emp.name,
                "floor": emp.floor,
                "cabin": emp.cabin_number,
            },
            user_query,
        )

    target = state["meeting_with_raw"]
    return await _llm_reply(
        f"Tell the visitor they are checked in. Let them know you are notifying the {target} team about their arrival. Ask them to make themselves comfortable in the seating area.",
        {"visitor_name": state["visitor_name"], "employee_name": target},
        user_query,
    )
