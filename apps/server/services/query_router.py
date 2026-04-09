# --- REPLACE ENTIRE query_router.py WITH THIS ---

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from sqlalchemy import or_, and_

from receptionist.database import SessionLocal
from receptionist.models import Employee, Visitor, Meeting, ReceptionLog
from models.groq_processor import GroqProcessor

logger = logging.getLogger(__name__)

COMPANY_NAME = "Sharp Software"
AI_NAME = "Sannika"

PRONOUNS = {"him", "her", "them", "he", "she", "they", "it", "that person", "someone", "this guy", "this person"}

class State:
    INIT               = "INIT"
    COLLECTING_NAME    = "COLLECTING_NAME"
    COLLECTING_HOST    = "COLLECTING_HOST"
    COLLECTING_PURPOSE = "COLLECTING_PURPOSE"
    COMPLETED          = "COMPLETED"

_client_sessions: Dict[str, Dict[str, Any]] = {}

def get_session_state(client_id: str) -> Dict[str, Any]:
    if client_id not in _client_sessions:
        _client_sessions[client_id] = _fresh_state()
    return _client_sessions[client_id]

def clear_session_state(client_id: str, retain_name=False) -> None:
    old_state = _client_sessions.get(client_id, {})
    _client_sessions[client_id] = _fresh_state()
    
    # FIX: Retain name if moving from Check-in to Scheduling
    if retain_name and old_state.get("visitor_name"):
        _client_sessions[client_id]["visitor_name"] = old_state["visitor_name"]
        _client_sessions[client_id]["visitor_email"] = old_state["visitor_email"]
    else:
        try:
            GroqProcessor.get_instance().reset_history(client_id)
        except Exception: pass

def _fresh_state() -> Dict[str, Any]:
    return {
        "conv_state": State.INIT,
        "visitor_name": None,
        "visitor_type": "Visitor/Guest", # Default category
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
        "visitor_email": None,
        "host_ask_count": 0,
    }

# ─────────────────────────────────────────────
# REGEX FALLBACK FOR VISITOR CATEGORIES
# ─────────────────────────────────────────────
def _determine_visitor_type(text: str, purpose: str, current_type: str) -> str:
    combined = f"{text} {purpose}".lower()
    if re.search(r'\b(interview|candidate)\b', combined): return "Interviewee"
    if re.search(r'\b(swiggy|zomato|food|lunch)\b', combined): return "Food Delivery"
    if re.search(r'\b(amazon|flipkart|delivery|courier|package|parcel)\b', combined): return "Delivery"
    if re.search(r'\b(vendor|electrician|plumber|maintenance|repair|service|contractor)\b', combined): return "Contractor/Vendor"
    if re.search(r'\b(client|customer|demo)\b', combined): return "Client"
    return current_type or "Visitor/Guest"

# ─────────────────────────────────────────────
# DATE/TIME FORMATTING & DB
# ─────────────────────────────────────────────

def _normalize_date(raw: str) -> Optional[str]:
    if not raw: return None
    s = raw.strip().lower()
    today = datetime.now().date()
    if s in ("today", "now"): return today.strftime("%Y-%m-%d")
    if s == "tomorrow": return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    m = re.match(r"^in\s+(\d+)\s+days?$", s)
    if m: return (today + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
    day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    m2 = re.match(r"^(?:next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s)
    if m2:
        target = day_map[m2.group(1)]
        days_ahead = (target - today.weekday() + 7) % 7 or 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    try: return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except ValueError: pass
    return None

def _normalize_time(raw: str) -> Optional[str]:
    if not raw: return None
    s = raw.strip().lower().replace("p.m.", "pm").replace("a.m.", "am").replace(" ", "")
    if re.match(r"^\d{2}:\d{2}$", s): return s
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if mer == "pm" and hour != 12: hour += 12
        if mer == "am" and hour == 12: hour = 0
        return f"{hour:02d}:{minute:02d}"
    m2 = re.match(r"^(\d{1,2})(?::(\d{2}))?$", s)
    if m2:
        hour, minute = int(m2.group(1)), int(m2.group(2) or 0)
        if 1 <= hour <= 7: hour += 12
        return f"{hour:02d}:{minute:02d}"
    return None

def _fmt_date_str(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%A, %B ") + str(dt.day)
    except Exception: return date_str

def _fmt_time_str(time_str: str) -> str:
    try:
        h, m = map(int, time_str.split(":"))
        return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"
    except Exception: return time_str

# FIX: Smart lookup for "Sales Manager", "HR Manager", etc.
def _lookup_employee(search_term: str) -> Optional[Any]:
    if not search_term or not search_term.strip(): return None
    clean = search_term.strip().lower()
    db = SessionLocal()
    try:
        # 1. Exact Name/Role/Dept match
        emp = db.query(Employee).filter(
            or_(Employee.name.ilike(f"%{clean}%"), Employee.role.ilike(f"%{clean}%"), Employee.department.ilike(f"%{clean}%"))
        ).first()
        if emp: return emp

        # 2. Smart Manager Lookup (e.g. "Sales manager")
        if "manager" in clean:
            dept = clean.replace("manager", "").strip()
            if dept:
                emp = db.query(Employee).filter(
                    and_(Employee.department.ilike(f"%{dept}%"), Employee.role.ilike(f"%manager%"))
                ).first()
                if emp: return emp

        return None
    finally:
        db.close()

def _commit_checkin(state: Dict[str, Any]) -> bool:
    db = SessionLocal()
    try:
        visitor = db.query(Visitor).filter(Visitor.name.ilike(state["visitor_name"])).first()
        if not visitor:
            visitor = Visitor(name=state["visitor_name"])
            db.add(visitor)
            db.flush()

        emp = None
        if state.get("meeting_with_raw"):
            emp = _lookup_employee(state["meeting_with_raw"])

        # Create log using the newly tracked visitor_type
        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=emp.id if emp else None,
            person_type=state["visitor_type"], 
            check_in_time=datetime.utcnow(),
            purpose=state["purpose"],
            notes=f"Meeting with: {state.get('meeting_with_raw')}"
        )

        db.add(log)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Check-in failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def _commit_meeting(emp_name: str, emp_email: Optional[str], date_str: str, time_str: str, purpose: str, organizer: str) -> bool:
    from receptionist.database import schedule_meeting
    res = schedule_meeting(
        organizer_name=organizer, organizer_type="visitor", employee_name=emp_name,
        meeting_date=date_str, meeting_time=time_str, purpose=purpose
    )
    return res is not None

def _clean_entity(val: Any) -> Optional[str]:
    if not val: return None
    s = str(val).strip()
    if s.lower() in ("null", "none", "") or s.lower() in PRONOUNS: return None
    return s

def _merge_checkin_entities(state: Dict[str, Any], entities: Dict[str, Any], client_id: str, user_query: str) -> None:
    name = _clean_entity(entities.get("visitor_name"))
    # FIX: If a new name is spoken, ALWAYS override the old one. Don't trap them with the wrong name.
    if name: 
        state["visitor_name"] = name.capitalize()

    target = _clean_entity(entities.get("employee_name")) or _clean_entity(entities.get("role"))
    if target and not state["meeting_with_raw"]: state["meeting_with_raw"] = target

    purpose = _clean_entity(entities.get("purpose"))
    if purpose and not state["purpose"] and len(purpose) > 3: state["purpose"] = purpose.capitalize()

    llm_type = _clean_entity(entities.get("visitor_type"))
    state["visitor_type"] = _determine_visitor_type(user_query, str(purpose or ""), llm_type)

    raw_emp = str(entities.get("employee_name") or "").strip().lower()
    if raw_emp in PRONOUNS and client_id:
        from client_context import get_last_employee_name
        resolved = get_last_employee_name(client_id)
        if resolved and not state["meeting_with_raw"]:
            state["meeting_with_raw"] = resolved

async def _llm_reply(situation: str, context: dict, user_query: str = None, client_id: str = None) -> str:
    llm = GroqProcessor.get_instance()
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p") 
    
    lines = [f"Your name: {AI_NAME}"]
    for k, v in context.items():
        if v: lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")

    prompt = f"""You are {AI_NAME}, a highly intelligent, proactive corporate receptionist.
CURRENT DATE & TIME: {current_time}

VERIFIED FACTS:
{chr(10).join(lines)}

YOUR GOAL FOR THIS RESPONSE:
{situation}

RULES:
1. Answer the visitor's question intelligently FIRST.
2. Respond naturally in 1 to 3 sentences. No robotic scripts.
3. Use ONLY facts listed above. Do NOT make up dates or names.
4. If the visitor's name is a company or job title (e.g., Electrician, Swiggy, Delivery), DO NOT address them by that name.
5. Never correct the visitor's greeting (e.g., if they say "Good morning" in the afternoon, just play along and ignore it).
6. Never argue with the visitor about their own name. Trust whatever name they give you.
"""
    if user_query: prompt += f"\nTHE VISITOR JUST SAID:\n\"{user_query}\"\n"
    return await llm.get_raw_response(prompt, client_id=client_id)


async def _handle_scheduling(client_id: str, user_query: str, state: Dict[str, Any], entities: Dict[str, Any], intent: str) -> str:
    target = _clean_entity(entities.get("employee_name")) or _clean_entity(entities.get("role"))
    if target and not state["sched_employee_raw"]: state["sched_employee_raw"] = target
    
    if entities.get("date") and not state["sched_date"]: state["sched_date"] = _normalize_date(str(entities["date"]))
    if entities.get("time") and not state["sched_time"]: state["sched_time"] = _normalize_time(str(entities["time"]))
    if _clean_entity(entities.get("purpose")) and not state["sched_purpose"]: state["sched_purpose"] = _clean_entity(entities["purpose"])

    if not state.get("visitor_name") and entities.get("visitor_name"):
        state["visitor_name"] = entities.get("visitor_name").capitalize()

    if not state.get("visitor_name"):
        return await _llm_reply("Ask the visitor for their name so you can schedule the meeting for them.", {}, user_query, client_id=client_id)

    if not state["sched_employee_raw"]:
        return await _llm_reply("Ask the visitor who they would like to schedule the meeting with.", {"visitor_name": state.get("visitor_name")}, user_query, client_id=client_id)

    if not state["sched_employee_name"]:
        emp = _lookup_employee(state["sched_employee_raw"])
        if not emp:
            bad = state["sched_employee_raw"]
            state["sched_employee_raw"] = None
            return await _llm_reply(f"Tell the visitor that '{bad}' was not found in the directory and ask to clarify who they want to meet.", {"visitor_name": state.get("visitor_name")}, user_query, client_id=client_id)
        state["sched_employee_name"] = emp.name
        state["sched_employee_email"] = getattr(emp, "email", None)
    
    emp_display = state["sched_employee_name"]

    if not state["sched_date"]: return await _llm_reply(f"Ask the visitor what date they want to schedule the meeting with {emp_display}.", {"visitor_name": state.get("visitor_name"), "employee_name": emp_display}, user_query, client_id=client_id)
    if not state["sched_time"]: return await _llm_reply(f"Ask the visitor what time they want to schedule the meeting.", {"visitor_name": state.get("visitor_name"), "employee_name": emp_display}, user_query, client_id=client_id)
    if not state["sched_purpose"]: return await _llm_reply(f"Ask the visitor what the purpose of the meeting is.", {"visitor_name": state.get("visitor_name"), "employee_name": emp_display}, user_query)

    date_str, time_str = _fmt_date_str(state["sched_date"]), _fmt_time_str(state["sched_time"])
    
    if not state["sched_pending_confirm"]:
        state["sched_pending_confirm"] = True
        return await _llm_reply("Read back the meeting details and ask them if you should go ahead and confirm it.", {
            "visitor_name": state.get("visitor_name"), "employee_name": emp_display, "date_str": date_str, "time_str": time_str, "purpose": state["sched_purpose"]
        }, user_query, client_id=client_id)

    text = user_query.lower()
    is_yes = intent == "confirm" or any(w in text.split() for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "do"])
    is_no = intent == "cancel" or any(w in text.split() for w in ["no", "cancel", "nevermind", "stop"])

    if is_no:
        state["scheduling_active"] = False
        return await _llm_reply("Tell the visitor you have cancelled the meeting request. Ask if they need help with anything else.", {"visitor_name": state.get("visitor_name")}, user_query, client_id=client_id)

    if is_yes:
        success = _commit_meeting(state["sched_employee_name"], state["sched_employee_email"], state["sched_date"], state["sched_time"], state["sched_purpose"], state.get("visitor_name") or "Visitor")
        state["scheduling_active"] = False 
        if success:
            state["conv_state"] = State.COMPLETED
            return await _llm_reply(f"Confirm the meeting is scheduled for {date_str} at {time_str}. They don't need to do anything else.", {"visitor_name": state.get("visitor_name"), "employee_name": emp_display}, user_query, client_id=client_id)
        else:
            return await _llm_reply("Apologize and say there was a technical issue saving the meeting to the system.", {"visitor_name": state.get("visitor_name")}, user_query, client_id=client_id)

    return await _llm_reply("You didn't catch if they confirmed or not. Ask them clearly if you should log the meeting.", {"visitor_name": state.get("visitor_name")}, user_query, client_id=client_id)


async def route_query(client_id: str, user_query: str) -> str:
    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()

    is_greeting = re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", user_query.lower())
    is_goodbye = re.search(r"\b(bye|goodbye|thank you|thanks|no thanks|no)\b", user_query.lower())

    active_states = [State.COLLECTING_NAME, State.COLLECTING_HOST, State.COLLECTING_PURPOSE]
    if is_greeting and state["conv_state"] not in (State.INIT, State.COMPLETED) and not state.get("scheduling_active"):
        clear_session_state(client_id, retain_name=False)
        state = get_session_state(client_id)

    extracted = await llm.extract_intent_and_entities(user_query)
    entities, intent = extracted.get("entities", {}), extracted.get("intent", "general_conversation")
    
    # --- FIX: Clean Goodbye Exits ---
    if is_goodbye:
        if state["conv_state"] == State.COMPLETED or state["conv_state"] == State.INIT:
            clear_session_state(client_id, retain_name=False)
            return await _llm_reply(
                "The visitor is saying thank you or goodbye to end the conversation. Warmly say goodbye and wish them a great day. DO NOT ask if they need help with anything else.", 
                {}, user_query, client_id
            )

    # --- FIX: Amnesia Bug (Retaining name if doing multiple tasks) ---
    if state["conv_state"] == State.COMPLETED:
        if intent in ["check_in", "employee_arrival", "schedule_meeting"]:
            clear_session_state(client_id, retain_name=True) 
            state = get_session_state(client_id)

    _merge_checkin_entities(state, entities, client_id, user_query)
    
    if intent == "facility_request":
        return await _llm_reply("Assure them you are pinging the administration team to handle it right away.", {"visitor_name": state.get("visitor_name")}, user_query, client_id=client_id)

    if intent == "schedule_meeting": state["scheduling_active"] = True
    if state["scheduling_active"]: return await _handle_scheduling(client_id, user_query, state, entities, intent)

    if state["visitor_type"] in ["Delivery", "Food Delivery"]: state["is_delivery"] = True

    if intent == "employee_lookup":
        target = _clean_entity(entities.get("employee_name")) or _clean_entity(entities.get("role")) or state.get("meeting_with_resolved") or state.get("meeting_with_raw")
        emp = _lookup_employee(target) if target else None
        if emp:
            from client_context import set_last_employee
            set_last_employee(client_id, name=emp.name, email=getattr(emp, 'email', None), cabin=getattr(emp, 'location', None), role=emp.role, department=getattr(emp, 'department', None))
            context = {"employee": {"name": emp.name, "role": emp.role, "cabin_number": getattr(emp, "location", ""), "department": emp.department}, "visitor_name": state.get("visitor_name")}
            return await llm.generate_grounded_response(context=context, question=user_query)
        return await _llm_reply(f"Apologize warmly that you couldn't find '{target}' in the directory.", {"visitor_name": state.get("visitor_name")}, user_query, client_id=client_id)

    if intent == "check_in" and state["conv_state"] != State.COMPLETED:
        return await _advance_checkin(state, user_query, client_id)

    if intent == "general_conversation" and state["conv_state"] != State.COMPLETED:
        company_info = {"company_name": COMPANY_NAME, "visitor_name": state.get("visitor_name") or "Visitor"}
        return await llm.get_response(client_id, user_query, company_info=company_info)

    emp = _lookup_employee(state.get("meeting_with_resolved") or state.get("meeting_with_raw"))
    company_info = {"company_name": COMPANY_NAME, "visitor_name": state.get("visitor_name") or "Visitor"}
    if emp: company_info["dynamic_employee"] = f"Name: {emp.name} | Role: {emp.role} | Location: {getattr(emp, 'location', 'N/A')}"
    return await llm.get_response(client_id, user_query, company_info=company_info)

async def _advance_checkin(state: Dict[str, Any], user_query: str, client_id: str) -> str:
    if state["meeting_with_raw"] and not state["meeting_with_resolved"]:
        emp = _lookup_employee(state["meeting_with_raw"])
        if emp:
            state["meeting_with_resolved"] = emp.name
            from client_context import set_last_employee
            set_last_employee(client_id, name=emp.name, role=emp.role, department=getattr(emp, 'department', None))

    if state.get("is_delivery") and not state.get("meeting_with_raw"):
        state["conv_state"] = State.COLLECTING_HOST
        return await _llm_reply("Warmly ask the delivery person who the package is for.", {}, user_query, client_id=client_id)

    if not state.get("visitor_name"):
        state["conv_state"] = State.COLLECTING_NAME
        return await _llm_reply(f"Warmly welcome the visitor. Ask for their name.", {}, user_query, client_id=client_id)

    if not state.get("meeting_with_raw"):
        state["host_ask_count"] = state.get("host_ask_count", 0) + 1
        state["conv_state"] = State.COLLECTING_HOST
        if state["host_ask_count"] >= 2:
            state["meeting_with_raw"] = "HR / Administration"
            state["host_ask_count"] = 0
            return await _llm_reply("Tell them you've notified HR to assist them. Ask them to take a seat.", {"visitor_name": state["visitor_name"]}, user_query)
        return await _llm_reply("Ask who they are here to see.", {"visitor_name": state["visitor_name"]}, user_query)

    if not state.get("purpose"):
        state["conv_state"] = State.COLLECTING_PURPOSE
        emp_display = state["meeting_with_resolved"] or state["meeting_with_raw"]
        return await _llm_reply(f"Politely ask the purpose of their visit with {emp_display}.", {"visitor_name": state["visitor_name"], "employee_name": emp_display}, user_query, client_id=client_id)

    return await _complete_checkin(state, user_query, client_id)

async def _complete_checkin(state: Dict[str, Any], user_query: str, client_id: str) -> str:
    emp = None
    if state["meeting_with_raw"] and state["meeting_with_raw"].lower() not in ["front desk", "reception"]:
        emp = _lookup_employee(state["meeting_with_resolved"] or state["meeting_with_raw"])
        if emp: state["meeting_with_resolved"] = emp.name

    success = _commit_checkin(state)
    state["conv_state"] = State.COMPLETED

    if state.get("is_delivery"):
        target = emp.name if emp else state["meeting_with_raw"]
        return await _llm_reply(f"Thank the delivery person. Tell them to leave the package at the front desk, and assure them you will ping {target}.", {"visitor_name": state["visitor_name"], "employee_name": target}, user_query, client_id=client_id)

    if emp:
        loc = getattr(emp, 'location', 'their workspace')
        return await _llm_reply(f"Tell the visitor they are fully checked in. You are notifying {emp.name}. Direct them to {loc}.", {"visitor_name": state["visitor_name"], "employee_name": emp.name, "location": loc, "visitor_type": state["visitor_type"]}, user_query, client_id=client_id)
    
    target = state["meeting_with_raw"]
    return await _llm_reply(f"Tell the visitor they are checked in. You are notifying the {target} team.", {"visitor_name": state["visitor_name"], "employee_name": target, "visitor_type": state["visitor_type"]}, user_query, client_id=client_id)