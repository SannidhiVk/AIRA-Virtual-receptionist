import logging
import re
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Set
from sqlalchemy import or_, and_

# Database and Model Imports
from receptionist.database import (
    SessionLocal,
    get_company_details,
    get_available_slots,
    schedule_meeting,
    get_employee_by_name,
    get_similar_employee,
    ALL_SLOTS_1H,
)
from receptionist.models import Employee, Visitor, Meeting, ReceptionLog
from models.groq_processor import BASE_SYSTEM_PROMPT, GroqProcessor
from services.notify_slack import (
    send_slack_arrival,
    send_slack_meeting_scheduled,
    get_slack_user_id_by_email,
    clear_session as clear_slack_cache,
)
from services.calendar_service import schedule_google_meeting_background

# Logger Configuration
logger = logging.getLogger(__name__)


def _fmt_display(hhmm: str) -> str:
    """Convert internal 24hr 'HH:MM' to display '12:30 PM' for the LLM and user."""
    if not hhmm:
        return hhmm
    try:
        dt = datetime.strptime(hhmm, "%H:%M")
        hour = dt.hour % 12 or 12
        return f"{hour}:{dt.strftime('%M %p')}"
    except ValueError:
        return hhmm


# Constants - STRICTLY PRESERVED
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

# Semantic Role-Group Mapping
SEMANTIC_MAP = {
    "fix": "Operations",
    "leak": "Operations",
    "broken": "Operations",
    "maintenance": "Operations",
    "ac": "Operations",
    "repair": "Operations",
    "hiring": "HR",
    "recruit": "HR",
    "interview": "HR",
    "money": "Finance",
    "account": "Finance",
    "invoice": "Finance",
    "bill": "Finance",
    "payment": "Finance",
    "tech": "IT Support",
    "computer": "IT Support",
    "system": "IT Support",
    "internet": "IT Support",
    "software": "IT Support",
}


class State:
    INIT = "INIT"
    COLLECTING = "COLLECTING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"


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
    try:
        GroqProcessor.get_instance().reset_history(client_id)
    except Exception as e:
        logger.error(f"Hardware reset failed for {client_id}: {e}")


def _fresh_state() -> Dict[str, Any]:
    return {
        "session_id": str(__import__("uuid").uuid4()),
        "conv_state": "INIT",
        "last_active": __import__("datetime").datetime.utcnow(),
        "visitor_name": None,
        "visitor_email": None,
        "visitor_type": "Visitor/Guest",
        "greeting_sent": False,
        "meeting_with_raw": None,
        "meeting_with_resolved": None,
        "host_details": None,
        "is_employee": False,
        "purpose": None,
        "scheduling_active": False,
        "sched_employee_name": None,
        "sched_employee_email": None,
        "sched_date": None,
        "sched_time": None,
        "sched_purpose": None,
        "identity_updated": False,
        "notified_hosts": set(),
        "declined_hosts": set(),
        "all_hosts": [],  # list of all resolved host names to notify
        "greeted": False,
        "force_admin": False,
        "awaiting_slack_reply": False,
        # Face verification for name-collision with employee
        "pending_employee_face_check": False,
        "pending_employee_name": None,
        "pending_employee_id": None,
        "face_verified_name": None,
        "verified_employee_id": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION & LOOKUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_time_greeting() -> str:
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
        return "Intern"
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


def _format_descriptive_purpose(
    state: Dict[str, Any], event_type: str = "VISIT"
) -> str:
    v_type = state.get("visitor_type", "Visitor")
    host = (
        state.get("meeting_with_resolved")
        or state.get("sched_employee_name")
        or "Administration Team"
    )
    base_purpose = (
        state.get("purpose") or state.get("sched_purpose") or "General Business"
    )
    if event_type == "SCHEDULED":
        return f"SCHEDULED: {base_purpose} with {host} (Set for {state.get('sched_date')} @ {state.get('sched_time')})"
    return f"{v_type.upper()}: {base_purpose} for {host}"


def _normalize_date(raw: str) -> Optional[str]:
    """Validates LLM-returned dates and catches any natural language that slipped through."""
    if not raw:
        return None
    s = str(raw).strip()

    # Happy path — LLM already resolved to YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    # Safety net — LLM returned natural language anyway
    today = datetime.now().date()
    s_low = s.lower()

    if "today" in s_low:
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in s_low:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    # "next <weekday>" — always the FOLLOWING week's occurrence
    match = re.search(
        r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s_low
    )
    if match:
        target_day = weekday_map[match.group(1)]
        days_ahead = (target_day - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        days_ahead += 7  # "next" always skips to the following week
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "this <weekday>" — the coming occurrence within this week
    match = re.search(
        r"this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s_low
    )
    if match:
        target_day = weekday_map[match.group(1)]
        days_ahead = (target_day - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return None  # couldn't parse — router will re-ask for the date


def _normalize_time(raw: str) -> Optional[str]:
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
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        h, mn, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if mer == "pm" and h != 12:
            h += 12
        if mer == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mn:02d}"
    return s if re.match(r"^\d{2}:\d{2}$", s) else None


def _lookup_employee(search_term: str) -> Optional[Employee]:
    if not search_term or len(str(search_term)) < 2:
        return None
    clean = re.sub(
        r"\b(the|is|who|of|this|company|with|for|at|his|her|name|cabin|room|manager|engineer|lead|ceo)\b",
        "",
        str(search_term).lower(),
    ).strip()
    for key, dept in SEMANTIC_MAP.items():
        if re.search(r"\b" + re.escape(key) + r"\b", clean):
            clean = dept
            break
    if clean in ["admin", "administration", "front desk", "anyone"]:
        return Employee(
            name="Administration Team", role="Support", location="Reception"
        )
    db = SessionLocal()
    try:
        emp = (
            db.query(Employee)
            .filter(
                or_(
                    Employee.role.ilike(f"%{clean}%"),
                    Employee.name.ilike(f"%{clean}%"),
                    Employee.department.ilike(f"%{clean}%"),
                )
            )
            .first()
        )
        return emp
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def _notify_host_pending(state: Dict[str, Any]) -> bool:
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name") or "Guest"
        all_hosts = state.get("all_hosts") or [state["sched_employee_name"]]

        for host_name in all_hosts:
            if host_name in state["notified_hosts"]:
                continue  # skip already-notified hosts

            emp = _lookup_employee(host_name)
            narrative = _format_descriptive_purpose(state, "SCHEDULED")
            mid = schedule_meeting(
                v_name,
                "Visitor",
                host_name,
                state["sched_date"],
                state["sched_time"],
                narrative,
            )
            if mid == -1:
                return -1  # conflict
            if not mid:
                return False

            visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
            if not visitor:
                visitor = Visitor(name=v_name)
                db.add(visitor)
                db.flush()

            log = ReceptionLog(
                visitor_id=visitor.id,
                employee_id=emp.id if emp and emp.id else None,
                person_type=state["visitor_type"],
                purpose=narrative,
                check_in_time=datetime.utcnow(),
            )
            db.add(log)
            db.commit()

            db_emp = db.query(Employee).filter(Employee.name.ilike(host_name)).first()
            host_email = (
                db_emp.email
                if db_emp and db_emp.email
                else state.get("sched_employee_email", "")
            )
            host_slack_user_id = get_slack_user_id_by_email(host_email)
            send_slack_meeting_scheduled(
                host_name=host_name,
                host_email=host_email,
                visitor_name=v_name,
                date_str=state["sched_date"],
                time_str=state["sched_time"],
                purpose=state.get("sched_purpose") or state.get("purpose") or "Meeting",
                session_id=state["session_id"],
                host_slack_user_id=host_slack_user_id,
            )
            state["notified_hosts"].add(host_name)

        state["awaiting_slack_reply"] = True
        return True
    except Exception as e:
        logger.error(f"Notify-host-pending failed: {e}")
        return False
    finally:
        db.close()


def _book_calendar_event(state: Dict[str, Any]) -> bool:
    """Book the DB meeting record and Google Calendar event.
    Called only AFTER the host has confirmed via Slack."""
    try:
        v_name = state.get("visitor_name") or "Guest"
        host_name = state["sched_employee_name"]
        narrative = _format_descriptive_purpose(state, "SCHEDULED")

        mid = schedule_meeting(
            v_name,
            "Visitor",
            host_name,
            state["sched_date"],
            state["sched_time"],
            narrative,
        )
        if not mid:
            return False

        if state.get("sched_employee_email"):
            schedule_google_meeting_background(
                v_name,
                state["sched_employee_email"],
                state["sched_date"],
                state["sched_time"],
            )
        return True
    except Exception as e:
        logger.error(f"Calendar booking failed: {e}")
        return False


def _commit_checkin(state: Dict[str, Any]) -> bool:
    db = SessionLocal()
    try:
        is_employee = state.get("is_employee", False)
        v_name = state.get("visitor_name") or "Guest"
        narrative = _format_descriptive_purpose(state)

        if is_employee:
            # ── EMPLOYEE check-in: no visitor record, link directly to employee ──
            emp_id = state.get("verified_employee_id")
            if not emp_id:
                # Fallback: look up by name
                emp_record = _lookup_employee(v_name)
                emp_id = emp_record.id if emp_record and emp_record.id else None
            log = ReceptionLog(
                visitor_id=None,  # employees are NOT stored in visitors table
                employee_id=emp_id,
                person_type="Employee",
                purpose=narrative,
                check_in_time=datetime.utcnow(),
            )
        else:
            # ── VISITOR check-in: get-or-create visitor record ────────────────
            visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
            if not visitor:
                visitor = Visitor(name=v_name)
                db.add(visitor)
                db.flush()

            # Resolve the host employee for the log entry
            host_name = state.get("meeting_with_resolved")
            emp = _lookup_employee(host_name) if host_name else None

            log = ReceptionLog(
                visitor_id=visitor.id,
                employee_id=emp.id if emp and emp.id else None,
                person_type=state["visitor_type"],
                purpose=narrative,
                check_in_time=datetime.utcnow(),
            )

        db.add(log)
        db.commit()
        logger.info(
            f"Reception log created: id={log.id}, visitor={v_name}, type={log.person_type}"
        )
        return True
    except Exception as e:
        logger.error(f"Commit error: {e}")
        db.rollback()
        return False
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────────────────────────


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], user_query: str
) -> None:
    query_low = user_query.lower()

    # ── Visitor (speaker) name ────────────────────────────────────────────────
    v_name = entities.get("visitor_name")
    if v_name and not _is_jarvis(v_name):
        new_name = v_name.capitalize()

        # Case 1: Speaker explicitly claims to be an employee
        speaker_claims_employee = (
            "i am an employee" in query_low
            or "i work here" in query_low
            or bool(
                re.search(
                    r"\bi\s+am\b.{0,30}\b" + re.escape(new_name.lower()) + r"\b",
                    query_low,
                )
            )
        )
        if speaker_claims_employee:
            emp_record = get_employee_by_name(new_name)
            if emp_record:
                state["is_employee"] = True
                state["visitor_type"] = "Employee"
                state["verified_employee_id"] = emp_record.id

        # Case 2: Name matches an employee — flag for face verification.
        # Only triggered once per session (cleared after verification).
        elif not state.get("is_employee") and not state.get("face_verified_name"):
            emp_record = get_employee_by_name(new_name)
            if emp_record:
                state["pending_employee_face_check"] = True
                state["pending_employee_name"] = new_name
                state["pending_employee_id"] = emp_record.id

        # Update visitor name in session
        if state.get("visitor_name") and state["visitor_name"] != new_name:
            state["visitor_name"] = new_name
            state["identity_updated"] = True
        elif not state.get("visitor_name"):
            state["visitor_name"] = new_name

    # ── Visitor type from context clues ──────────────────────────────────────
    if not state.get("is_employee"):
        state["visitor_type"] = _determine_visitor_type(
            user_query, entities.get("purpose", ""), state["visitor_type"]
        )
    # ── MULTI-HOST EXTRACTION ────────────────────────────────────────────────
    raw_targets: List[str] = []

    # 1. From LLM entities (may already be a list)
    # AFTER
    for key in ("employee_name", "employee_role", "employee_names"):
        val = entities.get(key)
        if isinstance(val, list):
            raw_targets.extend(val)
        elif val:
            for part in re.split(r"\s+and\s+|\s*,\s*", val, flags=re.IGNORECASE):
                part = part.strip()
                if part:
                    raw_targets.append(part)

    # 2. Scan raw query for "with X and Y" patterns
    # Stops at noise words so "for me today" doesn't get captured
    and_pattern = re.compile(
        r"\b(?:with|meet|see)\b\s+([A-Za-z.\s]+?)(?:\s+(?:for|today|tomorrow|at|about|regarding|and\s+me)\b|$)",
        re.IGNORECASE,
    )
    m = and_pattern.search(user_query)
    if m:
        chunk = m.group(1).strip()
        for part in re.split(r"\s+and\s+|\s*&\s*", chunk, flags=re.IGNORECASE):
            part = part.strip(" .,")
            # Skip noise words and single letters
            if (
                part
                and len(part) > 1
                and part.lower() not in {"me", "my", "us", "them", "the", "a", "an"}
            ):
                raw_targets.append(part)

    # Deduplicate while preserving order
    seen_keys: set = set()
    unique_targets: List[str] = []
    for t in raw_targets:
        k = t.lower().strip()
        if k and k not in seen_keys:
            seen_keys.add(k)
            unique_targets.append(t)

    # ── HOST RESOLUTION (fixed) ───────────────────────────────────────────────
    # Only run if new targets were identified in this turn.
    # This prevents overwriting previous multi-host arrays when user just says "yes" or "confirm".
    if unique_targets:
        resolved_hosts: List[str] = []
        primary_emp = None

        for target in unique_targets:
            if _is_jarvis(target):
                continue

            # 1. Strict name lookup (exact → prefix → high-cutoff fuzzy)
            emp = get_employee_by_name(target)

            # 2. Word-in-name + looser fuzzy (catches "Jack" inside "Jack Smith")
            if not emp:
                emp = get_similar_employee(target)

            # 3. Semantic/role/department fallback (for non-name inputs)
            if not emp:
                emp = _lookup_employee(target)

            logger.debug(f"[host-resolve] '{target}' → '{emp.name if emp else None}'")

            if emp:
                if emp.name not in resolved_hosts:
                    resolved_hosts.append(emp.name)
                if primary_emp is None:
                    primary_emp = emp
            else:
                if target not in resolved_hosts:
                    resolved_hosts.append(target)

        if resolved_hosts:
            state["meeting_with_raw"] = unique_targets[0]
            state["meeting_with_resolved"] = resolved_hosts[0]
            state["sched_employee_name"] = resolved_hosts[0]
            if primary_emp:
                state["sched_employee_email"] = primary_emp.email
            state["all_hosts"] = resolved_hosts

    # ── Date / time / purpose ─────────────────────────────────────────────────
    if entities.get("date"):
        new_date = _normalize_date(str(entities["date"]))
        if new_date and new_date != state.get("sched_date"):
            state["sched_date"] = new_date
            state["host_preapproved"] = False  # Reset

    if entities.get("time"):
        new_time = _normalize_time(str(entities["time"]))
        if new_time and new_time != state.get("sched_time"):
            state["sched_time"] = new_time
            state["host_preapproved"] = False  # Reset

    if entities.get("purpose"):
        state["purpose"] = state["sched_purpose"] = entities["purpose"]


async def _handle_availability_check(
    state: Dict[str, Any], query: str, client_id: str
) -> str:
    target = state.get("meeting_with_resolved") or query
    emp = _lookup_employee(target)
    if emp:
        today_str = datetime.now().strftime("%Y-%m-%d")
        after_time = datetime.now().strftime("%H:%M")
        slots = get_available_slots(emp.name, today_str, after_time=after_time)
        if slots:
            slot_str = ", ".join(_fmt_display(s) for s in slots[:3])
        else:
            slot_str = "no more slots today"
        return await _llm_reply(
            f"Inform them {emp.name} is the {emp.role}. Free slots: {slot_str}.",
            state,
            query,
            client_id,
        )
    return await _llm_reply(
        "Apologize you couldn't check availability.", state, query, client_id
    )


async def _handle_directory_lookup(
    state: Dict[str, Any], query: str, client_id: str
) -> str:
    emp = _lookup_employee(query)
    if emp:
        state["meeting_with_resolved"] = emp.name
        return await _llm_reply(
            f"Tell them {emp.name} is the {emp.role} at {emp.location}.",
            state,
            query,
            client_id,
        )
    return await _llm_reply(
        "Apologize politely that you couldn't find them.", state, query, client_id
    )


async def _finalize_checkin_and_respond(state, query, client_id):
    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name politely.", state, query, client_id)
    if state.get("is_employee"):
        _EMPLOYEE_ACTION_PHRASES = [
            "notify",
            "tell",
            "inform",
            "let them know",
            "let him know",
            "let her know",
            "please notify",
            "report",
            "issue",
            "problem",
            "temperature",
            "ac",
            "fix",
            "broken",
            "maintenance",
        ]
        is_action_request = any(w in query.lower() for w in _EMPLOYEE_ACTION_PHRASES)
        if not is_action_request:
            _commit_checkin(state)  # log the arrival in DB without notifying anyone
            state["conv_state"] = "COMPLETED"
            return await _llm_reply(
                "Identity verified. Wish the staff member a great day and let them proceed.",
                state,
                query,
                client_id,
            )

    if not state.get("meeting_with_resolved"):
        semantic_host = _lookup_employee(state.get("purpose", "") + " " + query)
        if semantic_host and semantic_host.name != "Administration Team":
            state["meeting_with_resolved"] = semantic_host.name

    # Build the full list of hosts to notify — fall back to single host if needed
    all_hosts: List[str] = state.get("all_hosts") or []
    if not all_hosts:
        fallback = state.get("meeting_with_resolved") or "Administration Team"
        all_hosts = [fallback]
        state["all_hosts"] = all_hosts  # persist so Slack uses the right list

    newly_notified: List[str] = []
    db = SessionLocal()
    try:
        for host in all_hosts:
            if host not in state["notified_hosts"]:
                # Look up the employee's Slack user ID from DB
                db_emp = db.query(Employee).filter(Employee.name.ilike(host)).first()
                host_slack_user_id = (
                    db_emp.slack_user_id
                    if db_emp and hasattr(db_emp, "slack_user_id")
                    else None
                )

                send_slack_arrival(
                    host,
                    state["visitor_name"],
                    state["visitor_type"],
                    state.get("purpose", "Arrival"),
                    state["session_id"],
                    host_slack_user_id=host_slack_user_id,  # ← new
                )
                state["notified_hosts"].add(host)
                newly_notified.append(host)
                logger.info(
                    f"Slack notification sent to '{host}' for visitor '{state['visitor_name']}'"
                )
    finally:
        db.close()

    if newly_notified:
        state["awaiting_slack_reply"] = True
        _commit_checkin(state)

    state["conv_state"] = "COMPLETED"
    host_display = " and ".join(all_hosts)
    situation = f"Confirm {host_display} {'has' if len(all_hosts) == 1 else 'have'} been notified."
    if "Delivery" in state["visitor_type"]:
        situation += " Tell them to leave the item at reception."
    elif state.get("is_employee"):
        situation += (
            " Tell them the team has been notified and will attend to it shortly."
        )
    else:
        situation += " Ask them to take a seat in the lobby and wait."
    return await _llm_reply(situation, state, query, client_id)


async def _handle_scheduling(
    client_id: str, query: str, state: Dict[str, Any], intent: str
) -> str:
    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name.", state, query, client_id)
    if not state.get("sched_employee_name"):
        return await _llm_reply("Ask who they want to meet.", state, query, client_id)
    if not state.get("sched_date"):
        return await _llm_reply("Ask for the date.", state, query, client_id)
    if not state.get("sched_time"):
        return await _llm_reply("Ask for the time.", state, query, client_id)

    # --- NEW: STRICT OFFICE HOURS CHECK ---
    # If the visitor asks for a time after 19:00 (7:00 PM)
    if state["sched_time"] > "19:00" and not state.get("host_preapproved"):
        state["sched_time"] = None  # Clear the invalid time so the AI asks again
        return await _llm_reply(
            "Politely inform the visitor that our office hours end at 7:00 PM, "
            "so meetings cannot be scheduled after this time. Ask them to choose an earlier time.",
            state,
            query,
            client_id,
        )
    # --------------------------------------

    today_str = datetime.now().strftime("%Y-%m-%d")
    after_time = (
        datetime.now().strftime("%H:%M") if state["sched_date"] == today_str else None
    )
    before_time = state.get("sched_time_before")

    host_after = state.get("sched_time_after")
    if host_after:
        after_time = max(after_time, host_after) if after_time else host_after

    slots = get_available_slots(
        state["sched_employee_name"],
        state["sched_date"],
        after_time=after_time,
        before_time=before_time,
    )

    # ✅ Grace window: don't disqualify visitor's requested time due to
    # processing delay (up to 15 mins tolerance)
    requested = state.get("sched_time")
    if requested and requested not in slots and requested in ALL_SLOTS_1H:
        now_dt = datetime.strptime(datetime.now().strftime("%H:%M"), "%H:%M")
        req_dt = datetime.strptime(requested, "%H:%M")
        diff_minutes = (now_dt - req_dt).total_seconds() / 60
        if diff_minutes <= 15:
            slots.append(requested)
            slots.sort()

    state.pop("sched_time_before", None)
    state.pop("sched_time_after", None)
    # --- 3. NEW: Trust the host's time even if outside normal DB hours ---
    if state.get("host_preapproved") and state.get("sched_time"):
        if state["sched_time"] not in slots:
            slots.append(state["sched_time"])
            slots.sort()

    if not slots:
        if before_time and after_time:
            constraint_desc = (
                f"between {_fmt_display(after_time)} and {_fmt_display(before_time)}"
            )
        elif before_time:
            constraint_desc = f"before {_fmt_display(before_time)}"
        elif after_time:
            constraint_desc = f"after {_fmt_display(after_time)}"
        else:
            constraint_desc = "today"

        return await _llm_reply(
            f"Tell the visitor there are no available slots {constraint_desc}. "
            f"Ask if they'd like a different day.",
            state,
            query,
            client_id,
        )

    # In _handle_scheduling, around line 746
    logger.info(f"DEBUG slots={slots} sched_time={state['sched_time']}")

    if state["sched_time"] not in slots:
        # Only suggest alternatives if NOT already confirmed by visitor
        if any(
            x in query.lower()
            for x in ["okay", "yes", "ok", "sure", "do it", "agreed", "that time"]
        ):
            # Visitor is confirming — trust the time, add it to slots
            slots.append(state["sched_time"])
        else:
            display_slots = [_fmt_display(s) for s in slots[:3]]
            return await _llm_reply(
                f"The requested time is unavailable. Suggest these alternate slots: "
                f"{', '.join(display_slots)}.",
                state,
                query,
                client_id,
            )

    if intent == "confirm" or any(
        x in query.lower()
        for x in [
            "okay",
            "yes",
            "correct",
            "book",
            "sure",
            "please",
            "ok",
            "do it",
            "perfect",
        ]
    ):
        # --- 4. NEW: Direct booking if host pre-approved ---
        if state.get("host_preapproved"):
            _book_calendar_event(state)
            state["scheduling_active"] = False
            state["conv_state"] = State.COMPLETED
            state["host_preapproved"] = False  # Clear state

            hosts_display = " and ".join(
                state.get("all_hosts") or [state["sched_employee_name"]]
            )
            time_disp = _fmt_display(state["sched_time"])
            return await _llm_reply(
                f"Tell the visitor their meeting is confirmed with {hosts_display} at {time_disp}. Ask them to wait.",
                state,
                query,
                client_id,
            )
        # ---------------------------------------------------

        result = _notify_host_pending(state)
        if result == -1:
            return await _llm_reply(
                f"Inform the visitor that the slot at {_fmt_display(state['sched_time'])} is already booked. Ask them to choose another time.",
                state,
                query,
                client_id,
            )
        elif result:
            # Keep scheduling_active — session is NOT done yet.
            state["scheduling_active"] = True
            state["awaiting_slack_reply"] = True
            hosts_display = " and ".join(
                state.get("all_hosts") or [state["sched_employee_name"]]
            )
            time = _fmt_display(state["sched_time"])
            return await _llm_reply(
                f"Tell the visitor: 'I have notified {hosts_display} for {time}. "
                f"Please give me a moment while I wait for their confirmation.'",
                state,
                query,
                client_id,
            )
        else:
            return await _llm_reply(
                "Apologize and say there was an error scheduling the meeting.",
                state,
                query,
                client_id,
            )

    hosts_display = " and ".join(
        state.get("all_hosts") or [state["sched_employee_name"]]
    )
    return await _llm_reply(
        f"Ask the visitor to confirm if they want to schedule the meeting with {hosts_display} at {_fmt_display(state['sched_time'])}.",
        state,
        query,
        client_id,
    )


async def route_query(client_id: str, user_query: str) -> str:
    from receptionist.database import get_company_details
    from models.groq_processor import GroqProcessor, BASE_SYSTEM_PROMPT

    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_low = user_query.lower().strip()

    # Wake word → fresh session
    if any(x in query_low for x in WAKE_WORDS):
        if state.get("awaiting_slack_reply"):
            return "I am still waiting for the host to reply. Please give me just a moment."

        clear_session_state(client_id)
        state = get_session_state(client_id)
        state["greeted"] = True
        return (
            f"{_get_time_greeting()}! Welcome to {COMPANY_NAME}. "
            f"I am {AI_NAME}, how can I assist you today?"
        )

    # ── 1. CHECK FOR PENDING SLACK REPLY ─────────────────────────────────────
    if user_query.startswith("SLACK_REPLY:"):
        parts = user_query.split(":", 2)
        host_name = parts[1] if len(parts) > 1 else "Host"
        slack_reply = parts[2] if len(parts) > 2 else ""

        state["awaiting_slack_reply"] = False

        if state.get("scheduling_active"):
            _CONFIRM_KEYWORDS = {
                "yes",
                "ok",
                "okay",
                "confirmed",
                "approve",
                "sure",
                "sounds good",
                "works",
                "fine",
                "perfect",
                "go ahead",
                "agreed",
                "accept",
            }
            reply_lower = slack_reply.lower()
            host_confirmed = any(kw in reply_lower for kw in _CONFIRM_KEYWORDS)

            if host_confirmed:
                _book_calendar_event(state)
                state["scheduling_active"] = False
                state["conv_state"] = State.COMPLETED

                # ✅ First responder wins — stop polling all other host threads
                from services.slack_reply_poller import unregister_thread

                unregister_thread(state["session_id"])

                sched_display = _fmt_display(state.get("sched_time") or "")
                all_hosts = state.get("all_hosts") or [host_name]
                hosts_display = " and ".join(all_hosts)
                situation = (
                    f"{host_name} confirmed the meeting. "
                    f"Tell the visitor the meeting is confirmed with {hosts_display} "
                    f"at {sched_display}. Ask them to wait in the lobby."
                )

            else:
                # ✅ Mark this host as declined and check if others are still pending
                state.setdefault("declined_hosts", set()).add(host_name)
                remaining = [
                    h
                    for h in state.get("all_hosts", [])
                    if h not in state["declined_hosts"]
                ]

                if remaining:
                    # Other hosts haven't replied yet — keep waiting silently
                    state["awaiting_slack_reply"] = True
                    return await _llm_reply(
                        f"{host_name} is unavailable at the requested time. "
                        f"Still waiting for confirmation from {' and '.join(remaining)}. "
                        f"Tell the visitor you are still checking with the other host(s).",
                        state,
                        "[System: Slack Reply Received]",
                        client_id,
                    )

                # All hosts have declined — fall through to reschedule logic
                time_match = re.search(
                    r"\b((?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:am|pm)?|(?:[01]?\d|2[0-3]):[0-5][0-9])\b",
                    slack_reply,
                    re.IGNORECASE,
                )

                if time_match:
                    constraint_time_24 = _normalize_time(time_match.group(1))

                    if constraint_time_24:
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        tomorrow_str = (datetime.now() + timedelta(days=1)).strftime(
                            "%Y-%m-%d"
                        )
                        if "today" in reply_lower:
                            state["sched_date"] = today_str
                        elif "tomorrow" in reply_lower:
                            state["sched_date"] = tomorrow_str

                        before_match = re.search(r"\bbefore\b", reply_lower)
                        after_match = re.search(r"\b(after|from)\b", reply_lower)

                        if before_match:
                            state["sched_time_before"] = constraint_time_24
                            state["sched_time"] = None
                        elif after_match:
                            state["sched_time_after"] = constraint_time_24
                            state["sched_time"] = None
                        else:
                            state["sched_time"] = constraint_time_24
                            state["host_preapproved"] = True

                    constraint_desc = ""
                    if state.get("sched_time_before"):
                        constraint_desc = (
                            f"before {_fmt_display(state['sched_time_before'])}"
                        )
                    elif state.get("sched_time_after"):
                        constraint_desc = (
                            f"after {_fmt_display(state['sched_time_after'])}"
                        )
                    elif state.get("sched_time"):
                        constraint_desc = f"at {_fmt_display(state['sched_time'])}"

                    situation = (
                        f"{host_name} responded to the scheduling request. "
                        f'Their reply indicates: "{slack_reply}". '
                        f"{'The host proposes a time ' + constraint_desc + '. ' if constraint_desc else ''}"
                        f"Understand their intent and respond to the visitor professionally. "
                        f"Do NOT quote the host's raw message. Instead, paraphrase their response naturally "
                        f"(e.g. say '{host_name} suggested...' or '{host_name} is available at...')."
                    )

                else:
                    # No time found in reply at all → soft decline
                    state["scheduling_active"] = False
                    state["awaiting_slack_reply"] = False
                    situation = (
                        f"{host_name} responded to the scheduling request. "
                        f'Their reply indicates: "{slack_reply}". '
                        f"Understand their intent and respond to the visitor professionally. "
                        f"Do NOT quote the host's raw message directly. Instead, interpret their response "
                        f"and communicate the meaning naturally (e.g. '{host_name} is currently unavailable' "
                        f"or '{host_name} suggested an alternative'). If it seems like a decline, "
                        f"apologize and offer to reschedule or contact someone else."
                    )

                return await _llm_reply(
                    situation, state, "[System: Slack Reply Received]", client_id
                )

            # Reached only on host_confirmed path
            return await _llm_reply(
                situation, state, "[System: Slack Reply Received]", client_id
            )

        # Non-scheduling flow (arrival notification reply)
        situation = (
            f"{host_name} responded via Slack. "
            f'Their reply indicates: "{slack_reply}". '
            f"Understand the intent of their reply and respond to the visitor professionally. "
            f"Do NOT quote the host's raw message directly. Instead, interpret what they said "
            f"and communicate it naturally (e.g. if they said 'ok I will collect it', say "
            f"'{host_name} has confirmed and will come to collect it shortly'). "
            f"Be helpful and keep the response concise."
        )
        return await _llm_reply(
            situation, state, "[System: Slack Reply Received]", client_id
        )
    # ─────────────────────────────────────────────────────────────────────────

    # ── 2. ENTITY EXTRACTION ON ACTUAL USER SPEECH ───────────────────────────
    extracted = await llm.extract_intent_and_entities(user_query)
    intent = extracted.get("intent", "general")
    entities = extracted.get("entities", {})
    _merge_checkin_entities(state, entities, user_query)
    logger.info(f"DEBUG all_hosts after merge = {state.get('all_hosts')}")
    logger.info(f"DEBUG entities from LLM = {entities}")

    if any(
        x in query_low
        for x in [
            "don't know",
            "do not know",
            "anyone",
            "notify admin",
            "admin team",
            "administration team",
            "notify the admin",
        ]
    ):
        state["force_admin"] = True
        state["meeting_with_resolved"] = "Administration Team"
        state["all_hosts"] = ["Administration Team"]

    is_farewell = any(w in query_low for w in ["thank you", "thanks", "bye", "goodbye"])
    if is_farewell:
        if state.get("scheduling_active") and not state.get("awaiting_slack_reply"):
            if any(
                x in query_low
                for x in [
                    "yes",
                    "okay",
                    "ok",
                    "sure",
                    "please",
                    "do it",
                    "correct",
                    "book",
                ]
            ):
                pass  # fall through to scheduling confirmation
            else:
                reply = await _llm_reply(
                    "Give a single warm farewell sentence. Use the visitor's name if you know it. Do NOT list multiple options.",
                    state,
                    user_query,
                    client_id,
                )
                clear_session_state(client_id)
                return reply
        elif state.get("awaiting_slack_reply"):
            return await _llm_reply(
                "Acknowledge their thanks politely. If they are waiting for a meeting confirmation, remind them you are still waiting for the host's reply. Otherwise, just tell them to have a seat.",
                state,
                user_query,
                client_id,
            )
        else:
            reply = await _llm_reply(
                "Give a single warm farewell sentence. Use the visitor's name if you know it. Do NOT list multiple options.",
                state,
                user_query,
                client_id,
            )
            clear_session_state(client_id)
            return reply

    if any(x in query_low for x in ["free time", "available", "is he free"]):
        return await _handle_availability_check(state, user_query, client_id)

    _LOOKUP_ONLY_PHRASES = ["who is", "where is", "which floor", "what department"]
    _NOTIFY_PHRASES = [
        "notify",
        "tell",
        "inform",
        "waiting",
        "let him know",
        "let her know",
        "please notify",
        "i'm here",
        "i am here",
    ]
    is_notify_intent = any(w in query_low for w in _NOTIFY_PHRASES)
    is_lookup_intent = intent == "employee_lookup" or any(
        x in query_low for x in _LOOKUP_ONLY_PHRASES
    )
    if is_lookup_intent and not is_notify_intent:
        return await _handle_directory_lookup(state, user_query, client_id)

    if intent == "schedule_meeting" or state["scheduling_active"]:
        state["scheduling_active"] = True
        return await _handle_scheduling(client_id, user_query, state, intent)

    if intent == "check_in" or state["meeting_with_resolved"]:
        return await _finalize_checkin_and_respond(state, user_query, client_id)

    return await llm.get_response(
        client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
    )


async def _llm_reply(
    situation: str, state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    llm = GroqProcessor.get_instance()
    visitor = state.get("visitor_name") or "Visitor"
    host = (
        state.get("meeting_with_resolved")
        or state.get("sched_employee_name")
        or "Admin"
    )
    now = datetime.now()
    hour = now.hour % 12 or 12
    now_str = f"{hour}:{now.strftime('%M %p on %A, %d %B %Y')}"
    time_context = f"[Current time: {now_str} — use for scheduling logic ONLY, do NOT mention to the visitor unless they ask] "

    info = f"TALKING TO: {visitor} | HOST: {host} | STATUS: {state['conv_state']}"
    is_first = not state.get("greeted", False)
    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"KB: {info}\n"
        f"{time_context}"
        f"TIME FORMAT RULE: Always say times in 12-hour format with AM/PM (e.g. '1:30 PM', not '13:30').\n"
        f"RULES: {'Greet warmly' if is_first else 'DO NOT greet again'}. "
        f"GOAL: {situation}\n"
        f"USER: {user_query}"
    )
    resp = await llm.get_raw_response(prompt, client_id=client_id)
    state["greeted"] = True
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# FACE VERIFICATION INTEGRATION HELPER
# Called by the WebSocket handler after running DeepFace verification
# when a visitor's name matched an employee record.
# ─────────────────────────────────────────────────────────────────────────────


def mark_employee_from_face_result(
    client_id: str,
    face_verified: bool,
    employee_id: Optional[int] = None,
    employee_name: Optional[str] = None,
) -> None:
    """
    Called by the WebSocket layer after face verification.

    If face_verified=True  → promote session to employee.
    If face_verified=False → confirm as visitor.
    """
    state = get_session_state(client_id)

    if face_verified:
        state["is_employee"] = True
        state["visitor_type"] = "Employee"
        # Use provided ID/name if available (from direct face match),
        # otherwise fall back to the pending name check logic.
        if employee_id:
            state["verified_employee_id"] = employee_id
        else:
            state["verified_employee_id"] = state.get("pending_employee_id")

        if employee_name:
            state["visitor_name"] = employee_name
            state["face_verified_name"] = employee_name

        logger.info(
            "[face] '%s' confirmed as employee via face verification.",
            state.get("visitor_name") or state.get("pending_employee_name"),
        )
    else:
        logger.info("[face] Face did NOT match employee record — treating as visitor.")

    # If we didn't have a direct name but had a pending name, mark it as checked
    if not state.get("face_verified_name"):
        state["face_verified_name"] = state.get("pending_employee_name")

    # Clear the pending flags regardless of outcome
    state.pop("pending_employee_face_check", None)
    state.pop("pending_employee_name", None)
    state.pop("pending_employee_id", None)
