import logging
import platform
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from sqlalchemy import or_

from receptionist.database import SessionLocal
from receptionist.models import Employee, Visitor, Meeting
from models.groq_processor import GroqProcessor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PER-CLIENT SESSION STATE
# ─────────────────────────────────────────────

_client_sessions: Dict[str, Dict[str, Any]] = {}


def get_session_state(client_id: str) -> Dict[str, Any]:
    if client_id not in _client_sessions:
        _client_sessions[client_id] = _fresh_state()
    return _client_sessions[client_id]


def clear_session_state(client_id: str) -> None:
    _client_sessions[client_id] = _fresh_state()


def _fresh_state() -> Dict[str, Any]:
    return {
        "visitor_name": None,
        "intent": None,
        "meeting_with": None,
        "purpose": None,
        "identity": "UNKNOWN",
        "checkin_done": False,
        "log_id": None,
        "employee_name": None,
        "time": None,
        "date": None,
        # scheduling sub-flow
        "scheduling_active": False,
        "sched_employee": None,
        "sched_date": None,
        "sched_time": None,
        "sched_purpose": None,
        "sched_pending_confirm": False,
    }


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PRONOUNS = {
    "him", "her", "them", "he", "she", "that person",
    "someone", "this guy", "they", "it",
}

_SCHEDULING_NOISE = re.compile(
    r"^(can (he|she|they|you)|schedule|help me|also|please confirm|confirm this|"
    r"would like to meet|i would|i want|i need|yes|no|okay|ok|sure)$",
    re.IGNORECASE,
)

COMPANY_NAME = "Sharp Software Development India Pvt Ltd"

# Short utterances that carry no information and should not advance the flow
_FILLER_UTTERANCES = re.compile(
    r"^\s*(thank you|thanks|okay|ok|sure|alright|alright then|"
    r"got it|great|perfect|sounds good|cool|nice|yep|yup|mm-hmm|uh-huh|hmm)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _is_filler(text: str) -> bool:
    return bool(_FILLER_UTTERANCES.match(text.strip()))


def _fmt_date(dt: datetime) -> str:
    """Cross-platform date format: 'Thursday, April 3' (no leading zero, works on Windows)."""
    day = dt.day  # plain int, no format flag needed
    return dt.strftime("%A, %B ") + str(day)


def _fmt_time(dt: datetime) -> str:
    """Cross-platform time format: '2:00 PM' (no leading zero, works on Windows)."""
    hour = dt.hour % 12 or 12
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p")
    return f"{hour}:{minute} {ampm}"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\.", "", str(text)).strip()


def _resolve_datetime(date_raw: Optional[str], time_raw: Optional[str]) -> Optional[datetime]:
    now = datetime.now()

    if not date_raw:
        base_date = now.date()
    else:
        s = date_raw.lower().strip()
        if "today" in s:
            base_date = now.date()
        elif "tomorrow" in s:
            base_date = (now + timedelta(days=1)).date()
        else:
            day_names = ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"]
            base_date = now.date()
            for i, d in enumerate(day_names):
                if d in s:
                    days_ahead = (i - now.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    base_date = (now + timedelta(days=days_ahead)).date()
                    break

    hour, minute = 9, 0
    if time_raw:
        time_match = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", time_raw.lower())
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            meridiem = time_match.group(3)
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            elif not meridiem and 1 <= hour <= 7:
                hour += 12

    return datetime(base_date.year, base_date.month, base_date.day, hour, minute)


def _is_noise_purpose(text: str) -> bool:
    if not text:
        return True
    stripped = text.strip().rstrip(".?!")
    return bool(_SCHEDULING_NOISE.match(stripped))


def _merge_entities(state: Dict[str, Any], entities: Dict[str, Any], raw_query: str) -> None:
    v_name = entities.get("visitor_name") or entities.get("name")
    if v_name and str(v_name).lower().strip() not in PRONOUNS:
        candidate = str(v_name).strip().capitalize()
        if len(candidate) > 1:
            state["visitor_name"] = candidate

    emp = entities.get("employee_name")
    role = entities.get("role")
    target = emp or role
    if target:
        clean_target = _normalise(target)
        if clean_target.lower() not in PRONOUNS and len(clean_target) > 1:
            if not state["meeting_with"] or emp:
                state["meeting_with"] = clean_target
                state["employee_name"] = clean_target

    purpose = entities.get("purpose")
    if purpose and not _is_noise_purpose(str(purpose)) and len(str(purpose)) > 4:
        state["purpose"] = str(purpose).strip().capitalize()
    elif not state.get("purpose"):
        patterns = [
            r"(?:regarding|about|concerning|purpose is|here (?:to|for)) (.+)",
            r"i (?:am|'m) here (?:to|for) (.+)",
            r"(?:to discuss|to talk about|to review) (.+)",
        ]
        for p in patterns:
            m = re.search(p, raw_query.lower())
            if m:
                extracted = m.group(1).strip().rstrip(".?!")
                if len(extracted) > 3 and not _is_noise_purpose(extracted):
                    state["purpose"] = extracted.capitalize()
                    break

    if entities.get("time"):
        state["time"] = entities.get("time")
    if entities.get("date"):
        state["date"] = entities.get("date")
    if entities.get("intent"):
        state["intent"] = entities.get("intent")


def _lookup_employee(search_term: str) -> Optional[Any]:
    if not search_term:
        return None
    clean = _normalise(search_term)
    db = SessionLocal()
    try:
        return db.query(Employee).filter(
            or_(
                Employee.name.ilike(f"%{clean}%"),
                Employee.role.ilike(f"%{clean}%"),
                Employee.department.ilike(f"%{clean}%"),
            )
        ).first()
    finally:
        db.close()


def _build_company_info(state: Dict[str, Any], emp: Optional[Any] = None) -> dict:
    info: dict = {
        "company_name": COMPANY_NAME,
        "visitor_name": state.get("visitor_name") or "Visitor",
    }
    if emp:
        info["dynamic_employee"] = (
            f"Name: {emp.name} | Role: {emp.role} | "
            f"Department: {emp.department} | Floor: {emp.floor} | "
            f"Cabin: {emp.cabin_number} | Extension: {emp.extension}"
        )
    return info


# ─────────────────────────────────────────────
# LLM-GENERATED RECEPTIONIST REPLIES
# ─────────────────────────────────────────────

async def _llm_reply(situation: str, context: dict) -> str:
    """
    Ask the LLM to generate a single natural receptionist line.
    Every visitor-facing message goes through here.

    `situation` — what the receptionist needs to communicate right now.
    `context`   — verified facts (names, floor, cabin, etc.) the LLM may use.
                  The LLM must not invent anything not listed here.
    """
    llm = GroqProcessor.get_instance()

    ctx_lines = []
    if context.get("visitor_name"):
        ctx_lines.append(f"Visitor name: {context['visitor_name']}")
    if context.get("employee_name"):
        ctx_lines.append(f"Employee name: {context['employee_name']}")
    if context.get("employee_role"):
        ctx_lines.append(f"Employee role: {context['employee_role']}")
    if context.get("floor"):
        ctx_lines.append(f"Floor: {context['floor']}")
    if context.get("cabin"):
        ctx_lines.append(f"Cabin: {context['cabin']}")
    if context.get("date_str"):
        ctx_lines.append(f"Meeting date: {context['date_str']}")
    if context.get("time_str"):
        ctx_lines.append(f"Meeting time: {context['time_str']}")
    if context.get("purpose"):
        ctx_lines.append(f"Meeting purpose: {context['purpose']}")
    if context.get("company_name"):
        ctx_lines.append(f"Company: {context['company_name']}")

    ctx_block = "\n".join(ctx_lines) if ctx_lines else "No additional context."

    prompt = f"""You are AlmostHuman, a professional corporate receptionist speaking out loud to a visitor standing at the front desk.

VERIFIED CONTEXT — use ONLY these facts, never invent anything:
{ctx_block}

YOUR TASK RIGHT NOW:
{situation}

RULES:
- Exactly 1 sentence. Maximum 25 words.
- Use the visitor's name naturally if provided.
- Use ONLY facts listed above. Never guess or invent.
- No filler openers: no "Certainly", "Of course", "Sure", "Absolutely", "Great".
- Never offer to call ahead, escort, or check real-time availability.
- Sound warm and natural — like a real human receptionist.
- Do not start the sentence with the word "I"."""

    return await llm.get_raw_response(prompt)


# ─────────────────────────────────────────────
# SCHEDULING SUB-FLOW
# ─────────────────────────────────────────────

async def _handle_scheduling(
    client_id: str,
    user_query: str,
    state: Dict[str, Any],
    entities: Dict[str, Any],
) -> Optional[str]:

    # Seed from check-in state where available
    if not state["sched_employee"]:
        emp_candidate = entities.get("employee_name") or state.get("meeting_with")
        if emp_candidate and _normalise(emp_candidate).lower() not in PRONOUNS:
            state["sched_employee"] = _normalise(emp_candidate)

    if not state["sched_date"] and entities.get("date"):
        state["sched_date"] = entities.get("date")
    if not state["sched_time"] and entities.get("time"):
        state["sched_time"] = entities.get("time")
    if not state["sched_purpose"]:
        purpose = entities.get("purpose")
        if purpose and not _is_noise_purpose(purpose):
            state["sched_purpose"] = purpose
        elif state.get("purpose"):
            state["sched_purpose"] = state["purpose"]

    if not state["sched_time"]:
        tm = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)", user_query.lower())
        if tm:
            state["sched_time"] = tm.group(0).strip()
    if not state["sched_date"]:
        for word in ["today", "tomorrow", "monday", "tuesday", "wednesday",
                     "thursday", "friday", "saturday", "sunday"]:
            if word in user_query.lower():
                state["sched_date"] = word
                break

    # Collect missing slots
    if not state["sched_employee"]:
        return await _llm_reply(
            "Ask the visitor who they would like to schedule a meeting with.",
            {"visitor_name": state.get("visitor_name")},
        )

    emp = _lookup_employee(state["sched_employee"])
    if not emp:
        bad_name = state["sched_employee"]
        state["sched_employee"] = None
        return await _llm_reply(
            f"Tell the visitor that '{bad_name}' was not found in the directory and ask them to check the name.",
            {"visitor_name": state.get("visitor_name")},
        )

    if not state["sched_date"]:
        return await _llm_reply(
            f"Ask the visitor what date they would like to meet {emp.name}.",
            {"visitor_name": state.get("visitor_name"), "employee_name": emp.name},
        )

    if not state["sched_time"]:
        return await _llm_reply(
            "Ask the visitor what time they would prefer for the meeting.",
            {"visitor_name": state.get("visitor_name"), "employee_name": emp.name},
        )

    if not state["sched_purpose"]:
        return await _llm_reply(
            f"Ask the visitor what the purpose of the meeting with {emp.name} is.",
            {"visitor_name": state.get("visitor_name"), "employee_name": emp.name},
        )

    # All slots filled — confirmation
    dt = _resolve_datetime(state["sched_date"], state["sched_time"])
    date_str = _fmt_date(dt) if dt else state["sched_date"]
    time_str = _fmt_time(dt) if dt else state["sched_time"]

    if not state["sched_pending_confirm"]:
        state["sched_pending_confirm"] = True
        return await _llm_reply(
            "Read back the meeting details to the visitor and ask them to confirm yes or no.",
            {
                "visitor_name": state.get("visitor_name"),
                "employee_name": emp.name,
                "date_str": date_str,
                "time_str": time_str,
                "purpose": state["sched_purpose"],
            },
        )

    # Resolve confirmation
    intent = entities.get("intent", "")
    confirm_words = re.search(
        r"\b(yes|confirm|proceed|go ahead|sure|okay|ok|please|yep|yup)\b",
        user_query.lower(),
    )
    cancel_words = re.search(
        r"\b(no|cancel|stop|don't|never mind|nope)\b",
        user_query.lower(),
    )

    def _reset_sched():
        for key in ["scheduling_active", "sched_employee", "sched_date",
                    "sched_time", "sched_purpose", "sched_pending_confirm"]:
            state[key] = False if key == "scheduling_active" else None

    if intent == "cancel" or cancel_words:
        _reset_sched()
        return await _llm_reply(
            "Tell the visitor their meeting request has been cancelled.",
            {"visitor_name": state.get("visitor_name")},
        )

    if intent == "confirm" or confirm_words:
        db = SessionLocal()
        try:
            meeting = Meeting(
                organizer_name=state.get("visitor_name") or "Visitor",
                organizer_type="visitor",
                visitor_name=state.get("visitor_name") or "Visitor",
                employee_name=emp.name,
                employee_email=emp.email,
                meeting_date=dt.date().isoformat() if dt else state["sched_date"],
                meeting_time=dt.strftime("%H:%M") if dt else state["sched_time"],
                scheduled_time=dt,
                purpose=state["sched_purpose"],
                status="scheduled",
                created_at=datetime.utcnow().isoformat(),
            )
            db.add(meeting)
            db.commit()
        except Exception as e:
            logger.error("Failed to save meeting: %s", e)
            db.rollback()
        finally:
            db.close()

        try:
            from services.notify_email import send_calendar_invite
            send_calendar_invite(
                employee_name=emp.name,
                employee_email=emp.email,
                organizer_name=state.get("visitor_name") or "Visitor",
                meeting_date=dt.date().isoformat() if dt else state["sched_date"],
                meeting_time=dt.strftime("%H:%M") if dt else state["sched_time"],
                purpose=state["sched_purpose"],
            )
        except Exception as e:
            logger.warning("Email notification failed: %s", e)

        _reset_sched()
        return await _llm_reply(
            f"Tell the visitor their meeting with {emp.name} on {date_str} at {time_str} is confirmed and a confirmation will be sent.",
            {
                "visitor_name": state.get("visitor_name"),
                "employee_name": emp.name,
                "date_str": date_str,
                "time_str": time_str,
            },
        )

    # Ambiguous — re-ask
    return await _llm_reply(
        "The visitor's response was unclear. Ask them simply to say yes or no to confirm the meeting.",
        {
            "visitor_name": state.get("visitor_name"),
            "employee_name": emp.name,
            "date_str": date_str,
            "time_str": time_str,
            "purpose": state["sched_purpose"],
        },
    )


async def _repeat_pending_question(state: Dict[str, Any]) -> str:
    """Re-ask whichever check-in slot is still missing, without re-extracting entities."""
    if not state["meeting_with"]:
        return await _llm_reply(
            "Acknowledge briefly and ask again who the visitor is here to see.",
            {"visitor_name": state.get("visitor_name")},
        )
    if not state["purpose"]:
        emp = _lookup_employee(state["meeting_with"])
        name_display = emp.name if emp else state["meeting_with"]
        return await _llm_reply(
            f"Acknowledge briefly and ask again what brings the visitor in to see {name_display}.",
            {"visitor_name": state.get("visitor_name"), "employee_name": name_display},
        )
    # All slots actually filled — shouldn't normally reach here
    return await _llm_reply(
        "Acknowledge the visitor warmly.",
        {"visitor_name": state.get("visitor_name")},
    )


# ─────────────────────────────────────────────
# CORE ROUTING
# ─────────────────────────────────────────────

async def route_query(client_id: str, user_query: str) -> str:
    llm = GroqProcessor.get_instance()
    state = get_session_state(client_id)

    # If the visitor says something with no informational content (e.g. "thank you",
    # "okay", "sure") while we are mid-flow, don't advance the state — just
    # acknowledge and re-ask the same question naturally.
    if _is_filler(user_query):
        # Only silently swallow the filler if check-in is already done or
        # if we have nothing at all collected yet (fresh session — let it fall
        # through so the LLM greets them).
        if state["checkin_done"] or state["scheduling_active"]:
            # Post check-in: filler → general LLM handles it (e.g. "You're welcome")
            pass
        elif state["visitor_name"] or state["meeting_with"]:
            # Mid check-in filler: acknowledge and repeat the pending question
            # by falling through without extracting — the check-in block below
            # will re-ask the next missing slot as normal.
            logger.info("Filler utterance detected mid-flow — repeating pending question.")
            # Skip extraction; go straight to check-in flow with unchanged state
            return await _repeat_pending_question(state)

    extracted = await llm.extract_intent_and_entities(user_query)
    entities = extracted.get("entities", {})
    _merge_entities(state, entities, user_query)
    intent = extracted.get("intent", "general_conversation")
    query_lower = user_query.lower()

    logger.debug("State after merge: %s", state)

    # ── SCHEDULING ───────────────────────────────────────────────────────
    wants_scheduling = (
        intent == "schedule_meeting"
        or state.get("scheduling_active")
        or (
            re.search(r"\b(schedule|book|arrange|set up)\b", query_lower)
            and re.search(r"\b(meeting|appointment|call)\b", query_lower)
        )
    )
    check_in_incomplete = not state["checkin_done"] and (
        not state["visitor_name"] or not state["meeting_with"] or not state["purpose"]
    )

    if wants_scheduling and not check_in_incomplete:
        state["scheduling_active"] = True
        reply = await _handle_scheduling(client_id, user_query, state, entities)
        if reply:
            return reply

    # ── EMPLOYEE LOOKUP ──────────────────────────────────────────────────
    is_lookup = (
        intent in ("employee_lookup", "role_lookup")
        or any(k in query_lower for k in ["who is", "where is", "which cabin", "which floor"])
    )
    if is_lookup:
        target = state.get("employee_name") or state.get("meeting_with")
        emp = _lookup_employee(target)
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
            return await llm.generate_grounded_response(context=context, question=user_query)
        return await _llm_reply(
            "Tell the visitor that the person they're looking for is not in the directory.",
            {"visitor_name": state.get("visitor_name")},
        )

    # ── CHECK-IN FLOW ────────────────────────────────────────────────────
    if not state["checkin_done"]:
        if not state["visitor_name"]:
            return await _llm_reply(
                "Greet the visitor warmly and ask for their name.",
                {"company_name": COMPANY_NAME},
            )

        if not state["meeting_with"]:
            return await _llm_reply(
                "Acknowledge the visitor by name and ask who they are here to see today.",
                {"visitor_name": state["visitor_name"]},
            )

        if not state["purpose"]:
            emp = _lookup_employee(state["meeting_with"])
            name_display = emp.name if emp else state["meeting_with"]
            return await _llm_reply(
                f"Ask the visitor what brings them in to see {name_display} today.",
                {
                    "visitor_name": state["visitor_name"],
                    "employee_name": name_display,
                },
            )

        # All slots filled — commit
        emp = _lookup_employee(state["meeting_with"])
        db = SessionLocal()
        try:
            v = Visitor(
                name=state["visitor_name"],
                meeting_with=state["meeting_with"],
                purpose=state["purpose"],
                checkin_time=datetime.utcnow(),
            )
            db.add(v)
            db.commit()
            state["checkin_done"] = True

            if emp:
                return await _llm_reply(
                    f"Tell the visitor they are checked in and direct them to floor {emp.floor}, cabin {emp.cabin_number} to find {emp.name}.",
                    {
                        "visitor_name": state["visitor_name"],
                        "employee_name": emp.name,
                        "floor": emp.floor,
                        "cabin": emp.cabin_number,
                    },
                )
            return await _llm_reply(
                "Tell the visitor they are checked in and ask them to have a seat.",
                {"visitor_name": state["visitor_name"]},
            )
        except Exception as e:
            logger.error("Check-in DB error: %s", e)
            db.rollback()
            return "Your visit has been noted — please have a seat."   # structural fallback only
        finally:
            db.close()

    # ── POST CHECK-IN GENERAL FALLBACK ───────────────────────────────────
    emp = _lookup_employee(state.get("meeting_with") or state.get("employee_name"))
    company_info = _build_company_info(state, emp)
    return await llm.get_response(user_query, company_info=company_info)