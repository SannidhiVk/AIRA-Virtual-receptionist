import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import or_
from sqlalchemy.orm import Session

from receptionist.database import SessionLocal
from receptionist.models import Employee, Visitor, Meeting
from models.groq_processor import GroqProcessor as OllamaProcessor

logger = logging.getLogger(__name__)

# State to track the interaction across turns
session_state: Dict[str, Any] = {
    "visitor_id": None,
    "visitor_name": None,
    "status": None,  # <--- ADDED THIS
    "employee_name": None,
    "time": None,
}


def _clear_state():
    logger.info("Cleaning session state...")
    for key in session_state:
        session_state[key] = None


def _parse_meeting_time(time_str: str) -> Optional[datetime]:
    """Robust parsing for messy STT output."""
    if not time_str:
        return None
    s = str(time_str).lower().strip()
    s = (
        s.replace("p.m.", "pm")
        .replace("a.m.", "am")
        .replace("to", "2")
        .replace(".", ":")
    )
    s = re.sub(r"[^0-9ap: ]", "", s)

    try:
        m = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", s)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            meridiem = m.group(3)
            if meridiem == "pm" and hour < 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
            if not meridiem and 1 <= hour <= 7:
                hour += 12
            return datetime.now().replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
    except Exception as e:
        logger.error(f"Time parse error: {e}")
    return None


def _merge_entities(entities: Dict[str, Any], raw_query: str) -> None:
    raw_lower = raw_query.lower()
    # Words that should never overwrite our stored names
    PRONOUNS = ["him", "her", "them", "he", "she", "that person", "someone", "this guy"]

    # 1. VISITOR NAME (Same logic as before)
    ent_visitor = entities.get("visitor_name") or entities.get("name")
    if ent_visitor and str(ent_visitor).lower() not in PRONOUNS:
        session_state["visitor_name"] = str(ent_visitor).capitalize()

    # 2. EMPLOYEE NAME / ROLE
    ent_employee = entities.get("employee_name") or entities.get("role")

    if ent_employee:
        ent_emp_str = str(ent_employee).lower()
        # ONLY update if the new name is NOT a pronoun
        if ent_emp_str not in PRONOUNS and len(ent_emp_str) > 2:
            session_state["employee_name"] = ent_employee

    # 3. TIME (Same logic as before)
    new_time = entities.get("time")
    if new_time and str(new_time).lower() not in ["today", "now", "soon"]:
        session_state["time"] = new_time


def log_initial_visitor(name: str, status: str = "Arrived") -> Optional[int]:
    session = SessionLocal()
    try:
        new_v = Visitor(name=name, status=status, checkin_time=datetime.now())
        session.add(new_v)
        session.commit()
        session.refresh(new_v)
        logger.info(f"Successfully logged visitor {name} to DB.")
        return new_v.id
    except Exception as e:
        logger.error(f"Visitor log error: {e}")
        return None
    finally:
        session.close()


def schedule_meeting_record() -> Optional[Dict]:
    v_name = session_state["visitor_name"]
    e_name = session_state["employee_name"]
    t_raw = session_state["time"]

    dt = _parse_meeting_time(t_raw)
    if not dt:
        return {"error": "invalid_time"}

    session = SessionLocal()
    try:
        emp = (
            session.query(Employee)
            .filter(
                or_(
                    Employee.name.ilike(f"%{e_name}%"),
                    Employee.role.ilike(f"%{e_name}%"),
                )
            )
            .first()
        )

        if not emp:
            return {"error": "employee_not_found"}

        new_meeting = Meeting(
            visitor_name=v_name, employee_name=emp.name, scheduled_time=dt
        )
        session.add(new_meeting)
        session.commit()

        logger.info(f"Meeting RECORDED: {v_name} with {emp.name} at {dt}")

        # --- Google Calendar Hook ---
        if getattr(emp, "email", None):
            try:
                from services.calendar_service import send_calendar_invite

                send_calendar_invite(v_name, emp.email, dt)
                logger.info("Successfully pushed invite wrapper request.")
            except Exception as e:
                logger.error(f"Calendar invite wrapper failed: {e}")
        # ----------------------------

        return {
            "employee": emp.name,
            "time": dt.strftime("%I:%M %p"),
            "cabin": emp.cabin_number,
        }
    except Exception as e:
        logger.error(f"Database error: {e}")
        return {"error": "db_error"}
    finally:
        session.close()


async def route_query(user_query: str) -> str:
    ollama = OllamaProcessor.get_instance()
    extracted = await ollama.extract_intent_and_entities(user_query)
    entities = extracted.get("entities") or {}
    intent = extracted.get("intent")

    # Update state (Pronouns will be ignored here, keeping the old name)
    _merge_entities(entities, user_query)

    # 2. PRIORITY: EMPLOYEE LOOKUP
    if intent in ["employee_lookup", "role_lookup"] or any(
        k in user_query.lower() for k in ["who is", "where is", "cabin"]
    ):
        session = SessionLocal()
        # Try to find by the name we just got OR the name we already had
        search_term = session_state["employee_name"]

        if search_term:
            emp = (
                session.query(Employee)
                .filter(
                    or_(
                        Employee.name.ilike(f"%{search_term}%"),
                        Employee.role.ilike(f"%{search_term}%"),
                    )
                )
                .first()
            )

            if emp:
                # CRITICAL: Update session_state with the REAL name found
                # So "him" in the next turn refers to "Priya" (the name), not "Manager" (the role)
                session_state["employee_name"] = emp.name
                session.close()
                return await ollama.generate_grounded_response(
                    context={
                        "intent": "lookup",
                        "employee": {
                            "name": emp.name,
                            "role": emp.role,
                            "cabin_number": emp.cabin_number,
                            "department": emp.department,
                        },
                    },
                    question=user_query,
                )
        session.close()

    # 3. CHECK-IN LOGIC (Keep your existing logic here...)
    # ...

    # 4. MEETING SCHEDULING LOGIC
    if intent == "schedule_meeting" or (
        session_state["employee_name"] and session_state["time"]
    ):
        # If the user said "him" and we have NO name in state, we must ask.
        if not session_state["employee_name"]:
            return "I'd be happy to schedule that. Who would you like to meet with?"

        if not session_state["visitor_name"]:
            return "I'd be happy to schedule that. May I have your name first?"

        if not session_state["time"]:
            return f"Understood. What time is your meeting with {session_state['employee_name']}?"

        res = schedule_meeting_record()
        if "error" in res:
            if res["error"] == "employee_not_found":
                # If the previous lookup was a role that doesn't exist, clear it
                err_name = session_state["employee_name"]
                session_state["employee_name"] = None
                return (
                    f"I'm sorry, I couldn't find '{err_name}' in our staff directory."
                )
            return "I had trouble saving the meeting. Could you repeat the time?"

        v_save = session_state["visitor_name"]
        _clear_state()  # Clear after success
        return f"Perfect {v_save}. I've scheduled your meeting with {res['employee']} for {res['time']}. You can head to cabin {res['cabin']}."

    # 5. FALLBACK
    return await ollama.get_response(
        f"Context: {session_state}. User says: {user_query}"
    )
