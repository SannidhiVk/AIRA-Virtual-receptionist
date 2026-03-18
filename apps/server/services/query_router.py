import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from models.ollama_processor import OllamaProcessor
from receptionist.database import SessionLocal
from receptionist.models import Employee, Visitor, Meeting

logger = logging.getLogger(__name__)

SCHEDULE_INTENT = "schedule_meeting"

# State for multi-turn scheduling
meeting_state: Dict[str, Optional[str]] = {
    "visitor_name": None,
    "employee_name": None,
    "time": None,
}


def _serialize_employee(emp: Employee) -> Dict[str, Any]:
    """Converts DB object to dictionary."""
    return {
        "id": emp.id,
        "name": emp.name,
        "role": emp.role,
        "department": emp.department,
        "cabin_number": emp.cabin_number,
    }


def _parse_meeting_time(time_str: str) -> Optional[datetime]:
    """Parses time strings like '2 PM'."""
    if not time_str or not str(time_str).strip():
        return None
    s = str(time_str).strip()
    try:
        m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
        if m:
            h, minute = int(m.group(1)), int(m.group(2))
            return datetime.now().replace(
                hour=h, minute=minute, second=0, microsecond=0
            )

        m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(AM|PM)$", s, re.IGNORECASE)
        if m:
            h, minute = int(m.group(1)), int(m.group(2) or 0)
            if m.group(3).upper() == "PM" and h != 12:
                h += 12
            elif m.group(3).upper() == "AM" and h == 12:
                h = 0
            return datetime.now().replace(
                hour=h, minute=minute, second=0, microsecond=0
            )
    except Exception:
        pass
    return None


def _log_visitor_history(name: str, status: str):
    """Saves the person to the general visitors history table."""
    session = SessionLocal()
    try:
        new_v = Visitor(name=name, status=status, checkin_time=datetime.now())
        session.add(new_v)
        session.commit()
        logger.info(f"Logged to Visitor History: {name}")
    except Exception as e:
        logger.error(f"Visitor log error: {e}")
        session.rollback()
    finally:
        session.close()


def handle_db_query(
    llm_entities: Dict[str, Any], raw_query: str = None
) -> Optional[Dict[str, Any]]:
    """Searches the Employee table with a manual keyword fallback."""
    session = SessionLocal()
    try:
        name_val = llm_entities.get("name") or llm_entities.get("employee_name")
        role_val = llm_entities.get("role")
        dept_val = llm_entities.get("department")

        print(
            f"[DEBUG] Searching DB with: Name='{name_val}', Role='{role_val}', Dept='{dept_val}'"
        )

        # 1. Standard Search (AI Entities)
        if name_val:
            emp = (
                session.query(Employee)
                .filter(Employee.name.ilike(f"%{name_val}%"))
                .first()
            )
            if emp:
                return {
                    "intent": "employee_lookup",
                    "employee": _serialize_employee(emp),
                }

        if role_val:
            employees = (
                session.query(Employee)
                .filter(Employee.role.ilike(f"%{role_val}%"))
                .all()
            )
            if employees:
                return {
                    "intent": "role_lookup",
                    "role": role_val,
                    "employees": [_serialize_employee(e) for e in employees],
                }

        # 2. SMART FALLBACK (Keyword search in raw query)
        if raw_query:
            print(
                f"[DEBUG] AI failed to extract entities. Scanning raw query: '{raw_query}'"
            )
            clean_query = raw_query.lower()
            all_employees = session.query(Employee).all()
            for emp in all_employees:
                # Check if role is mentioned in the sentence
                if emp.role and emp.role.lower() in clean_query:
                    print(f"[DEBUG] Found Role Match via Keyword: {emp.role}")
                    return {
                        "intent": "role_lookup",
                        "role": emp.role,
                        "employees": [_serialize_employee(emp)],
                    }
                # Check if name is mentioned
                if emp.name.lower() in clean_query:
                    print(f"[DEBUG] Found Name Match via Keyword: {emp.name}")
                    return {
                        "intent": "employee_lookup",
                        "employee": _serialize_employee(emp),
                    }

        return None
    finally:
        session.close()


def handle_schedule_meeting() -> Optional[Dict[str, Any]]:
    """Saves the appointment to the meetings table."""
    v_name = meeting_state.get("visitor_name")
    e_name = meeting_state.get("employee_name")
    t_raw = meeting_state.get("time")

    dt = _parse_meeting_time(t_raw)
    if not dt:
        return None

    session = SessionLocal()
    try:
        emp = session.query(Employee).filter(Employee.name.ilike(f"%{e_name}%")).first()
        if not emp:
            return {"error": "employee_not_found"}

        new_meeting = Meeting(
            visitor_name=v_name, employee_name=emp.name, scheduled_time=dt
        )
        session.add(new_meeting)
        session.commit()

        meeting_state.update(
            {"visitor_name": None, "employee_name": None, "time": None}
        )
        return {
            "visitor": v_name,
            "employee": emp.name,
            "time": dt.strftime("%I:%M %p"),
        }
    except Exception as e:
        session.rollback()
        return {"error": "db_error"}
    finally:
        session.close()


async def route_query(user_query: str) -> str:
    ollama = OllamaProcessor.get_instance()
    print(f"\n--- NEW QUERY: '{user_query}' ---")

    extracted_data = await ollama.extract_intent_and_entities(user_query)
    llm_entities = extracted_data.get("entities") or {}
    intent = extracted_data.get("intent", "general_conversation")

    # 1. LOG VISITOR HISTORY
    v_name = llm_entities.get("visitor_name") or llm_entities.get("name")
    if v_name:
        status = "Intern" if "intern" in user_query.lower() else "Visitor"
        _log_visitor_history(v_name, status)

    # 2. HANDLE MEETINGS
    if intent == SCHEDULE_INTENT:
        for key in ["visitor_name", "employee_name", "time"]:
            if llm_entities.get(key):
                meeting_state[key] = llm_entities[key]

        if not meeting_state["visitor_name"]:
            return "May I have your name, please?"
        if not meeting_state["employee_name"]:
            return "Who would you like to meet?"
        if not meeting_state["time"]:
            return "What time should I schedule this for?"

        res = handle_schedule_meeting()
        if res:
            if res.get("error") == "employee_not_found":
                return "I couldn't find that employee."
            return f"Perfect. I've scheduled a meeting for {res['visitor']} with {res['employee']} at {res['time']}."

    # 3. HANDLE EMPLOYEE LOOKUP
    lookup_keywords = ["who", "where", "manager", "cabin", "office", "department"]
    is_asking_lookup = intent in ["employee_lookup", "role_lookup"] or any(
        w in user_query.lower() for w in lookup_keywords
    )

    if is_asking_lookup:
        db_result = handle_db_query(llm_entities, raw_query=user_query)
        if db_result:
            return await ollama.generate_grounded_response(
                context=db_result, question=user_query
            )
        else:
            return "I'm sorry, I couldn't find any employee matching that description in our records."

    # 4. FALLBACK TO CHAT
    return await ollama.get_response(user_query)
