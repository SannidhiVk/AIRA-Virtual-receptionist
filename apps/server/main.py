import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ Correct imports (relative to apps/server)
from core.config import logger
from core.lifespan import lifespan
from client_context import set_last_employee, get_last_employee_name, get_context

app = FastAPI(
    title="AlmostHuman Voice Assistant",
    description="CPU-optimized voice assistant with real-time speech recognition, conversational brain, and text-to-speech.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],  # <-- Change this line
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PRONOUNS = {"him", "her", "them", "he", "she", "that person", "someone", "this guy"}


async def process_text_for_client(client_id: str, text: str) -> str:
    """
    Dispatcher for incoming text from WebSocket / HTTP routes.
    """
    if text is None:
        return ""

    # 1) Meeting scheduler first
    try:
        from meeting_scheduler import handle_meeting_request

        handled, reply_text = await handle_meeting_request(client_id, text)
        if handled:
            return reply_text
    except Exception as e:
        logger.error("Meeting dispatcher failed: %s", e)

    # 2) Conversational / grounded behaviour
    # ---> UPDATED: Added extra database imports needed for check-ins <---
    from receptionist.database import (
        get_employee_by_name_or_role,
        get_similar_employee,
        add_visitor,
        log_reception_entry,
    )
    from models.groq_processor import GroqProcessor

    llm = GroqProcessor.get_instance()
    text_lower = text.lower()

    try:
        extracted = await llm.extract_intent_and_entities(text)
        entities = extracted.get("entities") or {}
        intent = extracted.get("intent")
    except Exception:
        entities = {}
        intent = "general_conversation"

    # =========================================================
    # NEW FIX: CHECK-IN & DATABASE LOGGING LOGIC
    # =========================================================
    is_check_in = intent == "check_in" or any(
        phrase in text_lower
        for phrase in ["check in", "checking in", "here to see", "my name is"]
    )

    if is_check_in:
        visitor_name = entities.get("visitor_name") or entities.get("name")
        employee_to_meet = entities.get("employee_name") or entities.get("role")

        # If the LLM couldn't extract a name, politely ask for it
        if not visitor_name:
            return "Welcome to the office! Could you please tell me your name so I can check you in?"

        # Step A: Check if this is actually an Employee arriving (e.g., "I'm Rohit")
        emp_match = get_similar_employee(visitor_name)

        # If it's an employee (and they aren't here to meet someone else)
        if emp_match and not employee_to_meet:
            # ONLY add to ReceptionLog, skip the Visitor table
            log_reception_entry(
                person_name=emp_match.name,
                person_type="EMPLOYEE",
                linked_employee_id=emp_match.id,
                notes="Employee arrived",
            )
            return (
                f"Welcome back, {emp_match.name}. I've logged your arrival for the day."
            )

        # Step B: Otherwise, treat them as a Visitor
        badge_id, visitor_id = add_visitor(
            name=visitor_name,
            meeting_with=employee_to_meet or "Unknown",
            purpose="Check-in",
        )

        if visitor_id != -1:
            # Add to the unified ReceptionLog table
            log_reception_entry(
                person_name=visitor_name,
                person_type="VISITOR",
                linked_visitor_id=visitor_id,
                notes=(
                    f"Meeting with: {employee_to_meet}"
                    if employee_to_meet
                    else "General visit"
                ),
            )

            if employee_to_meet:
                return f"Welcome, {visitor_name}. I have successfully checked you in to meet {employee_to_meet}. Please take a seat."
            else:
                return f"Welcome, {visitor_name}. You are now checked in. Please have a seat."
        else:
            return "I had a little trouble saving your details to the system, but please take a seat!"
    # =========================================================

    # 3) Employee Lookups
    is_employee_lookup = intent in ["employee_lookup", "role_lookup"] or any(
        k in text_lower
        for k in ["who is", "where is", "cabin", "which floor", "extension"]
    )

    if is_employee_lookup:
        raw_term = (
            entities.get("employee_name")
            or entities.get("role")
            or entities.get("name")
            or ""
        )
        search_term = raw_term.strip().lower()

        # Pronoun resolution: "what floor is he on?" → use last looked-up employee
        if not search_term or search_term in PRONOUNS:
            fallback_name = get_last_employee_name(client_id)
            if fallback_name:
                logger.info(
                    "Pronoun lookup resolved: %r -> %r",
                    raw_term or "(empty)",
                    fallback_name,
                )
                search_term = fallback_name
            else:
                search_term = None

        if search_term:
            emp = get_employee_by_name_or_role(str(search_term))
            if emp:
                set_last_employee(
                    client_id,
                    name=emp.name,
                    email=getattr(emp, "email", None),
                    cabin=getattr(emp, "cabin_number", None),
                    role=getattr(emp, "role", None),
                    department=getattr(emp, "department", None),
                )
                return await llm.generate_grounded_response(
                    context={
                        "intent": "lookup",
                        "employee": {
                            "name": emp.name,
                            "role": emp.role,
                            "cabin_number": emp.cabin_number,
                            "department": emp.department,
                        },
                    },
                    question=text,
                )
            return f"Sorry, I couldn't find '{search_term}' in our staff directory. Can you repeat it?"

    # 4) General chat fallback
    last_emp = get_last_employee_name(client_id)
    groq_prompt = text
    if last_emp:
        groq_prompt = f"[Context: the visitor was just asking about {last_emp}] {text}"

    return await llm.get_response(groq_prompt)


@app.get("/")
async def health():
    return {"status": "running"}


# Include routes after dispatcher is defined to avoid circular imports.
from routes.api_routes import router as api_router
from routes.websocket_routes import router as websocket_router

app.include_router(api_router)
app.include_router(websocket_router)


def main():
    logger.info("Starting AlmostHuman Voice Assistant server...")

    config = uvicorn.Config(
        app="main:app",  # important change
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
