import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ Correct imports (relative to apps/server)
from core.config import logger
from core.lifespan import lifespan

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


async def process_text_for_client(client_id: str, text: str) -> str:
    """
    Dispatcher for incoming text from WebSocket / HTTP routes.

    1) Try meeting slot-filling flow (per-client state machine).
    2) If not a meeting, fallback to employee lookup grounding or general chat.
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

    # 2) Fallback conversational / grounded behavior
    from receptionist.database import get_employee_by_name_or_role
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

    is_employee_lookup = intent in ["employee_lookup", "role_lookup"] or any(
        k in text_lower for k in ["who is", "where is", "cabin"]
    )

    if is_employee_lookup:
        search_term = (
            entities.get("employee_name")
            or entities.get("role")
            or entities.get("name")
        )
        if search_term:
            emp = get_employee_by_name_or_role(str(search_term))
            if emp:
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

    return await llm.get_response(text)


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
