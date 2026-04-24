from core.config import logger

WAKE_WORD_TRIGGER_TEXT = "WAKE_WORD_TRIGGERED"
WAKE_WORD_GREETING = (
    "Welcome to Sharp Software Development India Private Limited. "
    "I am Jarvis, how can I assist you today?"
)


# --- IN processor_service.py ---


async def process_text_for_client(client_id: str, text: str) -> str:
    if not text or not text.strip():
        return ""

    # Always trigger a fresh start on the Wake Word
    if text == WAKE_WORD_TRIGGER_TEXT:
        from services.query_router import clear_session_state

        clear_session_state(client_id)

        # Get current hour to determine greeting
        from datetime import datetime

        hour = datetime.now().hour
        greeting = "Good Morning"
        if 12 <= hour < 17:
            greeting = "Good Afternoon"
        elif hour >= 17:
            greeting = "Good Evening"

        return f"{greeting}! Welcome to Sharp Software Development India Private Limited. I am Jarvis, how can I assist you today?"

    try:
        from services.query_router import route_query

        return await route_query(client_id, text)
    except Exception as exc:
        logger.error("route_query failed: %s", exc)
        return "I'm sorry, I'm having trouble with my internal systems."
