from core.config import logger

WAKE_WORD_TRIGGER_TEXT = "WAKE_WORD_TRIGGERED"
WAKE_WORD_GREETING = (
    "Welcome to Sharp Software Development India Private Limited. "
    "I am Jarvis, how can I assist you today?"
)


from services.query_router import route_query, clear_session_state


from datetime import datetime


async def process_text_for_client(client_id: str, text: str) -> str:
    if not text:
        return ""

    if text == "WAKE_WORD_TRIGGERED":
        from services.query_router import clear_session_state, get_session_state

        clear_session_state(client_id)

        # Precise Time Logic
        hour = datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good Morning"
        elif 12 <= hour < 17:
            greeting = "Good Afternoon"
        else:
            greeting = "Good Evening"

        state = get_session_state(client_id)
        state["greeted"] = True  # Mark as greeted so router doesn't repeat it

        return f"{greeting}! Welcome to Sharp Software Development India Private Limited. I am Jarvis, how can I assist you today?"

    try:
        from services.query_router import route_query

        return await route_query(client_id, text)
    except Exception as exc:
        logger.error(f"Routing Error: {exc}")
        return "I'm sorry, I'm having trouble connecting to my systems."
