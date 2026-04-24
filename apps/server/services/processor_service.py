from core.config import logger

WAKE_WORD_TRIGGER_TEXT = "WAKE_WORD_TRIGGERED"
WAKE_WORD_GREETING = (
    "Welcome to Sharp Software Development India Private Limited. "
    "I am Jarvis, how can I assist you today?"
)


from services.query_router import route_query, clear_session_state


async def process_text_for_client(client_id: str, text: str) -> str:
    if not text:
        return ""

    if text == "WAKE_WORD_TRIGGERED":
        clear_session_state(client_id)
        return "Good Afternoon! Welcome to Sharp Software Development India Private Limited. I am Jarvis, how can I assist you today?"

    try:
        return await route_query(client_id, text)
    except Exception as e:
        return "I am having trouble processing that. Could you repeat?"
