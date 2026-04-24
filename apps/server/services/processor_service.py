from core.config import logger

WAKE_WORD_TRIGGER_TEXT = "WAKE_WORD_TRIGGERED"
WAKE_WORD_GREETING = (
    "Welcome to Sharp Software Development India Private Limited. "
    "I am Jarvis, how can I assist you today?"
)


async def process_text_for_client(client_id: str, text: str) -> str:
    """
    Process a user utterance and return the assistant response text.
    """
    if not text or not text.strip():
        return ""

    if text == WAKE_WORD_TRIGGER_TEXT:
        return WAKE_WORD_GREETING

    try:
        from services.query_router import route_query

        return await route_query(client_id, text)
    except Exception as exc:
        logger.error("route_query failed: %s", exc, exc_info=True)
        return "I'm sorry, I'm having trouble connecting to my systems."
