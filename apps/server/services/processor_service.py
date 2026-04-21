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
        try:
            from services.query_router import clear_session_state
            from models.groq_processor import GroqProcessor

            # 1. Reset the session so we start completely fresh on a wake word
            clear_session_state(client_id, retain_name=False)

            # 2. Inject the hardcoded greeting into the LLM's memory
            # This tells the LLM it ALREADY welcomed the user, preventing repeats!
            llm = GroqProcessor.get_instance()
            llm.client_history[client_id].append(
                {"role": "assistant", "content": WAKE_WORD_GREETING}
            )
        except Exception as e:
            logger.error(f"Failed to inject wake word context: {e}")

        return WAKE_WORD_GREETING

    try:
        from services.query_router import route_query

        return await route_query(client_id, text)
    except Exception as exc:
        logger.error("route_query failed: %s", exc, exc_info=True)
        return "I'm sorry, I'm having trouble connecting to my systems."
