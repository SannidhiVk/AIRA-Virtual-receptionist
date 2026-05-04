from core.config import logger

WAKE_WORD_TRIGGER_TEXT = "WAKE_WORD_TRIGGERED"
WAKE_WORD_GREETING = (
    "Welcome to Sharp Software Development India Private Limited. "
    "I am Jarvis, how can I assist you today?"
)


async def process_text_for_client(client_id: str, text: str) -> str:
    if not text:
        return ""

    if text == "WAKE_WORD_TRIGGERED":
        from services.query_router import clear_session_state

        clear_session_state(client_id)

        # Correct Time Math
        from datetime import datetime

        hour = datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good Morning"
        elif 12 <= hour < 17:
            greeting = "Good Afternoon"
        else:
            greeting = "Good Evening"

        bot_reply = f"{greeting}! Welcome to Sharp Software Development India Private Limited. I am Jarvis, how can I assist you today?"

        # --- BULLETPROOF FIX: Find GroqProcessor dynamically from memory ---
        import sys

        GroqProcessor = None

        # Search loaded modules for groq_processor
        for mod_name, mod in list(sys.modules.items()):
            if "groq_processor" in mod_name and hasattr(mod, "GroqProcessor"):
                GroqProcessor = mod.GroqProcessor
                break

        # If found, inject the greeting into the AI's history
        if GroqProcessor:
            groq = GroqProcessor.get_instance()
            if client_id not in groq.client_history:
                groq.client_history[client_id] = []

            groq.client_history[client_id].append(
                {"role": "assistant", "content": bot_reply}
            )
        else:
            from core.config import logger

            logger.warning("Could not find GroqProcessor in memory to inject history.")
        # -------------------------------------------------------------------

        return bot_reply

    try:
        from services.query_router import route_query

        return await route_query(client_id, text)
    except Exception as exc:
        from core.config import logger

        logger.error(f"Routing error: {exc}")
        return "I'm sorry, I'm having trouble connecting to my systems."
