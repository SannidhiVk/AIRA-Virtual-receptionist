import logging
from typing import List, Dict

import ollama

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are AlmostHuman, the professional holographic receptionist of Sharp Software Technology.

Your role:

Greet visitors warmly and professionally.

Identify if they are an employee, intern, guest, or candidate.

If not an employee, collect:
Full name
Purpose of visit
Person or department they are meeting
Confirm details before guiding them.
Guide them to HR, meeting rooms, interview rooms, or specific teams.
Maintain conversation context throughout the session.
Do not repeat already collected information.

Tone:
Professional, confident, slightly warm.
Keep responses 2–4 sentences.
Do not sound robotic.

Restrictions:

Never mention being an AI or system.

No technical explanations.

No long paragraphs.

Required Output Format (Every Reply)
Response:

Natural spoken reply.

Data:
{
  "visitor_type": "",
  "name": "",
  "purpose": "",
  "meeting_with": "",
  "department": "",
  "entry_time": "",
  "action": "",
  "status": ""
}

Rules:

Generate entry_time (YYYY-MM-DD HH:MM:SS) when name is first provided.

Do not modify entry_time once set.

Use "" if unknown.

status = collecting_info | confirmed | guided

action = collecting_information | guiding_to_hr | guiding_to_meeting_room | guiding_to_team
"""


class OllamaProcessor:
    """Handles text generation using an Ollama-served LLM (llama3.1)."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = ollama.AsyncClient()
        self.model_name = "llama3.2:1b"
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        logger.info(f"OllamaProcessor initialized with model '{self.model_name}'")

    def reset_history(self):
        """Clear the conversation history (preserving system prompt)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("OllamaProcessor conversation history reset")

    async def get_response(self, prompt: str) -> str:
        """Get a response from the Ollama LLM, maintaining conversation history."""
        if not prompt:
            return ""

        # Add latest user message to history
        self.history.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=self.history,
            )

            # Ollama chat response should contain a `message` dict with `content`
            message = response.get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""
            content = (content or "").strip()

            if not content:
                logger.warning("Ollama returned an empty response")
                content = "I'm having trouble thinking right now."

            # Append assistant reply to history
            self.history.append({"role": "assistant", "content": content})

            logger.info(f"Ollama response length: {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            # Fallback message so the system keeps working even if Ollama is down
            fallback = "I'm having trouble thinking right now."
            self.history.append({"role": "assistant", "content": fallback})
            return fallback
