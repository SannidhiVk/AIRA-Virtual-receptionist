import json
import logging
from typing import List, Dict, Any

import ollama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are AlmostHuman, the receptionist at Sharp Software Technology.
Greet visitors politely.
Identify if they are employee, intern, guest, or candidate.
Confirm details, then guide them to HR, meeting room, or team.
Never mention being an AI.
Keep replies short and professional.

Response:
Natural reply.

Data:
{"t":"","n":"","p":"","m":"","a":"","s":""}
"""


class OllamaProcessor:
    """Handles text generation using an Ollama-served LLM (llama3.2)."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = ollama.AsyncClient()
        self.model_name = "llama3:8b-instruct-q4_0"
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        logger.info(f"OllamaProcessor initialized with model '{self.model_name}'")

    def reset_history(self):
        """Clear the conversation history (preserving system prompt)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("OllamaProcessor conversation history reset")

    async def get_response(self, prompt: str) -> str:
        if not prompt:
            return ""

        # Keep only last 6 messages + system prompt
        self.history = [self.history[0]] + self.history[-6:]

        self.history.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=self.history,
                stream=False,
            )

            if hasattr(response, "message"):
                content = response.message.content
            else:
                content = response.get("message", {}).get("content", "")

            content = (content or "").strip()

            if not content:
                logger.warning("Ollama returned empty response.")
                content = "I'm sorry, I couldn't process that. How can I help you?"

            self.history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "I'm having trouble thinking right now."

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        """
        Extract intent and entities from user query via LLM.
        Returns a dict with 'intent' and 'entities' keys.
        """
        EXTRACT_SYSTEM = (
            "You are a data extraction engine. Your ONLY output is raw JSON. "
            "Do NOT explain. Do NOT use markdown code blocks (```json). Do NOT converse. "
            "Valid Intents: 'employee_lookup', 'department_lookup', 'cabin_lookup', 'general_conversation'. "
            'Example Output: {"intent": "cabin_lookup", "entities": {"name": "Priya", "department": "HR"}}'
        )
        fallback = {"intent": "general_conversation", "entities": {}}

        if not user_query or not user_query.strip():
            return fallback

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": user_query.strip()},
                ],
                stream=False,
                options={"temperature": 0},
            )
            if hasattr(response, "message"):
                raw = response.message.content
            else:
                raw = response.get("message", {}).get("content", "")

            raw = (raw or "").strip()
            if not raw:
                return fallback

            if raw.startswith("```"):
                for prefix in ("```json", "```"):
                    if raw.startswith(prefix):
                        raw = raw[len(prefix) :].strip()
                        break
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

            parsed = json.loads(raw)
            intent = parsed.get("intent", "general_conversation")
            entities = (
                parsed.get("entities")
                if isinstance(parsed.get("entities"), dict)
                else {}
            )

            valid_intents = {
                "employee_lookup",
                "department_lookup",
                "cabin_lookup",
                "general_conversation",
            }
            if intent not in valid_intents:
                intent = "general_conversation"

            return {"intent": intent, "entities": entities}

        except (json.JSONDecodeError, TypeError, Exception) as e:
            logger.warning(f"Intent extraction failed, using fallback: {e}")
            return fallback

    async def generate_grounded_response(self, context: dict, question: str) -> str:
        """
        Converts database results into a natural sentence.
        """
        # 1. Format the data into a clear string based on the result type
        if "employee" in context:
            # Handles: "Where is Priya?" or "Where is cabin 202?"
            e = context["employee"]
            info = f"Name: {e['name']}, Cabin: {e['cabin_number']}, Department: {e['department']}"

        elif "employees" in context:
            # Handles: "Who is in HR?" or "Who is in Finance?"
            dept = context.get("department", "the requested")
            # Create a list of names and cabins
            people = ", ".join(
                [
                    f"{emp['name']} (Cabin {emp['cabin_number']})"
                    for emp in context["employees"]
                ]
            )
            info = f"Department: {dept}, Staff: {people}"

        else:
            info = "No matching records found."

        # 2. Create a direct instruction for this specific answer
        # This does NOT change your global system prompt.
        prompt = f""" u are a professional and polite office receptionist assisting visitors inside a company office.

    A visitor asked:
    "{question}"

    The system searched internal records and returned the following information:
    {info}

    The information is provided in JSON format and may contain:
    - employee details
    - department information
    - meeting information


    Your task:
    - Respond like a real office receptionist.
    - Use the information in {info} to guide the visitor.
    - If the information contains employee details, mention the employee's name, role, department, and cabin number.
    - If multiple employees are present, list all of them clearly.
    - If the information contains meeting details, guide the visitor to the correct meeting room or location.
    - If {info} is empty, politely say that the information could not be found and ask the visitor to confirm the details.

    IMPORTANT RULES:
    - The information in {info} refers to the person, department, or meeting the visitor is asking about.
    - Do NOT assume the visitor's identity.
    - Do NOT invent names, cabin numbers, roles, or meeting details.
    - Only use the information provided in {info}.
    - Keep the response short, natural, and conversational (1–3 sentences).

    Tone:
    Friendly, helpful, and professional — like a real office receptionist.

    Now generate the receptionist response."""

        # 3. Get the response from Ollama
        response = await self.get_response(prompt)
        return response.strip()
