import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import AsyncGroq

logger = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AI_NAME = "Sannika"
COMPANY_NAME = "Sharp Software"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = f"""You are {AI_NAME}, an intelligent, highly proactive, and warm virtual receptionist at {COMPANY_NAME}.
You exist solely to assist visitors and employees at this office.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY & TONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Your name is {AI_NAME}. Introduce yourself naturally, but don't over-repeat it.
• Be highly conversational, smart, human-like, and polite. Act exactly like a premium concierge or an advanced AI assistant (like Alexa).
• If a visitor asks a question, answer it directly and intelligently. Do not force them into rigid scripts.
• You are allowed to use natural conversational connectors ("Sure thing," "Got it," "I completely understand," "Welcome!").
• Keep replies concise but fluid (usually 1-3 sentences).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU CAN DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Visitor check-in: collect name → who they're seeing → purpose. 
2. Deliveries: ask who it is for, tell them to leave it at the desk, and say you will notify the recipient.
3. Employee/department lookup: floor, cabin, extension — ONLY from EMPLOYEE INFO below.
4. Meeting scheduling: help set up future appointments.
5. Basic company info: answer only from COMPANY CONTEXT below.
6. Facility requests: Tell them you will notify the administration or the person in charge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE PROHIBITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Never invent an employee name, floor, cabin, extension, or email.
✗ Never roleplay as any employee or manager.
✗ Never say you are an AI, a bot, or a computer.
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """You are an information extraction engine for a corporate receptionist system.
Given a visitor's spoken input, extract structured data and return ONLY a valid JSON object.
No markdown. No explanation. No preamble. Just the JSON.

Output format:
{
  "intent": <string>,
  "entities": {
    "visitor_name": <string|null>,
    "employee_name": <string|null>,
    "role": <string|null>,
    "date": <string|null>,
    "time": <string|null>,
    "purpose": <string|null>
  }
}

FIELD RULES:
- visitor_name: The speaking person's own name. If they say "I have a delivery", extract "Delivery".
- employee_name: The specific named person the visitor wants to see, meet, or drop a delivery for.
- role: A job title or department (e.g., "HR Manager", "Sales", "engineering department", "HR").
- date: A date or relative day ("today", "tomorrow"). Null if not mentioned.
- time: A specific clock time ("5:00 PM", "14:00"). Null if vague or not mentioned.
- purpose: The reason for the visit ("job interview", "sales demo", "delivery").
- intent: Exactly one of:
    "check_in"            — arriving to see someone NOW or drop off a package NOW.
    "schedule_meeting"    — wants to BOOK, SCHEDULE, or SET UP a future appointment/meeting.
    "employee_lookup"     — asking where someone/a department is located.
    "facility_request"    — complaining/asking about AC, temperature, lights, wifi, or cleaning.
    "general_conversation"— asking about office hours, directions, or casual conversation.
    "confirm"             — yes / go ahead / proceed / please do.
    "cancel"              — no / cancel / never mind / stop.

IMPORTANT:
- If a visitor says "I want to schedule an appointment" or "book a meeting for tomorrow", the intent MUST be "schedule_meeting".
"""

# ─────────────────────────────────────────────────────────────────────────────
# ENV / API KEY
# ─────────────────────────────────────────────────────────────────────────────


def _load_dotenv_from_any_location() -> None:
    try:
        from dotenv import load_dotenv as _load

        here = Path(__file__).resolve()
        candidates = [
            here.parent / ".env",
            here.parent.parent / ".env",
            here.parent.parent.parent / ".env",
        ]
        for path in candidates:
            if path.exists():
                _load(dotenv_path=str(path), override=False)
                return
    except ImportError:
        pass


def _read_api_key() -> str:
    _load_dotenv_from_any_location()
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        base_dir = Path(__file__).resolve().parent.parent
        key_path = base_dir / "GROQ_API_KEY.txt"
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    logger.error("GROQ_API_KEY not found.")
    return ""


def _build_system_message(company_info: Optional[dict] = None) -> str:
    system = BASE_SYSTEM_PROMPT
    if company_info:
        system += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCOMPANY CONTEXT\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if company_info.get("company_name"):
            system += f"\nCompany: {company_info['company_name']}"
        if company_info.get("company_location"):
            system += f"\nLocation: {company_info['company_location']}"
        if company_info.get("office_hours"):
            system += f"\nOffice Hours: {company_info['office_hours']}"
        if company_info.get("departments"):
            system += f"\nDepartments: {company_info['departments']}"
        if company_info.get("dynamic_employee"):
            system += f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nEMPLOYEE INFO\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{company_info['dynamic_employee']}"
        if company_info.get("visitor_name"):
            system += f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCURRENT VISITOR: {company_info['visitor_name']}\n⚠ Address this visitor ONLY by this name.\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    return system


def _clean_reply(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"^(AI|Assistant|Sannika|AlmostHuman)\s*:\s*", "", text, flags=re.IGNORECASE
    )
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# GROQ PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────


class GroqProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        api_key = _read_api_key()
        raw_base = os.getenv("GROQ_BASE_URL", "").strip().rstrip("/")
        if raw_base and "/openai" not in raw_base:
            self.client = AsyncGroq(api_key=api_key, base_url=raw_base)
        else:
            self.client = AsyncGroq(api_key=api_key)
        self.model_name = MODEL
        # Dictionary to store history strictly per client!
        self.client_history: Dict[str, List[Dict[str, str]]] = {}
        logger.info("GroqProcessor initialized with model '%s'", self.model_name)

    def reset_history(self, client_id: str):
        self.client_history[client_id] = []
        logger.info(f"GroqProcessor conversation history reset for {client_id}.")

    async def get_raw_response(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.6,
            )
            return _clean_reply(response.choices[0].message.content)
        except Exception as e:
            logger.error("get_raw_response error: %s", e)
            return "I'm sorry, I didn't quite catch that. Could you say it again?"

    async def get_response(
        self, client_id: str, prompt: str, company_info: Optional[dict] = None
    ) -> str:
        if not prompt:
            return ""

        if client_id not in self.client_history:
            self.client_history[client_id] = []

        if re.search(r"\b(bye|goodbye|thank you|thanks)\b", prompt.strip().lower()):
            self.reset_history(client_id)

        if len(self.client_history[client_id]) > 12:
            self.client_history[client_id] = self.client_history[client_id][-12:]

        self.client_history[client_id].append({"role": "user", "content": prompt})
        messages = [
            {"role": "system", "content": _build_system_message(company_info)}
        ] + self.client_history[client_id]

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.5,
            )
            content = _clean_reply(response.choices[0].message.content)
            self.client_history[client_id].append(
                {"role": "assistant", "content": content}
            )
            return content
        except Exception as e:
            logger.error("Groq chat error: %s", e)
            return "I'm sorry, my connection blinked. Could you repeat that?"

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        raw = None
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": user_query.strip()},
                ],
                max_tokens=250,
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]
            parsed = json.loads(raw)
            entities = parsed.get("entities", {})
            if not isinstance(entities, dict):
                entities = {}
            for k, v in list(entities.items()):
                if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
                    entities[k] = None
            return {
                "intent": parsed.get("intent", "general_conversation"),
                "entities": entities,
            }
        except Exception as e:
            logger.error("Extraction failed: %s", e)
            return {"intent": "general_conversation", "entities": {}}

    async def generate_grounded_response(self, context: dict, question: str) -> str:
        if "employee" in context:
            e = context["employee"]
            info = f"Name: {e.get('name')}, Role: {e.get('role')}, Floor: {e.get('floor')}, Cabin: {e.get('cabin_number')}, Department: {e.get('department')}"
        else:
            info = "No records found."

        visitor_name = context.get("visitor_name", "")
        address_clause = (
            f"Address the visitor as '{visitor_name}'." if visitor_name else ""
        )

        prompt = f"""You are {AI_NAME}, a corporate office receptionist.
The visitor asked: "{question}"
Verified directory records: {info}
{address_clause}

Respond naturally and conversationally in 1-3 sentences. State the floor and cabin if found. 
If no records are found, apologize politely and suggest they take a seat while you contact administration to help them. 
Tone: warm, smart, human, professional."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.4,
            )
            return _clean_reply(response.choices[0].message.content)
        except Exception:
            return "Please head over to the main lobby, and someone will assist you shortly."
