import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import AsyncGroq

logger = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Private Limited."

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
• Vary your conversational openers naturally. Never start two consecutive replies 
   the same way. Examples: "Of course!", "Absolutely!", "Happy to help!", 
   "Sure thing!", "No problem at all!", "Let me take care of that for you." 
   Avoid repeating "Got it" more than once per conversation.
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
# EXTRACTION PROMPT (PURE NLU ENGINE)
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
    "purpose": <string|null>,
    "visitor_type": <string|null>
    "email": <string|null>
  }
}

INTENT CLASSIFICATION RULES:
- "check_in" : Arriving at the office NOW to see someone, attend an interview, or drop off a package.
- "employee_arrival" : An employee who works here arriving for the day.
- "schedule_meeting" : Wants to BOOK, SCHEDULE, or SET UP a FUTURE appointment.
- "employee_lookup" : Asking where a person or department is.
- "facility_request" : Complaining or asking about the environment (AC, lights, cleaning).
- "general_conversation" : Greetings, small talk.
- "confirm" : Saying yes, go ahead.
- "cancel" : Saying no, cancel.

VISITOR TYPE CATEGORIES (MUST BE ONE OF THESE IF APPLICABLE):
- "Contractor/Vendor" : Maintenance, electrician, plumber, service staff.
- "Interviewee" : Job candidates, HR interviews.
- "Client" : Clients, demos, customer meetings.
- "Delivery" : Amazon, Flipkart, packages, couriers.
- "Food Delivery" : Swiggy, Zomato.
- "Visitor/Guest" : General personal/business meetings.

FEW-SHOT EXAMPLES:
Input: "I am here for an HR interview."
Output: {"intent": "check_in", "entities": {"visitor_name": null, "employee_name": null, "role": null, "date": null, "time": null, "purpose": "HR interview", "visitor_type": "Interviewee"}}

Input: "I have a package from Amazon for Sanjay."
Output: {"intent": "check_in", "entities": {"visitor_name": "Amazon", "employee_name": "Sanjay", "role": null, "date": null, "time": null, "purpose": "delivery", "visitor_type": "Delivery"}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# ENV / API KEY
# ─────────────────────────────────────────────────────────────────────────────


def _load_dotenv_from_any_location() -> None:
    try:
        from dotenv import load_dotenv as _load

        here = Path(__file__).resolve()
        candidates = [
            here.parent / ".env",  # apps/server/.env
            here.parent.parent / ".env",  # apps/.env
            here.parent.parent.parent / ".env",  # project root .env
        ]
        for path in candidates:
            if path.exists():
                _load(dotenv_path=str(path), override=False)
                logger.info("Loaded .env from: %s", path)
                return
        logger.warning("No .env file found in: %s", [str(c) for c in candidates])
    except ImportError:
        # python-dotenv not installed — parse manually
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
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    system = (
        BASE_SYSTEM_PROMPT
        + f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCURRENT DATE & TIME\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{current_time}"
    )

    if company_info:
        system += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCOMPANY CONTEXT\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if company_info.get("company_name"):
            system += f"\nCompany Name: {company_info['company_name']}"
        if company_info.get("company_address"):
            system += f"\nOffice Address: {company_info['company_address']}"
        if company_info.get("company_phone"):
            system += f"\nContact Phone: {company_info['company_phone']}"
        if company_info.get("company_email"):
            system += f"\nContact Email: {company_info['company_email']}"
        if company_info.get("company_website"):
            system += f"\nWebsite: {company_info['company_website']}"

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
        self.client_history: Dict[str, List[Dict[str, str]]] = {}
        logger.info("GroqProcessor initialized with model '%s'", self.model_name)

    def reset_history(self, client_id: str):
        self.client_history[client_id] = []
        logger.info(f"GroqProcessor conversation history reset for {client_id}.")

    async def get_raw_response(
        self, prompt: str, client_id: Optional[str] = None
    ) -> str:
        try:
            if client_id and client_id in self.client_history:
                # Use history for context but don't persist this structured prompt
                context_messages = self.client_history[client_id][-6:]  # last 3 turns
            else:
                context_messages = []

            messages = context_messages + [{"role": "user", "content": prompt}]

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=120,  # also increased — 100 cuts off responses
                temperature=0.6,
            )
            reply = _clean_reply(response.choices[0].message.content)

            # Save to history so next turn has context
            if client_id:
                if client_id not in self.client_history:
                    self.client_history[client_id] = []
                self.client_history[client_id].append(
                    {"role": "assistant", "content": reply}
                )

            return reply
        except Exception as e:
            logger.error("get_raw_response error: %s", e)
            return "I'm sorry, I didn't quite catch that. Could you say it again?"

    async def get_response(
        self,
        client_id: str = "default_client",
        prompt: Optional[str] = None,
        company_info: Optional[dict] = None,
    ) -> str:

        # ─── FIX: Fallback if called with only 1 argument (e.g. get_response(prompt)) ───
        if prompt is None:
            prompt = client_id
            client_id = "default_client"

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
