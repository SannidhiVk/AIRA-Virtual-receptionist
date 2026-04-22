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
# FIX D — Removed the instruction telling the LLM to use filler openers
# like "Certainly!", "Absolutely!", "Of course!" — the meta prompt explicitly
# bans these. Having both was contradictory; the guide's rule wins.
# ─────────────────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = f"""You are {AI_NAME}, the smart AI receptionist at {COMPANY_NAME}.

IDENTITY & TONE:
- Be warm and human. Use 1-2 concise sentences.
- DO NOT use filler openers like "Certainly!", "Absolutely!", "Of course!", or "I understand." Just answer.
- Never argue about names or greetings. Trust the visitor.

DELIVERY PROTOCOL:
- If someone mentions "Swiggy", "Zomato", "Food", or "Delivery", categorize them as 'food_delivery'.
- Tell them: "Please leave the package at the front desk. I'll notify the recipient right away."
- DO NOT ask "Should I log this?". Log it automatically.

SMART DIRECTORY SEARCH:
- If a user asks for a 'Sales Team' or 'Manager' and you don't have a specific name, check if you have a Department head (like Jack for Sales). 
- If no record is found, say: "I couldn't find a specific person for that, but I can notify our administration team to help you."
"""

EXTRACT_SYSTEM = """You are an NLU engine. Extract JSON.
- intent: "check_in", "schedule_meeting", "confirm", "general"
- visitor_type: "employee", "delivery", "guest"

EXAMPLES:
"I am Alex and I'm the new intern" -> { "visitor_name": "Alex", "visitor_type": "employee", "purpose": "new intern joining" }
"I am from Flipkart" -> { "visitor_name": "Flipkart", "visitor_type": "delivery" }

VISITOR_TYPE MUST BE: "employee", "food_delivery", "delivery", "interviewee", "client", or "contractor".
- Staff / I work here / Manager = "employee"
- Swiggy / Zomato / Food = "food_delivery"
- Amazon / Courier / Package = "delivery"
- Electrician / Plumber / Maintenance = "contractor"
- Job Interview / Candidate = "interviewee"
"""


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION PROMPT (NLU ENGINE)
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
    "visitor_type": <string|null>,
    "email": <string|null>
  }
}

INTENT CLASSIFICATION RULES:
- "check_in" : Arriving at the office NOW to start work, see someone, or drop off a package.
- "schedule_meeting" : Wants to BOOK or SET UP a FUTURE appointment.
- "employee_lookup" : Asking for availability or location of a colleague/department.
- "confirm" : Saying yes, correct, or "schedule it".
- "general_conversation" : Greetings or small talk.

VISITOR TYPE CATEGORIES (MUST BE ONE OF THESE):
- "Employee" : Staff members, managers, or anyone who says "I work here".
- "Delivery" : Amazon, Flipkart, DHL, or general package couriers.
- "Food Delivery" : Swiggy, Zomato, or food orders.
- "Interviewee" : Job candidates or HR interviews.
- "Contractor/Vendor" : Maintenance, electrician, plumber, or service staff.
- "Client" : External business customers or demos.
- "Visitor/Guest" : General personal or business meetings.

FEW-SHOT EXAMPLES:

Input: "I am Priya and I am an employee here."
Output: {"intent": "check_in", "entities": {"visitor_name": "Priya", "employee_name": null, "role": "Employee", "date": null, "time": null, "purpose": "reporting for work", "visitor_type": "Employee"}}

Input: "I'm from Amazon to drop off a parcel for Virat."
Output: {"intent": "check_in", "entities": {"visitor_name": "Amazon", "employee_name": "Virat", "role": null, "date": null, "time": null, "purpose": "parcel delivery", "visitor_type": "Delivery"}}

Input: "Yes, please schedule that for 5 p.m. today."
Output: {"intent": "confirm", "entities": {"visitor_name": null, "employee_name": null, "role": null, "date": "today", "time": "5:00 PM", "purpose": null, "visitor_type": null}}
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


# FIX E — Added "Jarvis" to the prefix-strip list so "Jarvis: ..." replies
# are cleaned the same way "AI: ..." and "Assistant: ..." are.
def _clean_reply(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"^(AI|Assistant|Jarvis|Sannika|AlmostHuman)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
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
                max_tokens=120,
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
Tone: warm, smart, human, professional. Do not use filler openers like "Certainly!" or "Absolutely!"."""
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
