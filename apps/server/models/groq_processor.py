import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import AsyncGroq
from openai import AsyncOpenAI  # Used for SambaNova failover

logger = logging.getLogger(__name__)

# Constants
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# FIXED: Updated model name to match current SambaNova Cloud availability
SAMBA_MODEL = "Meta-Llama-3.3-70B-Instruct"
AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Private Limited."

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = f"""You are {AI_NAME}, the expert AI receptionist at {COMPANY_NAME}.
STRICT TONE: Concise, professional, 1-2 sentences only. 

RULES:
1. GREETING: Check current time. 05:00-11:59: "Good Morning". 12:00-16:59: "Good Afternoon". 17:00-04:59: "Good Evening".
2. NO REPETITION: If you have already confirmed a name or host, NEVER ask for it again.
3. SCHEDULING: If the user says "Only me", "Just us", or "No one else", set attendees to 'Finalized' and proceed to Date/Time immediately.
4. TRUST THE USER: If they say "I am an employee," categorize them as 'Employee' immediately.
5. SLACK: Only notify the HOST (the person they are meeting). Never notify the visitor's colleague.
"""

EXTRACT_SYSTEM = """You are an information extraction engine for a corporate receptionist system.
Given a visitor's spoken input, extract structured data and return ONLY a valid JSON object.
No markdown. No explanation. No preamble. Just the JSON.

Output format:
{
  "intent": "check_in" | "schedule_meeting" | "employee_lookup" | "confirm" | "general",
  "entities": {
    "visitor_name": string | null,
    "employee_name": string | null,
    "role": string | null,
    "date": string | null,
    "time": string | null,
    "purpose": string | null,
    "visitor_type": string | null,
    "email": string | null
  }
}

INTENT CLASSIFICATION RULES:
- "check_in" : Arriving at the office NOW to start work, see someone, or drop off a package.
- "schedule_meeting" : Wants to BOOK or SET UP a FUTURE appointment.
- "employee_lookup" : Asking for availability or location of a colleague/department.
- "confirm" : Saying yes, correct, or "schedule it".
- "general" : Greetings or small talk.

VISITOR TYPE CATEGORIES (MUST BE ONE OF THESE):
- "Employee" : Staff members, managers, or anyone who says "I work here", "Internal", or "Intern".
- "Delivery" : Amazon, Flipkart, DHL, or general package couriers.
- "Food Delivery" : Swiggy, Zomato, or food orders.
- "Interviewee" : Job candidates or HR interviews.
- "Contractor/Vendor" : Maintenance, electrician, plumber, service staff, "Urban Company", "Fix".
- "Client" : External business customers, "Demo", or "HDFC".
- "Visitor/Guest" : General personal or business meetings.

FEW-SHOT EXAMPLES:
Input: "I am Priya and I am an employee here."
Output: {"intent": "check_in", "entities": {"visitor_name": "Priya", "employee_name": null, "role": "Employee", "date": null, "time": null, "purpose": "reporting for work", "visitor_type": "Employee"}}

Input: "I'm from Amazon to drop off a parcel for Virat."
Output: {"intent": "check_in", "entities": {"visitor_name": "Amazon", "employee_name": "Virat", "role": null, "date": null, "time": null, "purpose": "parcel delivery", "visitor_type": "Delivery"}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# ENV / API KEY HELPERS
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
                logger.info("Loaded .env from: %s", path)
                return
    except ImportError:
        pass


def _read_api_key() -> str:
    _load_dotenv_from_any_location()
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    return env_key


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
        r"^(AI|Assistant|Jarvis|Sannika|AlmostHuman)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# GROQ PROCESSOR (WITH UPDATED SAMBANOVA FAILOVER)
# ─────────────────────────────────────────────────────────────────────────────


class GroqProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 1. Initialize Groq
        api_key = _read_api_key()
        raw_base = os.getenv("GROQ_BASE_URL", "").strip().rstrip("/")
        if raw_base and "/openai" not in raw_base:
            self.groq_client = AsyncGroq(api_key=api_key, base_url=raw_base)
        else:
            self.groq_client = AsyncGroq(api_key=api_key)

        # 2. Initialize SambaNova (Failover)
        self.samba_client = AsyncOpenAI(
            api_key=os.getenv("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )

        self.groq_model = MODEL
        self.samba_model = SAMBA_MODEL
        self.client_history: Dict[str, List[Dict[str, str]]] = {}
        logger.info("GroqProcessor initialized with Dual-Provider Failover.")

    async def _call_with_failover(
        self, messages: list, max_tokens=150, temperature=0.5
    ):
        """Attempts Groq first, falls back to SambaNova on Rate Limit or Error."""
        try:
            # Primary: Groq
            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Check for Rate Limit (429) or general failure
            if "429" in str(e) or "rate_limit" in str(e).lower():
                logger.warning("Groq Rate Limit hit. Failing over to SambaNova...")
            else:
                logger.error(f"Groq error: {e}. Attempting failover...")

            try:
                # Secondary: SambaNova
                response = await self.samba_client.chat.completions.create(
                    model=self.samba_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e2:
                logger.critical(
                    f"FAILOVER FAILED: Both Groq and SambaNova (Model: {self.samba_model}) are unavailable. Error: {e2}"
                )
                return "I'm sorry, I'm having trouble connecting to my central systems. Could you please try again in a moment?"

    def reset_history(self, client_id: str):
        self.client_history[client_id] = []
        logger.info(f"GroqProcessor conversation history reset for {client_id}.")

    async def get_raw_response(
        self, prompt: str, client_id: Optional[str] = None
    ) -> str:
        try:
            if client_id and client_id in self.client_history:
                context_messages = self.client_history[client_id][-2:]
            else:
                context_messages = []

            messages = context_messages + [{"role": "user", "content": prompt}]
            content = await self._call_with_failover(
                messages, max_tokens=120, temperature=0.6
            )
            reply = _clean_reply(content)

            if client_id:
                if client_id not in self.client_history:
                    self.client_history[client_id] = []
                self.client_history[client_id].append(
                    {"role": "assistant", "content": reply}
                )

            return reply
        except Exception as e:
            logger.error("get_raw_response error: %s", e)
            return "I'm sorry, I didn't quite catch that."

    async def get_response(
        self,
        client_id: str = "default_client",
        prompt: Optional[str] = None,
        company_info: Optional[dict] = None,
    ) -> str:
        if prompt is None:
            prompt = client_id
            client_id = "default_client"
        if not prompt:
            return ""

        if client_id not in self.client_history:
            self.client_history[client_id] = []

        if re.search(r"\b(bye|goodbye|thank you|thanks)\b", prompt.strip().lower()):
            self.reset_history(client_id)

        if len(self.client_history[client_id]) > 6:
            self.client_history[client_id] = self.client_history[client_id][-6:]

        self.client_history[client_id].append({"role": "user", "content": prompt})
        messages = [
            {"role": "system", "content": _build_system_message(company_info)}
        ] + self.client_history[client_id]

        try:
            content = await self._call_with_failover(
                messages, max_tokens=100, temperature=0.5
            )
            reply = _clean_reply(content)
            self.client_history[client_id].append(
                {"role": "assistant", "content": reply}
            )
            return reply
        except Exception as e:
            logger.error("Groq chat error: %s", e)
            return "I'm sorry, my connection blinked. Could you repeat that?"

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user_query.strip()},
            ]
            content = await self._call_with_failover(
                messages, max_tokens=250, temperature=0
            )

            raw = (content or "").strip()
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
                "intent": parsed.get("intent", "general"),
                "entities": entities,
            }
        except Exception as e:
            logger.error("Extraction failed: %s", e)
            return {"intent": "general", "entities": {}}

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
            content = await self._call_with_failover(
                [{"role": "user", "content": prompt}], max_tokens=80, temperature=0.4
            )
            return _clean_reply(content)
        except Exception:
            return "Please head over to the main lobby, and someone will assist you shortly."
