import json
import logging
import os
import re
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import AsyncGroq

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Private Limited."

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS (DETAILED)
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

# Detailed NLU Extraction Prompt
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
- "general" : Greetings or small talk.

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
# ENV / API KEY HELPERS (PRESERVED)
# ─────────────────────────────────────────────────────────────────────────────


def _load_dotenv_from_any_location() -> None:
    """Attempts to find and load the .env file from multiple directory levels."""
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
        logger.warning("No .env file found in: %s", [str(c) for c in candidates])
    except ImportError:
        # Fallback manual parsing if python-dotenv is missing
        pass


def _read_api_key() -> str:
    """Reads the primary Groq API key."""
    _load_dotenv_from_any_location()
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    return ""


def _get_all_groq_keys() -> List[str]:
    """Retrieves all available keys for rotation."""
    _load_dotenv_from_any_location()
    keys = [
        os.getenv("GROQ_API_KEY", "").strip(),
        os.getenv("GROQ_API_KEY_2", "").strip(),
    ]
    return [k for k in keys if k]


def _build_system_message(company_info: Optional[dict] = None) -> str:
    """Constructs the final system prompt with current time and company context."""
    now = datetime.now()
    current_time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

    # CALCULATE GREETING ONCE HERE
    hour = now().hour
    if 5 <= hour < 12:
        greeting = "Good Morning"
    elif 12 <= hour < 17:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"

    system = (
        BASE_SYSTEM_PROMPT
        + f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + f"CURRENT GREETING: {greeting}\n"  # FORCE THE GREETING
        + f"CURRENT DATE & TIME: {current_time_str}\n"
        + f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
    """Strips common AI prefixes from the response text."""
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
# GROQ PROCESSOR CLASS (WITH ACCOUNT ROTATION)
# ─────────────────────────────────────────────────────────────────────────────
def _get_current_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"


class GroqProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Initialize Dual Account Rotation
        self.api_keys = _get_all_groq_keys()
        if not self.api_keys:
            logger.error("No Groq API keys found. System will fail.")

        # Create a pool of clients to cycle through
        self.clients = [AsyncGroq(api_key=key) for key in self.api_keys]
        self.client_cycle = itertools.cycle(self.clients)

        self.model_name = MODEL
        self.client_history: Dict[str, List[Dict[str, str]]] = {}
        logger.info(
            "GroqProcessor initialized with %d-account rotation.", len(self.api_keys)
        )

    async def _call_with_rotation(
        self, messages: list, max_tokens: int, temperature: float
    ) -> str:
        """Internal helper to attempt a call with the current key, rotating on 429 errors."""
        num_attempts = len(self.clients)

        for _ in range(num_attempts):
            current_client = next(self.client_cycle)
            try:
                response = await current_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                # Check specifically for rate limiting
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    logger.warning(
                        "Current Groq account rate limited. Rotating to next account..."
                    )
                    continue
                # For other errors, log and potentially bubble up or try next
                logger.error("Groq call error: %s", e)
                continue

        return "I'm sorry, my thinking systems are currently busy. Please try again in a moment."

    def reset_history(self, client_id: str):
        """Wipes the conversation memory for a specific client."""
        self.client_history[client_id] = []
        logger.info(f"GroqProcessor conversation history reset for {client_id}.")

    async def get_raw_response(
        self, prompt: str, client_id: Optional[str] = None
    ) -> str:
        """Direct LLM call without system prompt formatting."""
        try:
            if client_id and client_id in self.client_history:
                context_messages = self.client_history[client_id][-2:]
            else:
                context_messages = []

            messages = context_messages + [{"role": "user", "content": prompt}]

            content = await self._call_with_rotation(
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
            return "I'm sorry, I didn't quite catch that. Could you say it again?"

    async def get_response(
        self,
        client_id: str = "default_client",
        prompt: Optional[str] = None,
        company_info: Optional[dict] = None,
    ) -> str:
        """Main conversational entry point."""
        if prompt is None:
            prompt = client_id
            client_id = "default_client"

        if not prompt:
            return ""

        if client_id not in self.client_history:
            self.client_history[client_id] = []

        # Detect termination phrases
        if re.search(r"\b(bye|goodbye|thank you|thanks)\b", prompt.strip().lower()):
            self.reset_history(client_id)

        # Truncate history to save context tokens
        if len(self.client_history[client_id]) > 6:
            self.client_history[client_id] = self.client_history[client_id][-6:]

        self.client_history[client_id].append({"role": "user", "content": prompt})

        messages = [
            {"role": "system", "content": _build_system_message(company_info)}
        ] + self.client_history[client_id]

        try:
            content = await self._call_with_rotation(
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
        """Extracts structured data from user utterances using NLU engine."""
        try:
            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user_query.strip()},
            ]

            content = await self._call_with_rotation(
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
        """Generates a response based on specific database results."""
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
            messages = [{"role": "user", "content": prompt}]
            content = await self._call_with_rotation(
                messages, max_tokens=80, temperature=0.4
            )
            return _clean_reply(content)
        except Exception:
            return "Please head over to the main lobby, and someone will assist you shortly."


# ─────────────────────────────────────────────────────────────────────────────
# END OF FILE
# ─────────────────────────────────────────────────────────────────────────────
