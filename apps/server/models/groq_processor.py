import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import AsyncGroq

logger = logging.getLogger(__name__)


MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# 1) BASE SYSTEM PROMPT (guardrails)
# ---------------------------------------------------------------------------
BASE_SYSTEM_PROMPT = """You are AlmostHuman, a virtual receptionist for a corporate office. You ONLY do reception duties.

INTRODUCTION:
- Say your name "AlmostHuman" only once at the very start. Never again unless asked.

RESPONSE STYLE:
- Maximum 2 sentences. Maximum 40 words. Be direct. No filler phrases.
- Never ask follow-up questions unless absolutely necessary.
- Never say "How may I help you?" or "Is there anything else?"
- If visitor thanks you, give one short warm closing. End the conversation.

YOUR SCOPE — YOU CAN ONLY DO THESE THINGS:
1. Help visitors check in.
2. Tell visitors which floor/extension an employee or department is on — BUT ONLY if that info is given to you in EMPLOYEE INFO below.
3. Answer basic questions about the company using COMPANY CONTEXT below.
4. Give generic directions inside the office (washroom: usually ground floor, cafeteria: ask staff).
5. For job vacancies, say: "Please contact our HR department for vacancy information."

WHAT YOU MUST NEVER DO — THESE ARE ABSOLUTE RULES:
- NEVER invent any employee name. If you do not see the name in EMPLOYEE INFO, you do not know it.
- NEVER roleplay as any other person — not HR, not a manager, not anyone else.
- NEVER pretend to transfer a call or put someone on hold.
- NEVER ask for or mention internal database IDs, department codes, or employee codes.
- Never mention being an AI.
"""

# ---------------------------------------------------------------------------
# 2) EXTRACTION PROMPT (structured JSON output)
# ---------------------------------------------------------------------------
# --- groq_processor.py ---

EXTRACT_SYSTEM = """
Extract entities from the user's input. Return ONLY a JSON object.
Rules:
1. 'visitor_name': The person speaking.
2. 'employee_name': The specific person they want to meet. 
   - IMPORTANT: NEVER extract pronouns like 'him', 'her', 'them', 'that person' as an employee_name. Leave it null if only a pronoun is used.
3. 'role': Job title mentioned (e.g. HR, Manager).
4. 'time': Specific time (ignore 'today', 'now', 'soon').
5. 'intent': 'check_in', 'schedule_meeting', or 'employee_lookup'.

Example: "Schedule a meeting with him at 2pm"
Output: {"intent": "schedule_meeting", "entities": {"time": "2:00 PM"}}
"""


def _load_dotenv_from_any_location() -> None:
    """
    Manually load .env from the most likely locations because uv does not
    automatically inject .env variables into the subprocess environment.
    We try several candidate paths so this works regardless of where the
    .env file lives relative to the project root.
    """
    try:
        from dotenv import load_dotenv as _load
        here = Path(__file__).resolve()
        candidates = [
            here.parent / ".env",                    # apps/server/.env
            here.parent.parent / ".env",             # apps/.env
            here.parent.parent.parent / ".env",      # project root .env
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
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
                    logger.info("Manually parsed .env from: %s", path)
                    return
                except Exception as e:
                    logger.error("Failed to parse .env at %s: %s", path, e)


def _read_api_key() -> str:
    """
    Load .env first, then read GROQ_API_KEY.
    Fallback: GROQ_API_KEY.txt for legacy local workflow.
    """
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
                logger.info("Loaded GROQ_API_KEY from GROQ_API_KEY.txt")
                return key
    except Exception as e:
        logger.error("Failed reading GROQ_API_KEY.txt: %s", e)

    logger.error(
        "GROQ_API_KEY not found! Checked env, .env file, and GROQ_API_KEY.txt. "
        "Make sure your .env exists at apps/server/.env or the project root."
    )
    return ""


def _build_system_message(company_info: Optional[dict] = None) -> str:
    system = BASE_SYSTEM_PROMPT

    if company_info:
        system += "\n\nCOMPANY CONTEXT:"
        system += f"\nCompany: {company_info.get('company_name', '')}"
        system += f"\nLocation: {company_info.get('company_location', '')}"
        system += f"\nOffice Hours: {company_info.get('office_hours', '')}"
        system += f"\nDepartments: {company_info.get('departments', '')}"

        if company_info.get("dynamic_employee"):
            system += (
                "\n\nRELEVANT EMPLOYEE INFO "
                "(The visitor is asking about someone here. "
                "Ignore minor name typos):"
            )
            system += f"\n{company_info.get('dynamic_employee')}"

        if company_info.get("hr_name"):
            system += "\n\nHR CONTACT INFO:"
            system += (
                f"\nHR Manager: {company_info.get('hr_name')} — "
                f"{company_info.get('hr_floor')} — Extension {company_info.get('hr_extension')}"
            )

    system += "\n\nREMINDER: You are ONLY a receptionist. Never roleplay as anyone else. Never invent names."
    return system


def _clean_reply(text: str) -> str:
    text = re.sub(
        r"^(AI|Assistant|AlmostHuman)\s*:\s*", "", text or "", flags=re.IGNORECASE
    )
    return (text or "").strip()


class GroqProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        api_key = _read_api_key()

        # Guard against GROQ_BASE_URL env var doubling the path.
        # The Groq SDK already uses https://api.groq.com as its default —
        # passing a base_url that already includes /openai/v1 causes the SDK
        # to produce /openai/v1/openai/v1/... (404).  Only pass base_url if
        # the env var is set AND looks like a bare host (no path component).
        raw_base = os.getenv("GROQ_BASE_URL", "").strip().rstrip("/")
        if raw_base and "/openai" not in raw_base:
            self.client = AsyncGroq(api_key=api_key, base_url=raw_base)
        else:
            # Let the SDK use its own default — never pass a broken base_url.
            self.client = AsyncGroq(api_key=api_key)
        self.model_name = MODEL
        self.history: List[Dict[str, str]] = []
        logger.info("GroqProcessor initialized with model '%s'", self.model_name)

    def reset_history(self):
        self.history = []
        logger.info("GroqProcessor conversation history reset.")

    async def get_response(
        self, prompt: str, company_info: Optional[dict] = None
    ) -> str:
        if not prompt:
            return ""

        if prompt.strip().lower() in ["bye", "goodbye", "thank you", "thanks"]:
            self.reset_history()

        if len(self.history) > 12:
            self.history = self.history[-12:]

        self.history.append({"role": "user", "content": prompt})
        messages = [
            {"role": "system", "content": _build_system_message(company_info)}
        ] + self.history

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.5,
            )
            content = _clean_reply(response.choices[0].message.content)
            self.history.append({"role": "assistant", "content": content})
            return content
        except Exception as e:
            logger.error("Groq chat error: %s", e)
            return "I'm sorry, I didn't catch that. Could you please repeat?"

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        raw = None
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": user_query.strip()},
                ],
                max_tokens=200,
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()

            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip()

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            parsed = json.loads(raw)
            entities = parsed.get("entities", parsed)
            if not isinstance(entities, dict):
                entities = parsed

            return {
                "intent": parsed.get("intent", "general_conversation"),
                "entities": entities,
            }
        except Exception as e:
            logger.error("Extraction failed: %s | Raw: %s", e, raw)
            return {"intent": "general_conversation", "entities": {}}

    async def generate_grounded_response(self, context: dict, question: str) -> str:
        if "employee" in context:
            e = context["employee"]
            info = (
                f"Name: {e['name']}, Role: {e['role']}, "
                f"Cabin: {e['cabin_number']}, Department: {e['department']}"
            )
        else:
            info = "No records found."

        prompt_text = f"""You are a professional office receptionist.
The person asked: "{question}"
Internal records show: {info}

Task:
- Use the info to guide them to the right place.
- Keep it to 1-2 sentences.
- Tone: Friendly and professional.
- Never mention being an AI."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": prompt_text}],
                max_tokens=100,
                temperature=0.5,
            )
            return _clean_reply(response.choices[0].message.content)
        except Exception as e:
            logger.error("Grounded response error: %s", e)
            return "I found the information. Please follow the directions to the cabin."