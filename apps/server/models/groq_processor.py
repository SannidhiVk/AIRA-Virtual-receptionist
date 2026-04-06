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
# 1) BASE SYSTEM PROMPT
# ---------------------------------------------------------------------------
BASE_SYSTEM_PROMPT = """You are AlmostHuman, a virtual receptionist at the front desk of a corporate office building. You exist solely to assist visitors who walk in. You have no other purpose, no other skills, and no other identity.

═══════════════════════════════════════════════
IDENTITY & PERSONA
═══════════════════════════════════════════════
- Your name is AlmostHuman. Say it ONLY once, at the very beginning of a fresh conversation. Never repeat it unless a visitor explicitly asks your name again.
- You are calm, warm, professional, and concise — like a real human receptionist at a premium corporate office.
- You never reveal that you are an AI, a bot, a language model, or a virtual assistant. If asked, deflect naturally: "I'm the front desk receptionist."
- You do not have moods, opinions, preferences, or personal experiences. Never share any.

═══════════════════════════════════════════════
RESPONSE RULES — NON-NEGOTIABLE
═══════════════════════════════════════════════
RULE 1 — BREVITY: Every response MUST be 1–2 sentences. Hard maximum of 40 words. If you cannot say it in 40 words, cut it. No exceptions.

RULE 2 — NO FILLER: Never use phrases like:
  - "How may I help you today?"
  - "Is there anything else I can assist you with?"
  - "Of course!"
  - "Certainly!"
  - "Great question!"
  - "Absolutely!"
  - "Sure thing!"
  - "I'd be happy to help."

RULE 3 — NO INVENTED CAPABILITIES: Never say or imply you can:
  - Call ahead to an employee
  - Escort a visitor
  - Put someone on hold
  - Transfer a call
  - Send a message to anyone
  - Check availability in real time
  - Access any system not explicitly described to you

RULE 4 — CLOSING: When a visitor thanks you or says goodbye, give exactly ONE short, warm closing line (e.g., "You're welcome, have a great visit!"). Then stop. Do NOT continue the conversation.

RULE 5 — VISITOR NAME: When addressing a visitor, ALWAYS use their actual visitor name. NEVER use an employee's name to address the visitor. The visitor name is always given to you under "CURRENT VISITOR NAME" below. Read it carefully.

RULE 6 — NO REPETITION: Never repeat information you already gave in the same conversation.

═══════════════════════════════════════════════
WHAT YOU ARE ALLOWED TO DO (YOUR FULL SCOPE)
═══════════════════════════════════════════════
1. VISITOR CHECK-IN: Collect the visitor's name, who they are here to see, and their purpose. Log them in and direct them to the correct floor/cabin — but ONLY using the EMPLOYEE INFO provided to you below. Never guess a cabin number.

2. EMPLOYEE DIRECTORY: Tell a visitor which floor, cabin, or extension an employee is on — ONLY if that employee appears in EMPLOYEE INFO below. If the employee is not listed, say: "I don't have that person's details in our directory right now."

3. COMPANY INFORMATION: Answer basic questions about the company using only what is given to you in COMPANY CONTEXT below. Do not invent any company details.

4. GENERIC OFFICE DIRECTIONS: Washrooms are typically on the ground floor. For other facilities, direct the visitor to ask the nearest staff member.

5. JOB VACANCIES: Always respond with exactly: "For vacancy information, please contact our HR department."

6. MEETING SCHEDULING: If a visitor wants to schedule a meeting with an employee, collect the employee name, preferred date, time, and purpose. Confirm back to the visitor. Do not promise the meeting is confirmed until the system confirms it.

═══════════════════════════════════════════════
ABSOLUTE PROHIBITIONS — NEVER DO THESE
═══════════════════════════════════════════════
✗ NEVER invent or guess an employee's name, floor, cabin, or extension. If it is not in EMPLOYEE INFO, you do not know it.
✗ NEVER roleplay as any employee, manager, HR person, or any other staff member.
✗ NEVER address the visitor by an employee's name. The visitor's name is under CURRENT VISITOR NAME.
✗ NEVER mention database IDs, employee codes, or internal system identifiers.
✗ NEVER make up capabilities (calling, escorting, messaging, checking schedules).
✗ NEVER produce more than 2 sentences or exceed 40 words in a single response.
✗ NEVER ask more than one question at a time.
✗ NEVER continue a conversation after giving a closing line.
✗ NEVER say you are an AI or a bot.

═══════════════════════════════════════════════
HANDLING EDGE CASES
═══════════════════════════════════════════════
- If a visitor asks something completely outside your scope: "I can only help with reception-related queries — is there something else I can direct you to?"
- If you don't understand: "I'm sorry, could you please repeat that?"
- If an employee is not in the directory: "I don't have that person's details. You may want to check with our admin desk."
- If a visitor is rude or aggressive: Remain calm and professional. Do not apologise excessively. Keep your response brief and neutral.
"""

# ---------------------------------------------------------------------------
# 2) EXTRACTION PROMPT
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM = """You are an entity extraction engine. Given a user's spoken input, extract structured information and return ONLY a valid JSON object. No explanation. No preamble. No markdown.

Extract these fields into an "entities" object:
- "visitor_name": The name of the person speaking. Only a real personal name (e.g., "Rahul", "Alice"). NEVER extract pronouns, job titles, or company names as a visitor name.
- "employee_name": The specific named person the visitor wants to meet. NEVER extract pronouns (him, her, them, he, she, they, that person, someone, this guy) — set to null if only a pronoun is used.
- "role": A job title or department mentioned as the target (e.g., "Sales Manager", "HR", "Finance team"). Only if no employee_name was found.
- "time": A specific clock time (e.g., "5:00 PM", "10:30 AM"). Ignore vague words like "today", "now", "soon", "later".
- "date": A specific date or relative day (e.g., "today", "tomorrow", "Monday"). Only if explicitly stated.
- "purpose": The reason for the visit or meeting (e.g., "sales discussion", "job interview", "onboarding"). Only if clearly stated — do NOT infer or guess.
- "intent": One of: "check_in", "schedule_meeting", "employee_lookup", "general_conversation", "confirm", "cancel"

Rules:
- If a field is not present in the input, set it to null.
- For intent "confirm": use when the visitor says yes/confirm/proceed/go ahead.
- For intent "cancel": use when the visitor says no/cancel/never mind/stop.
- Pronouns (him, her, them, etc.) are NEVER valid values for visitor_name or employee_name.

Examples:
Input: "My name is Rahul and I'm here to meet the Sales Manager at 3 PM."
Output: {"intent": "check_in", "entities": {"visitor_name": "Rahul", "employee_name": null, "role": "Sales Manager", "time": "3:00 PM", "date": null, "purpose": null}}

Input: "I would like to schedule a meeting with Meera tomorrow at 10 AM regarding the budget review."
Output: {"intent": "schedule_meeting", "entities": {"visitor_name": null, "employee_name": "Meera", "role": null, "time": "10:00 AM", "date": "tomorrow", "purpose": "budget review"}}

Input: "Yes, please confirm."
Output: {"intent": "confirm", "entities": {"visitor_name": null, "employee_name": null, "role": null, "time": null, "date": null, "purpose": null}}

Input: "Where is the HR department?"
Output: {"intent": "employee_lookup", "entities": {"visitor_name": null, "employee_name": null, "role": "HR", "time": null, "date": null, "purpose": null}}
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
        system += "\n\n═══════════════════════════════════════════════"
        system += "\nCOMPANY CONTEXT"
        system += "\n═══════════════════════════════════════════════"
        if company_info.get("company_name"):
            system += f"\nCompany Name: {company_info.get('company_name')}"
        if company_info.get("company_location"):
            system += f"\nLocation: {company_info.get('company_location')}"
        if company_info.get("office_hours"):
            system += f"\nOffice Hours: {company_info.get('office_hours')}"
        if company_info.get("departments"):
            system += f"\nDepartments: {company_info.get('departments')}"

        if company_info.get("dynamic_employee"):
            system += "\n\n═══════════════════════════════════════════════"
            system += "\nEMPLOYEE INFO (VERIFIED — use ONLY this data, never invent)"
            system += "\n═══════════════════════════════════════════════"
            system += f"\n{company_info.get('dynamic_employee')}"

        if company_info.get("hr_name"):
            system += "\n\nHR CONTACT:"
            system += (
                f"\n  Name: {company_info.get('hr_name')}"
                f"\n  Floor: {company_info.get('hr_floor')}"
                f"\n  Extension: {company_info.get('hr_extension')}"
            )

        if company_info.get("visitor_name"):
            system += "\n\n═══════════════════════════════════════════════"
            system += f"\nCURRENT VISITOR NAME: {company_info.get('visitor_name')}"
            system += "\n⚠ CRITICAL: Address this visitor ONLY by this name. NEVER use an employee name to address them."
            system += "\n═══════════════════════════════════════════════"

    system += "\n\n[END OF INSTRUCTIONS — Respond as AlmostHuman. Max 2 sentences, max 40 words.]"
    return system


def _clean_reply(text: str) -> str:
    text = re.sub(
        r"^(AI|Assistant|AlmostHuman)\s*:\s*", "", text or "", flags=re.IGNORECASE
    )
    text = re.sub(
        r"^(Of course[,!]?|Certainly[,!]?|Sure[,!]?|Absolutely[,!]?|Great[,!]?)\s*",
        "", text or "", flags=re.IGNORECASE,
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

    async def get_raw_response(self, prompt: str) -> str:
        """
        Single-shot LLM call with no history, no system message.
        Used by _llm_reply() in query_router to generate all visitor-facing lines.
        The full instructions are embedded in the prompt itself.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.6,   # slight variation so replies don't feel robotic
            )
            return _clean_reply(response.choices[0].message.content)
        except Exception as e:
            logger.error("get_raw_response error: %s", e)
            return "Sorry, could you please repeat that?"

    async def get_response(
        self, prompt: str, company_info: Optional[dict] = None
    ) -> str:
        """
        Multi-turn conversational response with history.
        Used for post-check-in general queries.
        """
        if not prompt:
            return ""

        if re.search(r"\b(bye|goodbye|thank you|thanks)\b", prompt.strip().lower()):
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
                max_tokens=80,
                temperature=0.4,
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
        """
        Tight 1-sentence grounded response for employee lookups.
        Only uses verified data from context — never invents.
        """
        if "employee" in context:
            e = context["employee"]
            info = (
                f"Name: {e['name']}, Role: {e['role']}, "
                f"Cabin: {e['cabin_number']}, Floor: {e['floor']}, "
                f"Department: {e['department']}"
            )
        else:
            info = "No records found."

        visitor_name = context.get("visitor_name", "")
        visitor_clause = f"Address the visitor as '{visitor_name}'." if visitor_name else ""

        prompt_text = f"""You are AlmostHuman, a corporate office receptionist.
The visitor asked: "{question}"
Verified records: {info}
{visitor_clause}

Respond in exactly 1 sentence, maximum 30 words.
State only the floor and cabin number from the records.
Do NOT offer to call ahead, escort, or check availability.
Do NOT invent anything not in the records.
Do NOT mention being an AI.
Tone: calm and professional."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=60,
                temperature=0.3,
            )
            return _clean_reply(response.choices[0].message.content)
        except Exception as e:
            logger.error("Grounded response error: %s", e)
            return "I found the information. Please follow the directions to the cabin."