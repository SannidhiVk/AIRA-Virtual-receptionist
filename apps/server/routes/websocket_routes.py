import asyncio
import base64
import json
import logging
import time
import io
import os
import wave
import re
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from silero_vad import load_silero_vad, get_speech_timestamps
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from managers.connection_manager import manager
from models.whisper_processor import WhisperProcessor
from models.tts_processor import KokoroTTSProcessor
from services.processor_service import process_text_for_client
from services.wake_word_service import get_wake_word_service

# --- Other imports remain the same ---
from services.face_recognition_service import verify_person_face, warmup_deepface
from services.person_detection_service import (
    get_person_detection_service,
    warmup_mediapipe,
)
from services.query_router import (
    clear_session_state,
    mark_employee_from_face_result,  # <--- ADD THIS IMPORT
)

# Thread pool for running blocking DeepFace and MediaPipe calls without blocking the async event loop.
_face_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="deepface")

# Trigger model warmup immediately when the server file is loaded
_face_executor.submit(warmup_deepface)
_face_executor.submit(warmup_mediapipe)

logging.basicConfig(
    # ... rest of your code ...
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
router = APIRouter()

vad_model = load_silero_vad()
WAKE_WORD = os.getenv("OPENWAKEWORD_WAKEWORD", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.getenv("OPENWAKEWORD_THRESHOLD", "0.35"))
ww_service = get_wake_word_service(WAKE_WORD)
ww_service.threshold = WAKE_WORD_THRESHOLD

SAMPLE_RATE = 16000
OWW_CHUNK_SAMPLES = 1280
OWW_CHUNK_BYTES = OWW_CHUNK_SAMPLES * 2
SILERO_WINDOW_SAMPLES = 8000
MAX_SILENCE_MS = 1200
FOLLOWUP_TIMEOUT_SECONDS = float(os.getenv("FOLLOWUP_TIMEOUT", "12.0"))
MAX_MISMATCH_STRIKES = 3

_NAME_STOPWORDS = frozenset(
    {
        "i",
        "i'm",
        "im",
        "it",
        "it's",
        "its",
        "i've",
        "ive",
        "hello",
        "hi",
        "hey",
        "sorry",
        "so",
        "not",
        "no",
        "yes",
        "me",
        "my",
        "is",
        "am",
        "are",
        "was",
        "the",
        "a",
        "an",
        "here",
        "there",
        "this",
        "that",
        "just",
        "now",
        "please",
        "name",
        "call",
        "called",
        "actually",
        "really",
        "well",
        "oh",
        "ah",
        "um",
        "uh",
        "ok",
        "okay",
        "yeah",
        "yep",
        "and",
        "from",
        "for",
        "with",
        "but",
        "or",
        "to",
        "at",
        "in",
        "on",
        "of",
        "could",
        "would",
        "should",
        "can",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "need",
        "want",
        "like",
        "please",
        "help",
        "hr",
        "interview",
        "delivery",
        "meeting",
        "schedule",
        "visit",
        "here",
    }
)

# Keywords that indicate the speaker is a visitor/delivery person, not an employee
_VISITOR_KEYWORDS = (
    "delivery",
    "courier",
    "package",
    "parcel",
    "swiggy",
    "zomato",
    "amazon",
    "flipkart",
    "dunzo",
    "porter",
    "blinkit",
    "zepto",
    "visitor",
    "guest",
    "appointment",
    "here to meet",
    "here to see",
    "meeting with",
    "visiting",
    "i have a delivery",
    "drop off",
    "interview",
    "interviewing",
    "intern",
    "internship",
    "joining",
    "new joiner",
    "onboarding",
    "new employee",
    "starting today",
    "client",
    "sales demo",
    "demo",
    "sales meeting",
    "here for a",
)


def _detect_person_type(text: str) -> str:
    """
    Returns 'visitor' if the transcript contains delivery/visitor keywords
    OR if the person is simply introducing themselves (not an employee greeting).
    Otherwise returns 'employee'.
    """
    lower = text.lower()

    # Explicit visitor/delivery keywords
    for keyword in _VISITOR_KEYWORDS:
        if keyword in lower:
            return "visitor"

    # Self-introduction patterns ("I am X", "I'm X", "my name is X", "X here")
    # without any employee claim → treat as visitor until DB proves otherwise
    intro_pattern = re.search(r"\b(i am|i'm|my name is|this is)\b", lower)
    if intro_pattern:
        # If they also say "i work here" / "i'm an employee" → employee
        if re.search(
            r"\b(i work here|i am an employee|i'm an employee|staff)\b", lower
        ):
            return "employee"
        return "visitor"

    return "employee"


def _extract_spoken_name(text: str) -> str | None:
    """
    Try to extract a candidate name from phrases like:
      - "I'm John Doe"
      - "I am John Doe"
      - "This is John Doe"
      - "My name is John Doe"
      - "Lucy here" / "It's Lucy here"
    """
    if not text:
        return None

    # Capture exactly 1 or 2 words immediately following the intro phrase.
    # Capture 1-2 words: each word is letters/hyphens/apostrophes only.
    # No dots allowed at end of token — prevents "here." being swallowed.
    name_pattern = r"([A-Za-z'\-]+(?:\s+(?!and\b|or\b|but\b|for\b|from\b|that\b|who\b|which\b)[A-Za-z'\-]+)?)"

    patterns = [
        rf"\b(?:i am|i'm)\s+{name_pattern}",
        rf"\bmy name is\s+{name_pattern}",
        rf"\bthis is\s+{name_pattern}",
        rf"\bit'?s\s+{name_pattern}",
        rf"\b([a-zA-Z'-]+)\s+here\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,!?:;")
            # Strip trailing "here" if captured
            name = re.sub(r"\s+here$", "", name, flags=re.IGNORECASE).strip()

            # --- UPDATED LOGIC ---
            # Reject if the whole phrase is a stopword, OR if the first word is a stopword
            if name.lower() not in _NAME_STOPWORDS and len(name) >= 2:
                first_word = name.split()[0].lower()
                if first_word not in _NAME_STOPWORDS:
                    return name
            # ---------------------

    # Fallback for short direct intros like "John" or "John Doe".
    cleaned = re.sub(r"[^a-zA-Z\s.'-]", " ", text).strip()
    if cleaned:
        words = [w for w in cleaned.split() if w]
        # If the whole utterance is very short (1-3 words)
        if 1 <= len(words) <= 3:
            name_words = [w for w in words if w.lower() not in _NAME_STOPWORDS]
            if 1 <= len(name_words) <= 2 and all(w[0].isupper() for w in name_words):
                return " ".join(name_words)
    return None


def _candidate_names_from_transcript(text: str) -> list[str]:
    if not text:
        return []

    # 1. Strip meeting-target names FIRST, before any extraction runs.
    # Added "meeting with" and "schedule.*?with" to catch phrases like "schedule a meeting with Lucy"
    safe_text = re.sub(
        r"\b(meet|see|looking for|appointment with|here for|visiting|meeting with|schedule.*?with|scheduled.*?with)\s+([A-Z][a-z.\'-]+)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    candidates: list[str] = []

    # 2. Try explicit intro phrases (e.g., "I am Raksha", "Raksha here")
    primary = _extract_spoken_name(safe_text)

    if primary:
        candidates.append(primary)
        parts = primary.split()
        if len(parts) > 1:
            candidates.append(parts[0])

        # --- CRITICAL FIX ---
        # If we found an explicit introduction, DO NOT fall back to scanning
        # other capitalized words. This prevents the system from picking up
        # the host's name just because the visitor's name wasn't in the DB.
    else:
        # 3. Fallback: capitalized words (ONLY if no explicit intro was found)
        capitalized_words = re.findall(r"\b[A-Z][a-z.\'-]+\b", safe_text)
        for cw in capitalized_words:
            if len(cw) >= 3 and cw.lower() not in _NAME_STOPWORDS:
                candidates.append(cw)

    # Preserve order, remove duplicates
    seen = set()
    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate.strip())

    return unique


def _resolve_employee_name(candidate_name: str) -> str | None:
    # Reject obvious non-names before touching the DB to prevent fuzzy false matches
    # e.g. "I'm" fuzzy-matching to "Jim"
    if not candidate_name or len(candidate_name) < 3:
        return None
    if candidate_name.lower() in _NAME_STOPWORDS:
        return None
    try:
        from receptionist.database import get_employee_by_name

        employee = get_employee_by_name(candidate_name)
        if employee:
            return employee.name
    except Exception:
        return None
    return None


def create_wav_from_pcm(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return wav_io.getvalue()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    from services.query_router import (
        clear_session_state,
        get_session_state as _get_existing_state,
    )

    # ✅ Don't wipe session if we're mid-scheduling and waiting for Slack reply
    _existing = None
    try:
        _existing = _get_existing_state(client_id)
    except Exception:
        pass

    # Preserve session if we are waiting for a Slack reply, OR if we recently
    # got a reply and are in the middle of a conversation with a known visitor.
    _preserve_session = _existing and (
        _existing.get("awaiting_slack_reply")
        or (
            _existing.get("visitor_name") is not None
            and not _existing.get("conversation_complete")
        )
    )

    if not _preserve_session:
        clear_session_state(client_id)
    else:
        logger.info(
            f"[{client_id}] Reconnect with pending Slack reply — preserving session."
        )

    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()
    text_queue: asyncio.Queue[str] = asyncio.Queue()
    # ── FIX: Pull any pending reply that was generated but not delivered ────────
    # If the previous connection broke while brain() was sending audio, the reply
    # text is stored in _pending_reply so we can re-speak it on reconnect.
    _pending_reply = _existing.get("_pending_reply") if _existing else None
    if _pending_reply and _preserve_session:
        logger.info(f"[{client_id}] Reconnect: found undelivered reply, will re-speak.")
    # ────────────────────────────────────────────────────────────────────────────

    session_state = {
        "mode": (
            "FOLLOWUP" if _preserve_session else "PASSIVE"
        ),  # ✅ stay in FOLLOWUP if waiting
        "awaiting_face": False,
        "is_verified": False,
        "verified_name": None,
        "claimed_name": None,
        "visitor_reference_image_b64": None,
        "pending_identity_name": None,
        "person_type": "employee",
        "mismatch_strikes": 0,
        "face_verify_in_progress": False,
        "conversation_complete": False,
        "visitor_captured": False,
        "brain_is_thinking": False,
        "presence_count": 0,
        "last_presence_trigger": 0.0,
    }
    try:
        await websocket.send_text(
            json.dumps(
                {"status": "connected", "client_id": client_id, "state": "passive"}
            )
        )

        # ── FIX: Re-queue any reply that was generated but not delivered ─────────
        # This happens when the WebSocket broke while brain() was sending audio.
        # We inject the stored reply directly into text_queue so it gets re-spoken.
        if _pending_reply:
            await text_queue.put(f"RESEND_REPLY:{_pending_reply}")
            if _existing:
                _existing.pop("_pending_reply", None)  # consume it — don't loop
        # ─────────────────────────────────────────────────────────────────────────

        async def send_keepalive():
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)
                except WebSocketDisconnect:
                    break
                except RuntimeError:
                    break
                except Exception:
                    await asyncio.sleep(2)

        async def listener():
            oww_carry = bytearray()
            audio_buffer = bytearray()
            speech_seen = False
            followup_entered_at = (
                time.time()
            )  # safe default — prevents stale timeout on first FOLLOWUP
            previous_mode = session_state["mode"]
            bytes_received_count = 0

            while True:
                try:
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    logger.info(f"Client {client_id} disconnected normally.")
                    break
                except RuntimeError as e:
                    if 'Cannot call "receive"' in str(e):
                        logger.info(f"WebSocket {client_id} closed cleanly.")
                        break
                    logger.error(f"WebSocket RuntimeError: {e}")
                    break

                current_mode = session_state["mode"]

                if current_mode == "PASSIVE" and previous_mode != "PASSIVE":
                    oww_carry.clear()
                    audio_buffer.clear()
                    speech_seen = False
                    followup_entered_at = (
                        time.time()
                    )  # reset so next session can't inherit a stale clock
                    session_state["is_verified"] = False
                    session_state["verified_name"] = None  # <--- ADD THIS
                    session_state["claimed_name"] = None
                    session_state["visitor_reference_image_b64"] = None
                    session_state["visitor_captured"] = False  # ← ADD THIS
                    session_state["pending_identity_name"] = None
                    session_state["person_type"] = "employee"
                    session_state["mismatch_strikes"] = 0
                    session_state["conversation_complete"] = False
                    session_state["presence_count"] = 0

                    try:
                        ww_service.model.reset()
                    except Exception:
                        pass
                elif current_mode == "FOLLOWUP" and previous_mode != "FOLLOWUP":
                    audio_buffer.clear()
                    speech_seen = False
                    bytes_received_count = 0
                    logger.info(
                        f"[{client_id}] Entered FOLLOWUP/LISTENING mode. Waiting for audio..."
                    )
                    # Don't start the timeout clock while waiting for capture_reference.
                    # The camera round-trip can take seconds on slow hardware.
                    # Clock starts once capture completes and pending_text is queued.
                    if not session_state.get("awaiting_face"):
                        followup_entered_at = time.time()

                previous_mode = current_mode

                raw_bytes = message.get("bytes") or message.get("data")
                raw_text = message.get("text")

                if raw_text:
                    try:
                        msg = json.loads(raw_text)

                        if msg.get("type") == "presence_frame":
                            if session_state["mode"] != "PASSIVE":
                                continue

                            # Cooldown: skip if we triggered recently (prevent rapid re-activation)
                            PRESENCE_FRAME_COOLDOWN = float(
                                os.getenv("PRESENCE_FRAME_COOLDOWN", "9.0")
                            )
                            if (
                                time.time() - session_state["last_presence_trigger"]
                                < PRESENCE_FRAME_COOLDOWN
                            ):
                                continue

                            image_b64 = msg.get("image_b64", "")
                            if not image_b64:
                                continue

                            # Run MediaPipe detection in thread pool (non-blocking)
                            detection_service = get_person_detection_service()
                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                _face_executor,
                                lambda: detection_service.detect_person(image_b64),
                            )

                            if result["detected"]:
                                session_state["presence_count"] += 1
                                logger.info(
                                    f"[{client_id}] Person detected (count={session_state['presence_count']}, "
                                    f"confidence={result['confidence']:.2f}, face_ratio={result['face_ratio']:.3f})"
                                )

                                PRESENCE_CONFIRM_FRAMES = int(
                                    os.getenv("PRESENCE_CONFIRM_FRAMES", "3")
                                )
                                if (
                                    session_state["presence_count"]
                                    >= PRESENCE_CONFIRM_FRAMES
                                ):
                                    logger.info(
                                        f"[{client_id}] ✅ Person confirmed! Activating AIRA."
                                    )
                                    session_state["presence_count"] = 0
                                    session_state["last_presence_trigger"] = time.time()
                                    session_state["mode"] = "PROCESSING"

                                    # Notify frontend
                                    await websocket.send_text(
                                        json.dumps({"type": "person_detected"})
                                    )

                                    # Reuse existing wake-up flow
                                    await text_queue.put("WAKE_WORD_TRIGGERED")
                                    oww_carry.clear()
                                    audio_buffer.clear()
                                    speech_seen = False
                            else:
                                # No face → reset consecutive count
                                session_state["presence_count"] = 0
                            continue

                        # ── Face verification request from frontend ──────────────
                        # Triggered when the employee says their name and LLM identifies them.
                        # Frontend sends: { type: "verify_face", audio_name: "John", image_b64: "..." }
                        if msg.get("type") == "verify_face":
                            if session_state["mode"] == "PASSIVE":
                                logger.info(
                                    f"[{client_id}] Ignoring face verify — session is PASSIVE."
                                )
                                continue
                            if session_state["mode"] in (
                                "PROCESSING",
                                "SPEAKING",
                            ):  # ← ADD THIS
                                session_state["face_verify_in_progress"] = False
                                continue
                            if session_state.get("face_verify_in_progress"):
                                continue
                            if session_state.get("conversation_complete"):
                                continue
                            session_state["face_verify_in_progress"] = True

                            audio_name = msg.get("audio_name", "")
                            image_b64 = msg.get("image_b64", "")
                            # Always trust server-side session_state over frontend-sent person_type
                            # (frontend may be hardcoded to "employee")
                            person_type = session_state.get("person_type", "employee")
                            session_action = msg.get("session_action") or (
                                "capture_reference"
                                if person_type == "visitor"
                                else "compare_reference"
                            )
                            session_state["awaiting_face"] = False
                            logger.info(
                                f"[{client_id}] Face verification requested for: '{audio_name}' "
                                f"(person_type={person_type}, action={session_action})"
                            )

                            if (
                                person_type == "visitor"
                                and session_action == "capture_reference"
                            ):
                                # ── VISITOR: Use DB-backed photo deduplication ──────
                                # On first visit: saves the photo to DB
                                # On return visit: compares live frame against stored photo
                                loop = asyncio.get_running_loop()
                                result = await loop.run_in_executor(
                                    _face_executor,
                                    lambda: verify_person_face(
                                        person_type=person_type,
                                        audio_name=audio_name,
                                        image_b64=image_b64,
                                    ),
                                )

                                session_state["is_verified"] = True
                                session_state["visitor_captured"] = True
                                session_state["pending_identity_name"] = None

                                try:
                                    await websocket.send_text(
                                        json.dumps(
                                            {
                                                "type": "face_verification_result",
                                                "verified": result.get(
                                                    "verified", True
                                                ),
                                                "distance": result.get(
                                                    "distance", -1.0
                                                ),
                                                "audio_name": audio_name,
                                                "has_photo": result.get(
                                                    "has_photo", True
                                                ),
                                                "message": result.get("message", ""),
                                                "person_type": person_type,
                                                "session_action": session_action,
                                                "reference_captured": True,
                                                "is_new": result.get("is_new", True),
                                                "visitor_id": result.get("visitor_id"),
                                            }
                                        )
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"[{client_id}] Could not send visitor result. Error: {e}"
                                    )
                                    break

                                # AFTER:
                                pending_text = session_state.get("pending_text")
                                handled_by_pending = False
                                if pending_text:
                                    session_state["mode"] = "PROCESSING"
                                    await websocket.send_text(
                                        json.dumps({"state": "processing"})
                                    )
                                    await text_queue.put(pending_text)
                                    session_state["pending_text"] = None
                                    handled_by_pending = True
                                    logger.info(
                                        f"[{client_id}] Employee face match — resuming LLM response."
                                    )
                                else:
                                    session_state["mode"] = "PASSIVE"
                                    await websocket.send_text(
                                        json.dumps({"state": "passive"})
                                    )

                                session_state["face_verify_in_progress"] = False
                                # Visitor capture/compare is done — stop accepting face frames
                                session_state["conversation_complete"] = True
                                continue

                            was_already_verified = session_state.get(
                                "is_verified", False
                            )
                            loop = asyncio.get_running_loop()

                            result = await loop.run_in_executor(
                                _face_executor,
                                lambda: verify_person_face(
                                    person_type=person_type,
                                    audio_name=audio_name,
                                    image_b64=image_b64,
                                ),
                            )

                            # --- NEW: FALLBACK TO VISITOR ON GENUINE MISMATCH ---
                            if person_type == "employee" and not result.get("verified"):
                                # If a face was detected and a photo exists, but it didn't match -> It's a visitor with the same name!
                                if result.get("has_photo") and result.get(
                                    "face_detected", True
                                ):
                                    logger.info(
                                        f"[{client_id}] Employee face mismatch for '{audio_name}'. Falling back to visitor flow."
                                    )

                                    session_state["person_type"] = "visitor"
                                    person_type = "visitor"
                                    session_action = "capture_reference"

                                    # Re-run verification as a visitor to save their photo and log the visit
                                    result = await loop.run_in_executor(
                                        _face_executor,
                                        lambda: verify_person_face(
                                            person_type="visitor",
                                            audio_name=audio_name,
                                            image_b64=image_b64,
                                        ),
                                    )

                                    # Process them exactly like a successful visitor capture
                                    session_state["is_verified"] = True
                                    session_state["visitor_captured"] = True
                                    session_state["pending_identity_name"] = None
                                    session_state["face_verify_in_progress"] = False
                                    session_state["conversation_complete"] = True

                                    # ✅ Sync mismatch to query_router (flips is_employee to False)
                                    mark_employee_from_face_result(client_id, False)

                                    try:
                                        await websocket.send_text(
                                            json.dumps(
                                                {
                                                    "type": "face_verification_result",
                                                    "verified": result.get(
                                                        "verified", True
                                                    ),
                                                    "distance": result.get(
                                                        "distance", -1.0
                                                    ),
                                                    "audio_name": audio_name,
                                                    "has_photo": result.get(
                                                        "has_photo", True
                                                    ),
                                                    "message": result.get(
                                                        "message", ""
                                                    ),
                                                    "person_type": "visitor",
                                                    "session_action": "capture_reference",
                                                    "reference_captured": True,
                                                    "is_new": result.get(
                                                        "is_new", True
                                                    ),
                                                    "visitor_id": result.get(
                                                        "visitor_id"
                                                    ),
                                                }
                                            )
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"[{client_id}] Could not send visitor fallback result. Error: {e}"
                                        )
                                        break

                                    pending_text = session_state.get("pending_text")
                                    if pending_text:
                                        session_state["mode"] = "PROCESSING"
                                        await websocket.send_text(
                                            json.dumps({"state": "processing"})
                                        )
                                        await text_queue.put(pending_text)
                                        session_state["pending_text"] = None
                                    else:
                                        session_state["mode"] = "PASSIVE"
                                        await websocket.send_text(
                                            json.dumps({"state": "passive"})
                                        )

                                    continue  # Skip the rest of the employee strike logic
                            # ----------------------------------------------------

                            session_state["face_verify_in_progress"] = False
                            if result.get("verified"):
                                session_state["is_verified"] = True
                                session_state["verified_name"] = (
                                    audio_name  # <--- ADD THIS
                                )
                                session_state["pending_identity_name"] = None

                                # ✅ Sync to query_router session state
                                mark_employee_from_face_result(
                                    client_id,
                                    True,
                                    employee_id=result.get("employee_id"),
                                    employee_name=audio_name,
                                )

                                # Face verified — restart the FOLLOWUP timeout clock now
                                # (it was parked at +3600 while we waited for the camera frame)
                                followup_entered_at = time.time()

                                # ✅ AFTER face verification: Trigger the LLM response if we had original text queued
                                pending_text = session_state.get("pending_text")
                                if pending_text:
                                    session_state["mode"] = "PROCESSING"
                                    await websocket.send_text(
                                        json.dumps({"state": "processing"})
                                    )
                                    await text_queue.put(pending_text)
                                    session_state["pending_text"] = None
                                    logger.info(
                                        f"[{client_id}] Employee face match — resuming LLM response."
                                    )

                            elif person_type == "employee":
                                result["message"] = (
                                    f"I wasn't able to verify you as {audio_name} from our employee records. "
                                    f"I'll check you in as a visitor — could you let me know who you're here to meet?"
                                )
                                session_state["is_verified"] = False

                                # ✅ Sync mismatch to query_router (flips is_employee to False)
                                mark_employee_from_face_result(client_id, False)

                                # Now flip this session to visitor mode so the visitor flow takes over
                                session_state["person_type"] = "visitor"
                                session_state["pending_identity_name"] = None
                                session_state["claimed_name"] = (
                                    audio_name  # <--- ADD THIS   # stop re-asking for face
                                )
                                session_state["visitor_captured"] = (
                                    False  # allow visitor photo capture
                                )
                                followup_entered_at = time.time()
                                logger.info(
                                    f"[{client_id}] Employee face mismatch for '{audio_name}' — "
                                    f"switching to visitor flow."
                                )

                            logger.info(
                                f"[{client_id}] Face verify result for '{audio_name}': "
                                f"verified={result['verified']}, distance={result['distance']}"
                            )

                            # Send result back to frontend (for the UI badge)
                            # --- CALCULATE STRIKES & FRONTEND STATE BEFORE SENDING ---
                            frontend_verified = result["verified"]
                            speak_message = False

                            if result["has_photo"] and result["message"]:
                                if result["verified"]:
                                    # ── Successful match ──────────────────────
                                    session_state["mismatch_strikes"] = 0
                                    if not speech_seen:
                                        followup_entered_at = time.time()
                                        audio_buffer.clear()
                                    if (
                                        not was_already_verified
                                        and not handled_by_pending
                                    ):
                                        logger.info(
                                            f"[{client_id}] Initial face match — queueing confirmation."
                                        )
                                        speak_message = True

                                else:
                                    # ── Real mismatch OR No face in frame (person ducked out) ──
                                    reason = (
                                        "No face detected"
                                        if not result.get("face_detected", True)
                                        else "Mismatch"
                                    )

                                    if not was_already_verified:
                                        # INITIAL VERIFICATION: Reject immediately and speak!
                                        logger.info(
                                            f"[{client_id}] Initial {reason} — rejecting immediately."
                                        )
                                        speak_message = True
                                        session_state["mismatch_strikes"] = 0
                                        session_state["is_verified"] = False
                                        frontend_verified = False
                                    else:
                                        # CONTINUOUS VERIFICATION: Use strike/debounce logic
                                        session_state["mismatch_strikes"] += 1
                                        strikes = session_state["mismatch_strikes"]

                                        if strikes >= MAX_MISMATCH_STRIKES:
                                            logger.info(
                                                f"[{client_id}] {reason} Strike {strikes} — Revoking verification."
                                            )
                                            speak_message = True
                                            session_state["mismatch_strikes"] = 0
                                            session_state["is_verified"] = False
                                            frontend_verified = False
                                        else:
                                            logger.info(
                                                f"[{client_id}] {reason} Strike {strikes} — Debouncing. Hiding from frontend."
                                            )
                                            frontend_verified = (
                                                True  # Keep UI green during debounce
                                            )
                                            if not speech_seen:
                                                followup_entered_at = time.time()
                                                audio_buffer.clear()
                            # --- NOW SEND TO FRONTEND ---
                            try:
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "face_verification_result",
                                            "verified": frontend_verified,  # <-- Uses the masked value
                                            "distance": result["distance"],
                                            "audio_name": audio_name,
                                            "has_photo": result["has_photo"],
                                            "message": result.get("message", ""),
                                            "person_type": person_type,
                                            "session_action": session_action,
                                        }
                                    )
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[{client_id}] Could not send face result. Error: {e}"
                                )
                                break

                            # --- HANDLE AI SPEECH & UI STATE ---
                            if speak_message:
                                # Add SYSTEM: prefix to bypass the LLM
                                await text_queue.put(f"SYSTEM:{result['message']}")
                                session_state["mode"] = "FOLLOWUP"
                                await websocket.send_text(
                                    json.dumps({"state": "listening"})
                                )
                            elif was_already_verified:
                                # If it was a silent debounce, OR a successful continuous match, keep listening
                                session_state["mode"] = "FOLLOWUP"
                                await websocket.send_text(
                                    json.dumps({"state": "listening"})
                                )

                                # --------------------------
                        # ── Stop-speaking control message ────────────────────────
                        if msg.get("action") == "stop_speaking":
                            session_state["mode"] = "PASSIVE"
                            session_state["awaiting_face"] = False
                            session_state["visitor_reference_image_b64"] = None
                            session_state["pending_identity_name"] = None
                            await websocket.send_text(json.dumps({"state": "passive"}))
                    except Exception:
                        pass
                    continue

                if not raw_bytes:
                    continue

                bytes_received_count += 1
                if bytes_received_count == 1:
                    logger.info(
                        f"[{client_id}] Successfully receiving audio stream from frontend (Mode: {current_mode})"
                    )

                if session_state["mode"] in ("PROCESSING", "SPEAKING"):
                    continue
                if session_state.get("face_verify_in_progress") or session_state.get(
                    "awaiting_face"
                ):
                    audio_buffer.clear()
                    speech_seen = False
                    continue

                if session_state["mode"] == "PASSIVE":
                    oww_carry.extend(raw_bytes)
                    while len(oww_carry) >= OWW_CHUNK_BYTES:
                        chunk = bytes(oww_carry[:OWW_CHUNK_BYTES])
                        oww_carry = oww_carry[OWW_CHUNK_BYTES:]
                        triggered, score = ww_service.process_chunk(chunk)
                        if triggered:
                            logger.info(
                                f"[{client_id}] Wake word triggered! Score: {score}"
                            )
                            session_state["mode"] = "PROCESSING"
                            await text_queue.put("WAKE_WORD_TRIGGERED")
                            oww_carry.clear()
                            audio_buffer.clear()
                            speech_seen = False
                            break

                elif session_state["mode"] in ("ACTIVE", "FOLLOWUP"):
                    # Time out back to PASSIVE if they don't say anything
                    # Time out back to PASSIVE if they don't say anything
                    if (
                        current_mode == "FOLLOWUP"
                        and not speech_seen
                        and not session_state.get("brain_is_thinking")
                        and (
                            time.time() - followup_entered_at > FOLLOWUP_TIMEOUT_SECONDS
                        )
                    ):
                        # --- NEW: Prevent timeout if waiting for Slack ---
                        from services.query_router import (
                            get_session_state as _get_router_state,
                        )

                        r_state = _get_router_state(client_id)
                        if r_state.get("awaiting_slack_reply"):
                            # Keep waiting. Reset the clock so we don't loop tightly.
                            followup_entered_at = time.time()
                            continue
                        # -------------------------------------------------

                        logger.info(
                            f"[{client_id}] Followup timeout reached (no speech detected). Returning to PASSIVE."
                        )
                        session_state["conversation_complete"] = True
                        session_state["mode"] = "PASSIVE"
                        session_state["awaiting_face"] = False
                        await websocket.send_text(json.dumps({"state": "passive"}))
                        continue

                    audio_buffer.extend(raw_bytes)
                    audio_np = (
                        np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    if len(audio_np) >= SILERO_WINDOW_SAMPLES:
                        speech = get_speech_timestamps(
                            audio_np,
                            vad_model,
                            sampling_rate=SAMPLE_RATE,
                            min_speech_duration_ms=200,
                            min_silence_duration_ms=250,
                            return_seconds=False,
                        )

                        if speech:
                            if not speech_seen:
                                logger.info(f"[{client_id}] VAD detected speech start.")
                                speech_seen = True

                            last_speech_end = speech[-1]["end"]
                            total_samples = len(audio_np)
                            current_silence_ms = (
                                (total_samples - last_speech_end) / SAMPLE_RATE
                            ) * 1000

                            if current_silence_ms >= MAX_SILENCE_MS:
                                logger.info(
                                    f"[{client_id}] VAD detected speech end (User paused for {current_silence_ms:.0f}ms). Processing..."
                                )
                                session_state["mode"] = "PROCESSING"
                                await websocket.send_text(
                                    json.dumps({"state": "processing"})
                                )
                                wav_bytes = create_wav_from_pcm(bytes(audio_buffer))
                                text = await whisper_processor.transcribe_audio(
                                    wav_bytes
                                )

                                logger.info(
                                    f"[{client_id}] Whisper transcribed: '{text}'"
                                )

                                if not text or text in ("NOISE_DETECTED", "NO_SPEECH"):
                                    logger.info(
                                        f"[{client_id}] Ignored noise/silence. Returning to FOLLOWUP."
                                    )
                                    session_state["mode"] = "FOLLOWUP"

                                    # --- ADD THESE 3 LINES ---
                                    audio_buffer.clear()
                                    speech_seen = False
                                    followup_entered_at = time.time()
                                    # -------------------------

                                    await websocket.send_text(
                                        json.dumps({"state": "listening"})
                                    )
                                    continue

                                if text and text not in ("NOISE_DETECTED", "NO_SPEECH"):
                                    text_lower = text.lower()
                                    if any(
                                        w in text_lower
                                        for w in [
                                            "thank you",
                                            "thanks",
                                            "bye",
                                            "goodbye",
                                            "that's all",
                                        ]
                                    ):
                                        session_state["conversation_complete"] = True
                                        await text_queue.put(text)
                                        continue

                                    employee_name = None

                                    if not session_state.get(
                                        "is_verified"
                                    ) and not session_state.get("visitor_captured"):
                                        known_name = session_state.get("verified_name")
                                        if known_name:
                                            employee_name = known_name
                                            logger.info(
                                                f"[{client_id}] Face tracking was lost. Re-verifying known user: '{employee_name}'"
                                            )
                                        else:
                                            candidates = (
                                                _candidate_names_from_transcript(text)
                                            )
                                            loop = asyncio.get_running_loop()
                                            employee_name = None

                                            # 1. ALWAYS check the DB first for any extracted name
                                            for candidate in candidates:
                                                found_name = await loop.run_in_executor(
                                                    _face_executor,
                                                    _resolve_employee_name,
                                                    candidate,
                                                )
                                                if found_name:
                                                    employee_name = found_name
                                                    session_state["person_type"] = (
                                                        "employee"
                                                    )
                                                    logger.info(
                                                        f"[{client_id}] Name '{found_name}' found in DB. Categorizing as employee for face check."
                                                    )
                                                    break

                                            # 2. If NOT found in DB, treat them as a visitor
                                            if not employee_name:
                                                from services.query_router import (
                                                    get_session_state as _get_router_state,
                                                )

                                                router_state = _get_router_state(
                                                    client_id
                                                )
                                                known_visitor = router_state.get(
                                                    "visitor_name"
                                                )

                                                if known_visitor:
                                                    employee_name = known_visitor
                                                    session_state["person_type"] = (
                                                        "visitor"
                                                    )
                                                    logger.info(
                                                        f"[{client_id}] Visitor already known as '{known_visitor}' from router state."
                                                    )
                                                elif candidates:
                                                    employee_name = candidates[0]
                                                    session_state["person_type"] = (
                                                        "visitor"
                                                    )
                                                    logger.info(
                                                        f"[{client_id}] Name '{employee_name}' not in DB. Categorizing as visitor."
                                                    )

                                    # If we found a name, trigger face capture/verification
                                    if employee_name:
                                        person_type = session_state["person_type"]
                                        logger.info(
                                            f"[{client_id}] identity_detected emitted: "
                                            f"'{employee_name}' (person_type={person_type})"
                                        )
                                        session_state["is_verified"] = False
                                        session_state["awaiting_face"] = True
                                        session_state["pending_identity_name"] = (
                                            employee_name
                                        )
                                        session_state["pending_text"] = text

                                        # Switch to FOLLOWUP so face frames are not blocked
                                        # by the PROCESSING guard, and clear the audio buffer
                                        # so the same utterance is not re-processed in a loop.
                                        # NOTE: followup_entered_at is NOT set here.
                                        # awaiting_face=True means the FOLLOWUP transition guard
                                        # will defer the clock until capture_reference completes.
                                        session_state["mode"] = "FOLLOWUP"
                                        audio_buffer.clear()
                                        speech_seen = False
                                        # Park the timeout clock far in the future so the
                                        # FOLLOWUP timeout check cannot fire while we are
                                        # waiting for the camera frame from the frontend.
                                        # The clock is restarted in two places:
                                        #   • capture_reference handler (visitor path)
                                        #   • after face_verify_in_progress clears (employee path)
                                        followup_entered_at = time.time() + 3600
                                        await websocket.send_text(
                                            json.dumps({"state": "listening"})
                                        )

                                        try:
                                            await websocket.send_text(
                                                json.dumps(
                                                    {
                                                        "type": "employee_identified",
                                                        "name": employee_name,
                                                        "person_type": person_type,
                                                        "session_action": (
                                                            "capture_reference"
                                                            if person_type == "visitor"
                                                            else "compare_reference"
                                                        ),
                                                    }
                                                )
                                            )
                                        except WebSocketDisconnect:
                                            logger.warning(
                                                f"[{client_id}] Client disconnected before face request could be sent."
                                            )
                                            break
                                        except Exception as e:
                                            logger.error(
                                                f"[{client_id}] Failed to send face request: {e}"
                                            )
                                    elif session_state.get("pending_identity_name"):
                                        # Add SYSTEM: prefix to bypass the LLM
                                        await text_queue.put(
                                            "SYSTEM:Please complete face verification before continuing."
                                        )
                                    else:
                                        # Already verified OR no name found — chat normally
                                        await text_queue.put(text)

        async def brain():
            while True:
                text = await text_queue.get()

                # --- 1. DETECT TERMINAL STATE ---
                is_terminal = False
                if (
                    text
                    and text
                    not in (
                        "WAKE_WORD_TRIGGERED",
                        "SYSTEM:Please complete face verification before continuing.",
                    )
                    and not text.startswith("SYSTEM:")
                    and not text.startswith("SLACK_REPLY:")
                    and not text.startswith("RESEND_REPLY:")
                ):
                    text_lower = text.lower()
                    if any(
                        w in text_lower
                        for w in ["thank you", "thanks", "bye", "goodbye", "that's all"]
                    ):
                        is_terminal = True
                        session_state["conversation_complete"] = True
                # --------------------------------

                manager.client_state[client_id] = "THINKING"
                session_state["brain_is_thinking"] = True

                # --- THE TRY BLOCK NOW WRAPS EVERYTHING (FIX #2) ---
                # Previously the try only wrapped TTS/audio sending.
                # If process_text_for_client raised on the SLACK_REPLY path,
                # task_done() was never called and the queue jammed permanently.
                try:
                    # --- RESEND: re-deliver a reply that was dropped due to disconnect ---
                    if text and text.startswith("RESEND_REPLY:"):
                        reply_text = text.split("RESEND_REPLY:", 1)[1]
                        session_state["mode"] = "FOLLOWUP"
                        logger.info(f"[{client_id}] Re-speaking undelivered reply.")

                    # --- BYPASS LLM FOR SLACK REPLY SENTINELS ---
                    elif text and text.startswith("SLACK_REPLY:"):
                        # Force mode back to FOLLOWUP so the session is "alive"
                        # when brain() tries to send audio back to the user.
                        session_state["mode"] = "FOLLOWUP"
                        reply_text = await process_text_for_client(client_id, text)

                    # --- BYPASS LLM FOR SYSTEM MESSAGES ---
                    elif text and text.startswith("SYSTEM:"):
                        reply_text = text.split("SYSTEM:", 1)[1]

                    else:
                        # --- INJECT USER IDENTITY INTO LLM PROMPT ---
                        verified_name = session_state.get("verified_name")
                        if verified_name:
                            prompt_text = f"[System Note: The user speaking is verified as {verified_name}] {text}"
                        else:
                            prompt_text = text

                        reply_text = await process_text_for_client(
                            client_id, prompt_text
                        )

                    logger.info(f"[{client_id}] AI Response generated: '{reply_text}'")

                    if not reply_text or not reply_text.strip():
                        session_state["mode"] = "PASSIVE"
                        await websocket.send_text(json.dumps({"state": "passive"}))
                        continue

                    # ── FIX: Save reply into router state BEFORE attempting to send
                    # audio. If the WebSocket dies mid-send, the reconnect handler
                    # will find this and re-inject it via RESEND_REPLY so the user
                    # actually hears what Jarvis wanted to say.
                    try:
                        from services.query_router import get_session_state as _gss

                        _router_state = _gss(client_id)
                        _router_state["_pending_reply"] = reply_text
                    except Exception:
                        pass
                    # ────────────────────────────────────────────────────────────

                    audio, word_timings = (
                        await tts_processor.synthesize_remaining_speech_with_timing(
                            reply_text
                        )
                    )
                    if audio is not None and len(audio) > 0:
                        session_state["mode"] = "SPEAKING"
                        manager.client_state[client_id] = "SPEAKING"
                        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                        wav_bytes = create_wav_from_pcm(audio_bytes, sample_rate=24000)
                        b64 = base64.b64encode(wav_bytes).decode("utf-8")
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "audio": b64,
                                    "word_timings": word_timings,
                                    "sample_rate": 24000,
                                    "method": "native_kokoro_timing",
                                    "state": "speaking",
                                }
                            )
                        )

                        # Wait for audio to physically finish playing on the frontend
                        await asyncio.sleep((len(audio) / 24000.0) + 0.5)

                    # ── FIX: Audio delivered successfully — clear the pending reply ─
                    try:
                        from services.query_router import get_session_state as _gss

                        _router_state = _gss(client_id)
                        _router_state.pop("_pending_reply", None)
                    except Exception:
                        pass
                    # ────────────────────────────────────────────────────────────────

                    # --- 2. ROUTE TO CORRECT MODE ---
                    if is_terminal:
                        logger.info(
                            f"[{client_id}] Terminal phrase detected. Returning to PASSIVE mode."
                        )
                        session_state["mode"] = "PASSIVE"
                        session_state["awaiting_face"] = False
                        await websocket.send_text(json.dumps({"state": "passive"}))
                    else:
                        # Give the user 12 seconds to reply after Jarvis speaks
                        session_state["mode"] = "FOLLOWUP"
                        await websocket.send_text(json.dumps({"state": "listening"}))
                    # --------------------------------

                except (RuntimeError, WebSocketDisconnect):
                    logger.info(
                        f"[{client_id}] Client disconnected before AI could finish responding."
                    )
                    # _pending_reply is intentionally NOT cleared here — it will be
                    # re-delivered on the next reconnect via RESEND_REPLY.
                    break
                finally:
                    session_state["brain_is_thinking"] = False
                    # task_done() now ALWAYS runs — no matter which branch above
                    # was taken, and no matter whether an exception was raised.
                    text_queue.task_done()

        async def slack_watcher():
            """
            Polls Slack's API directly every 3 seconds for a reply in the
            thread that was opened for this session.
            No webhook, no tunnel, no Events API required.
            """
            from services.slack_reply_poller import poll_for_reply
            from services.query_router import get_session_state as _get_state

            POLL_INTERVAL = 3
            MAX_WAIT = 120

            logger.info(f"[{client_id}] slack_watcher STARTED")
            elapsed = 0

            while True:
                await asyncio.sleep(POLL_INTERVAL)

                try:
                    state = _get_state(client_id)
                except Exception as e:
                    logger.warning(
                        f"[{client_id}] slack_watcher: session gone ({e}), stopping."
                    )
                    break

                if not state.get("awaiting_slack_reply"):
                    continue

                elapsed += POLL_INTERVAL

                if elapsed >= MAX_WAIT:
                    logger.info(
                        f"[{client_id}] slack_watcher: no reply after {MAX_WAIT}s, giving up."
                    )
                    state["awaiting_slack_reply"] = False
                    elapsed = 0
                    continue

                session_id = state.get("session_id")
                if not session_id:
                    continue

                # Poll Slack API directly — no webhook needed
                result = await poll_for_reply(session_id)
                if not result:
                    continue

                sender_name, reply_text = result
                elapsed = 0

                logger.info(
                    f"[{client_id}] slack_watcher REPLY from '{sender_name}': {reply_text}"
                )

                # Interrupt frontend and inject into brain()
                await websocket.send_json({"type": "interrupt_listening"})
                await text_queue.put(f"SLACK_REPLY:{sender_name}:{reply_text}")

            logger.info(f"[{client_id}] slack_watcher exiting.")

        io_tasks = [
            asyncio.create_task(listener()),
            asyncio.create_task(send_keepalive()),
        ]
        brain_task = asyncio.create_task(brain())
        watcher_task = asyncio.create_task(slack_watcher())

        await asyncio.wait(io_tasks, return_when=asyncio.FIRST_COMPLETED)

        for t in io_tasks:
            t.cancel()

        from services.query_router import get_session_state as _qs

        try:
            _state = _qs(client_id)
            still_waiting = _state.get("awaiting_slack_reply", False)
        except Exception:
            still_waiting = False

        if still_waiting and not brain_task.done() and not watcher_task.done():
            await asyncio.wait(
                [brain_task, watcher_task],
                return_when=asyncio.ALL_COMPLETED,
                timeout=120,
            )

        # ✅ Only cancel watcher if NOT still waiting for Slack reply
        try:
            await asyncio.wait_for(text_queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        brain_task.cancel()

        try:
            _state = _qs(client_id)
            if not _state.get("awaiting_slack_reply"):
                watcher_task.cancel()
            else:
                logger.info(
                    f"[{client_id}] Keeping watcher alive — still awaiting Slack reply."
                )
        except Exception:
            watcher_task.cancel()

    finally:
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)
