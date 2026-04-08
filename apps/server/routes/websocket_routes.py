import asyncio
import base64
import json
import logging
import time
import io
import os
import wave
import webrtcvad
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from managers.connection_manager import manager
from models.whisper_processor import WhisperProcessor
from models.tts_processor import KokoroTTSProcessor
from services.processor_service import process_text_for_client

from services.wake_word_service import get_wake_word_service

# ── Force DEBUG level ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

router = APIRouter()

vad = webrtcvad.Vad(3)
WAKE_WORD = os.getenv("OPENWAKEWORD_WAKEWORD", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.getenv("OPENWAKEWORD_THRESHOLD", "0.35"))
ww_service = get_wake_word_service(WAKE_WORD)
ww_service.threshold = WAKE_WORD_THRESHOLD
logger.info(
    "[OWW] Wake word configured: '%s' (threshold=%.3f)",
    WAKE_WORD,
    WAKE_WORD_THRESHOLD,
)

SAMPLE_RATE = 16_000
VAD_FRAME_BYTES = 960
OWW_CHUNK_SAMPLES = 1280
OWW_CHUNK_BYTES = OWW_CHUNK_SAMPLES * 2
SILENCE_FRAME_THRESHOLD = 50
OWW_SCORE_LOG_MIN = 0.05

FOLLOWUP_TIMEOUT_SECONDS = float(os.getenv("FOLLOWUP_TIMEOUT", "12.0"))


def create_wav_from_pcm(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return wav_io.getvalue()


def _reply_expects_followup(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if t.endswith("?"):
        return True

    followup_phrases = (
        "your name",
        "who are you",
        "who you",
        "tell me",
        "could you",
        "can you",
        "would you",
        "what is",
        "what's",
        "what time",
        "what date",
        "which day",
        "purpose of",
        "reason for",
        "who would you",
        "who do you",
        "shall i",
        "should i",
        "go ahead",
        "confirm it",
        "let me know",
        "anything else",
        "help you with",
        "welcome to sharp software",  # Added greeting to followup triggers
    )
    lower = t.lower()
    return any(lower.endswith(p) or p in lower[-60:] for p in followup_phrases)


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    text_queue: asyncio.Queue[str] = asyncio.Queue()
    session_state = {"mode": "PASSIVE"}

    DEBUG_INTERVAL = 5.0
    dbg = {
        "packets": 0,
        "oww_chunks": 0,
        "oww_max_score": 0.0,
        "vad_speech": 0,
        "vad_silence": 0,
        "last_report": time.time(),
        "last_heartbeat": time.time(),
        "total_messages": 0,
    }

    try:
        await websocket.send_text(
            json.dumps(
                {"status": "connected", "client_id": client_id, "state": "passive"}
            )
        )
        logger.info(f"[SESSION] Client {client_id} connected.")

        async def send_keepalive():
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)
                except Exception:
                    break

        async def listener():
            oww_carry, vad_carry, audio_buffer = bytearray(), bytearray(), bytearray()
            silence_frames, speech_seen, first_audio = 0, False, False
            followup_entered_at: float = 0.0
            previous_mode = session_state["mode"]

            try:
                while True:
                    message = await websocket.receive()
                    current_mode = session_state["mode"]

                    if current_mode == "PASSIVE" and previous_mode != "PASSIVE":
                        oww_carry.clear()
                        vad_carry.clear()
                        audio_buffer.clear()
                        silence_frames = 0
                        speech_seen = False
                        try:
                            ww_service.model.reset()
                        except:
                            pass
                    elif current_mode == "FOLLOWUP" and previous_mode != "FOLLOWUP":
                        vad_carry.clear()
                        audio_buffer.clear()
                        silence_frames = 0
                        speech_seen = False
                        followup_entered_at = time.time()

                    previous_mode = current_mode
                    raw_bytes = message.get("bytes") or message.get("data")
                    raw_text = message.get("text")

                    if raw_text:
                        try:
                            msg = json.loads(raw_text)
                            if msg.get("action") == "stop_speaking":
                                session_state["mode"] = "PASSIVE"
                                await websocket.send_text(
                                    json.dumps({"state": "passive"})
                                )
                        except:
                            pass
                        continue

                    if raw_bytes:
                        if session_state["mode"] in ("PROCESSING", "SPEAKING"):
                            continue

                        if session_state["mode"] == "PASSIVE":
                            oww_carry.extend(raw_bytes)
                            while len(oww_carry) >= OWW_CHUNK_BYTES:
                                chunk = bytes(oww_carry[:OWW_CHUNK_BYTES])
                                oww_carry = oww_carry[OWW_CHUNK_BYTES:]
                                triggered, score = ww_service.process_chunk(chunk)
                                if triggered:
                                    # ── GREETING FIRST LOGIC ──
                                    logger.info(
                                        f"[OWW] 🔔 TRIGGERED! Injecting Greeting."
                                    )
                                    session_state["mode"] = "PROCESSING"
                                    await text_queue.put("WAKE_WORD_TRIGGERED")
                                    oww_carry.clear()
                                    audio_buffer.clear()
                                    vad_carry.clear()
                                    break

                        elif session_state["mode"] in ("ACTIVE", "FOLLOWUP"):
                            if current_mode == "FOLLOWUP" and (
                                time.time() - followup_entered_at
                                > FOLLOWUP_TIMEOUT_SECONDS
                            ):
                                session_state["mode"] = "PASSIVE"
                                await websocket.send_text(
                                    json.dumps({"state": "passive"})
                                )
                                continue

                            audio_buffer.extend(raw_bytes)
                            vad_carry.extend(raw_bytes)
                            while len(vad_carry) >= VAD_FRAME_BYTES:
                                frame = bytes(vad_carry[:VAD_FRAME_BYTES])
                                vad_carry = vad_carry[VAD_FRAME_BYTES:]
                                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                                if is_speech:
                                    speech_seen = True
                                    silence_frames = 0
                                else:
                                    silence_frames += 1

                            if (
                                speech_seen
                                and silence_frames >= SILENCE_FRAME_THRESHOLD
                            ):
                                session_state["mode"] = "PROCESSING"
                                await websocket.send_text(
                                    json.dumps({"state": "processing"})
                                )
                                wav_bytes = create_wav_from_pcm(bytes(audio_buffer))
                                transcribed_text = (
                                    await whisper_processor.transcribe_audio(wav_bytes)
                                )
                                if transcribed_text and transcribed_text not in (
                                    "NOISE_DETECTED",
                                    "NO_SPEECH",
                                ):
                                    await text_queue.put(transcribed_text)
                                else:
                                    session_state["mode"] = "PASSIVE"
                                    await websocket.send_text(
                                        json.dumps({"state": "passive"})
                                    )
                                audio_buffer.clear()
                                vad_carry.clear()
                                silence_frames = 0
                                speech_seen = False

            except Exception as exc:
                logger.error(f"[LISTENER] Error: {exc}")

        async def brain():
            try:
                while True:
                    text = await text_queue.get()
                    manager.client_state[client_id] = "THINKING"

                    reply_text = await process_text_for_client(client_id, text)

                    if not reply_text or not reply_text.strip():
                        text_queue.task_done()
                        session_state["mode"] = "PASSIVE"
                        await websocket.send_text(json.dumps({"state": "passive"}))
                        continue

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
                        base64_audio = base64.b64encode(wav_bytes).decode("utf-8")

                        await websocket.send_text(
                            json.dumps(
                                {
                                    "audio": base64_audio,
                                    "word_timings": word_timings,
                                    "sample_rate": 24000,
                                    "method": "native_kokoro_timing",
                                    "state": "speaking",
                                }
                            )
                        )
                        await asyncio.sleep((len(audio) / 24000.0) + 0.5)

                    # ── STAY IN LOOP LOGIC ──
                    # If it was the initial greeting OR the AI asked a question, stay in FOLLOWUP
                    if text == "WAKE_WORD_TRIGGERED" or _reply_expects_followup(
                        reply_text
                    ):
                        session_state["mode"] = "FOLLOWUP"
                        await websocket.send_text(json.dumps({"state": "listening"}))
                        logger.info("[BRAIN] Playback complete. Staying in FOLLOWUP.")
                    else:
                        session_state["mode"] = "PASSIVE"
                        await websocket.send_text(json.dumps({"state": "passive"}))

                    text_queue.task_done()
            except Exception as exc:
                logger.error(f"[BRAIN] Error: {exc}")

        listener_task = asyncio.create_task(listener())
        brain_task = asyncio.create_task(brain())
        keepalive_task = asyncio.create_task(send_keepalive())

        await asyncio.wait(
            [listener_task, brain_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)
