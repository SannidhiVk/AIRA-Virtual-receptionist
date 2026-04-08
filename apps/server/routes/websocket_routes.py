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
from main import process_text_for_client

from services.wake_word_service import get_wake_word_service

# ── Force DEBUG level so every log.debug() line shows in the terminal ─────────
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
VAD_FRAME_BYTES = 960  # 30 ms @ 16 kHz, 16-bit mono
OWW_CHUNK_SAMPLES = 1280  # 80 ms @ 16 kHz
OWW_CHUNK_BYTES = OWW_CHUNK_SAMPLES * 2  # 2560 bytes
SILENCE_FRAME_THRESHOLD = 50  # × 30 ms = 1 500 ms of silence
OWW_SCORE_LOG_MIN = 0.05  # only print OWW scores above this


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

    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    text_queue: asyncio.Queue[str] = asyncio.Queue()
    session_state = {"mode": "PASSIVE"}

    # Rolling debug counters – reset every DEBUG_INTERVAL seconds
    DEBUG_INTERVAL = 5.0
    dbg = {
        "packets": 0,
        "oww_chunks": 0,
        "oww_max_score": 0.0,
        "vad_speech": 0,
        "vad_silence": 0,
        "last_report": time.time(),
        "last_heartbeat": time.time(),
        "total_messages": 0,  # every ws message, not just audio
    }

    try:
        await websocket.send_text(
            json.dumps(
                {"status": "connected", "client_id": client_id, "state": "passive"}
            )
        )
        logger.info(f"[SESSION] Client {client_id} connected.")

        # ── keepalive ─────────────────────────────────────────────────────────
        async def send_keepalive():
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)
                except Exception:
                    break

        # ── listener ──────────────────────────────────────────────────────────
        async def listener():
            oww_carry = bytearray()
            vad_carry = bytearray()
            audio_buffer = bytearray()
            silence_frames = 0
            speech_seen = False
            first_audio = False

            previous_mode = session_state["mode"]

            logger.info(
                f"[LISTENER] ▶️  Loop started. "
                f"OWW_CHUNK_BYTES={OWW_CHUNK_BYTES}  VAD_FRAME_BYTES={VAD_FRAME_BYTES}"
            )

            try:
                while True:
                    try:
                        message = await websocket.receive()
                    except RuntimeError as exc:
                        if "disconnect" in str(exc).lower():
                            logger.info(
                                "[LISTENER] Client disconnected (RuntimeError)."
                            )
                            break
                        raise

                    # 🚨 FIX 2: Strict State Transition Monitor
                    # If we just switched back to PASSIVE, we MUST wipe the OpenWakeWord
                    # internal memory and all buffers so it doesn't hallucinate a wake word.
                    current_mode = session_state["mode"]
                    if current_mode == "PASSIVE" and previous_mode != "PASSIVE":
                        logger.info(
                            "[LISTENER] 🧹 Transitioned to PASSIVE. Wiping buffers and OWW memory."
                        )
                        oww_carry.clear()
                        vad_carry.clear()
                        audio_buffer.clear()
                        try:
                            ww_service.model.reset()  # Wipes OWW internal state
                        except Exception as e:
                            logger.debug(f"[LISTENER] OWW reset note: {e}")
                    previous_mode = current_mode

                    dbg["total_messages"] += 1

                    _now = time.time()
                    if _now - dbg["last_heartbeat"] >= 1.0:
                        dbg["last_heartbeat"] = _now

                    if message.get("type") == "websocket.disconnect":
                        logger.info("[LISTENER] Received websocket.disconnect.")
                        break

                    raw_bytes = message.get("bytes") or message.get("data")
                    raw_text = message.get("text")

                    # ── Text / JSON control messages ───────────────────────────
                    if raw_text:
                        try:
                            msg = json.loads(raw_text)
                            if msg.get("action") == "stop_speaking":
                                session_state["mode"] = "PASSIVE"
                                await websocket.send_text(
                                    json.dumps({"state": "passive"})
                                )
                        except json.JSONDecodeError:
                            pass
                        continue

                    # ── Binary audio data ──────────────────────────────────────
                    if raw_bytes is not None and len(raw_bytes) > 0:
                        data: bytes = raw_bytes
                        dbg["packets"] += 1

                        if not first_audio:
                            first_audio = True

                        now = time.time()
                        if now - dbg["last_report"] >= DEBUG_INTERVAL:
                            dbg["oww_max_score"] = 0.0
                            dbg["vad_speech"] = 0
                            dbg["vad_silence"] = 0
                            dbg["last_report"] = now

                        # If we are thinking or speaking, THROW AWAY incoming audio.
                        if session_state["mode"] in ("PROCESSING", "SPEAKING"):
                            oww_carry.clear()
                            vad_carry.clear()
                            audio_buffer.clear()
                            continue

                        # ── PASSIVE: wake-word ─────────────────────────────────
                        if session_state["mode"] == "PASSIVE":
                            oww_carry.extend(data)

                            while len(oww_carry) >= OWW_CHUNK_BYTES:
                                chunk = bytes(oww_carry[:OWW_CHUNK_BYTES])
                                oww_carry = oww_carry[OWW_CHUNK_BYTES:]

                                triggered, score = ww_service.process_chunk(chunk)

                                dbg["oww_chunks"] += 1
                                if score > dbg["oww_max_score"]:
                                    dbg["oww_max_score"] = score

                                if score > OWW_SCORE_LOG_MIN:
                                    logger.info(
                                        f"[OWW] chunk #{dbg['oww_chunks']:05d}  "
                                        f"score={score:.4f}  "
                                        f"{'🔔 WAKE WORD!' if triggered else ''}"
                                    )

                                if triggered:
                                    session_state["mode"] = "ACTIVE"
                                    oww_carry.clear()
                                    audio_buffer.clear()
                                    vad_carry.clear()
                                    silence_frames = 0
                                    speech_seen = False
                                    logger.info(
                                        f"[OWW] 🔔 TRIGGERED at score={score:.4f} — switching to ACTIVE"
                                    )
                                    await websocket.send_text(
                                        json.dumps({"state": "listening"})
                                    )
                                    break

                        # ── ACTIVE: VAD + accumulate ───────────────────────────
                        elif session_state["mode"] == "ACTIVE":
                            audio_buffer.extend(data)
                            vad_carry.extend(data)

                            while len(vad_carry) >= VAD_FRAME_BYTES:
                                frame = bytes(vad_carry[:VAD_FRAME_BYTES])
                                vad_carry = vad_carry[VAD_FRAME_BYTES:]

                                try:
                                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                                except Exception:
                                    is_speech = False

                                if is_speech:
                                    speech_seen = True
                                    silence_frames = 0
                                    dbg["vad_speech"] += 1
                                else:
                                    silence_frames += 1
                                    dbg["vad_silence"] += 1

                            if (
                                speech_seen
                                and silence_frames >= SILENCE_FRAME_THRESHOLD
                            ):
                                session_state["mode"] = "PROCESSING"
                                dur_s = len(audio_buffer) / (SAMPLE_RATE * 2)
                                logger.info(
                                    f"[VAD] End of utterance — buffer={len(audio_buffer)} bytes ({dur_s:.2f}s)"
                                )
                                await websocket.send_text(
                                    json.dumps({"state": "processing"})
                                )

                                wav_bytes = create_wav_from_pcm(bytes(audio_buffer))
                                transcribed_text = (
                                    await whisper_processor.transcribe_audio(wav_bytes)
                                )
                                logger.info(f"[WHISPER] '{transcribed_text}'")

                                if transcribed_text and transcribed_text not in (
                                    "NOISE_DETECTED",
                                    "NO_SPEECH",
                                ):
                                    await text_queue.put(transcribed_text)
                                else:
                                    # If no speech, go back to passive. The transition monitor at the top
                                    # of the loop will catch this and wipe the OWW memory.
                                    session_state["mode"] = "PASSIVE"
                                    await websocket.send_text(
                                        json.dumps({"state": "passive"})
                                    )

                                audio_buffer.clear()
                                vad_carry.clear()
                                silence_frames = 0
                                speech_seen = False

            except WebSocketDisconnect:
                logger.info("[LISTENER] WebSocket closed.")
            except Exception as exc:
                logger.error(f"[LISTENER] Unexpected error: {exc}", exc_info=True)

        # ── brain ─────────────────────────────────────────────────────────────
        async def brain():
            try:
                while True:
                    text = await text_queue.get()
                    manager.client_state[client_id] = "THINKING"
                    logger.info(f"[BRAIN] Processing: '{text}'")

                    reply_text = await process_text_for_client(client_id, text)
                    logger.info(f"[BRAIN] Reply: '{reply_text}'")

                    if not reply_text or not reply_text.strip():
                        text_queue.task_done()
                        manager.client_state[client_id] = "WAITING_FOR_PLAYBACK"
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

                        # 🚨 FIX 1: Wrap the raw PCM in a WAV header so the frontend can actually play it!
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
                                    "modality": "audio_only",
                                    "state": "speaking",
                                }
                            )
                        )

                        # Wait for the audio to actually finish playing
                        audio_duration = len(audio) / 24000.0
                        logger.info(
                            f"[BRAIN] Speaking for {audio_duration:.2f}s. Pausing listening."
                        )

                        # Sleep for the duration of the audio + 0.5s buffer
                        await asyncio.sleep(audio_duration + 0.5)

                    text_queue.task_done()

                    if session_state["mode"] == "SPEAKING":
                        manager.client_state[client_id] = "WAITING_FOR_PLAYBACK"
                        session_state["mode"] = "PASSIVE"
                        await websocket.send_text(json.dumps({"state": "passive"}))
                        logger.info(
                            "[BRAIN] Playback complete. Back to PASSIVE (listening for wake word)."
                        )

            except WebSocketDisconnect:
                logger.info("[BRAIN] WebSocket closed.")
            except Exception as exc:
                logger.error(f"[BRAIN] Error: {exc}", exc_info=True)
                session_state["mode"] = "PASSIVE"

        # ── run tasks ─────────────────────────────────────────────────────────
        listener_task = asyncio.create_task(listener())
        brain_task = asyncio.create_task(brain())
        keepalive_task = asyncio.create_task(send_keepalive())

        manager.current_tasks[client_id]["processing"] = brain_task
        manager.current_tasks[client_id]["tts"] = None

        done, pending = await asyncio.wait(
            [listener_task, brain_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for task in done:
            try:
                task.result()
            except Exception as exc:
                logger.error(f"[SESSION] Task error: {exc}")

    except WebSocketDisconnect:
        logger.info(f"[SESSION] Client {client_id} disconnected normally.")
    except Exception as exc:
        logger.error(f"[SESSION] Error for {client_id}: {exc}")
    finally:
        logger.info(f"[SESSION] Cleaning up {client_id}.")
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)
