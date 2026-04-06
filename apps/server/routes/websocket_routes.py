import asyncio
import base64
import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from managers.connection_manager import manager
from models.whisper_processor import WhisperProcessor
from models.tts_processor import KokoroTTSProcessor
from services.query_router import route_query, clear_session_state

logger = logging.getLogger(__name__)

router = APIRouter()

COMPANY_NAME = "Sharp Software Development India Pvt Ltd"

WELCOME_MESSAGE = (
    f"Welcome to {COMPANY_NAME}! "
    "I'm AlmostHuman, your virtual receptionist. "
    "How can I help you today?"
)


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time Receptionist AI interaction (audio-only)."""
    await manager.connect(websocket, client_id)

    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    # Per-connection queue: STT text → brain
    text_queue: asyncio.Queue[str] = asyncio.Queue()

    # Tracks whether this session has been welcomed yet.
    # Set to True only when wake_word_detected arrives from the frontend.
    session_started = False

    try:
        # ── Confirm connection (silent — no welcome yet) ───────────────────
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

        # ── Keepalive ─────────────────────────────────────────────────────
        async def send_keepalive():
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)
                except Exception:
                    break

        # ── Listener: audio → STT → queue ────────────────────────────────
        async def listener():
            nonlocal session_started
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)

                        # ── Wake word detected → greet and open mic ───────
                        # Your wake word library (e.g. Porcupine / openWakeWord)
                        # should send:  { "wake_word_detected": true }
                        # when it hears the trigger phrase.
                        if message.get("wake_word_detected"):
                            if not session_started:
                                session_started = True
                                logger.info(f"[{client_id}] Wake word detected — greeting visitor")
                                await _send_tts_response(
                                    websocket, tts_processor, WELCOME_MESSAGE, client_id
                                )
                            else:
                                # Already in a session — wake word just re-opens the mic,
                                # no need to re-greet.
                                logger.info(f"[{client_id}] Wake word re-detected mid-session")
                            continue

                        # ── Audio segment — only process after session starts ──
                        if "audio_segment" in message:
                            if not session_started:
                                # Ignore audio before wake word fires
                                continue

                            audio_data = base64.b64decode(message["audio_segment"])
                            logger.info(f"Audio segment received: {len(audio_data)} bytes")

                            transcribed_text = await whisper_processor.transcribe_audio(audio_data)
                            logger.info(f"STT result: '{transcribed_text}'")

                            if transcribed_text in ("NOISE_DETECTED", "NO_SPEECH", None, ""):
                                continue

                            await text_queue.put(transcribed_text)

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                    except Exception as e:
                        logger.error(f"Listener error: {e}")
                        await websocket.send_text(json.dumps({"error": str(e)}))

            except WebSocketDisconnect:
                logger.info("WebSocket closed during listener")

        # ── Brain: text → route_query → TTS → send audio ─────────────────
        async def brain():
            try:
                while True:
                    text = await text_queue.get()
                    manager.client_state[client_id] = "THINKING"

                    # Route query through intent detection + DB grounding + LLM
                    reply_text = await process_text_for_client(client_id, text)
                    logger.info(f"Grounded reply: '{reply_text}'")

                    # Skip synthesis if there's nothing meaningful to say
                    if not reply_text or not reply_text.strip():
                        text_queue.task_done()
                        manager.client_state[client_id] = "WAITING_FOR_PLAYBACK"
                        continue

                    # Synthesize speech with Kokoro TTS (non-blocking)
                    audio, word_timings = await tts_processor.synthesize_remaining_speech_with_timing(  # type: ignore[attr-defined]
                        reply_text
                    )

                    if audio is not None and len(audio) > 0:
                        import numpy as np  # Local import to avoid unused at module level

                        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

                        audio_message = {
                            "audio": base64_audio,
                            "word_timings": word_timings,
                            "sample_rate": 24000,
                            "method": "native_kokoro_timing",
                            "modality": "audio_only",
                        }
                        manager.client_state[client_id] = "SPEAKING"
                        await websocket.send_text(json.dumps(audio_message))

                    # Mark queue task done and return to waiting for next utterance
                    text_queue.task_done()
                    manager.client_state[client_id] = "WAITING_FOR_PLAYBACK"

            except WebSocketDisconnect:
                logger.info("WebSocket closed during brain loop")
            except Exception as e:
                logger.error(f"Brain task error for {client_id}: {e}")

        # ── Run all three concurrently ────────────────────────────────────
        listener_task = asyncio.create_task(listener())
        brain_task = asyncio.create_task(brain())
        keepalive_task = asyncio.create_task(send_keepalive())

        manager.current_tasks[client_id]["processing"] = brain_task

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
            except Exception as e:
                logger.error(f"Task ended with error: {e}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket session error for {client_id}: {e}")
    finally:
        logger.info(f"Cleaning up for {client_id}")
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)