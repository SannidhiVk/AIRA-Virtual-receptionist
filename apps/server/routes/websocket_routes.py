import asyncio
import base64
import json
import logging
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from managers.connection_manager import manager
from models.whisper_processor import WhisperProcessor
from models.tts_processor import KokoroTTSProcessor
from services.query_router import route_query, clear_session_state

logger = logging.getLogger(__name__)

router = APIRouter()

COMPANY_NAME = "Sharp Software Development India Private Limited"

WELCOME_MESSAGE = (
    f"Welcome to {COMPANY_NAME}! "
    "I'm Sannika, your virtual receptionist. "
    "How can I help you today?"
)


# ─────────────────────────────────────────────
# TTS HELPER — must be at module level, above all callers
# ─────────────────────────────────────────────

async def _send_tts_response(
    websocket: WebSocket,
    tts_processor: KokoroTTSProcessor,
    text: str,
    client_id: str,
) -> None:
    """Synthesise text with Kokoro TTS and push audio + word timings to the client."""
    try:
        audio, word_timings = await tts_processor.synthesize_remaining_speech_with_timing(text)

        if audio is not None and len(audio) > 0:
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            await websocket.send_text(
                json.dumps({
                    "audio": base64_audio,
                    "word_timings": word_timings,
                    "sample_rate": 24000,
                    "method": "native_kokoro_timing",
                    "modality": "audio_only",
                    "text": text,
                })
            )
            manager.client_state[client_id] = "SPEAKING"
        else:
            await websocket.send_text(json.dumps({"text": text, "audio": None}))

    except Exception as e:
        logger.error(f"TTS error for {client_id}: {e}")
        try:
            await websocket.send_text(json.dumps({"text": text, "audio": None}))
        except Exception:
            pass


# ─────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time Receptionist AI interaction (audio-only)."""
    await manager.connect(websocket, client_id)

    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    text_queue: asyncio.Queue[str] = asyncio.Queue()
    session_started = False

    try:
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

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
            nonlocal session_started
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)

                        if message.get("wake_word_detected"):
                            if not session_started:
                                session_started = True
                                logger.info(f"[{client_id}] Wake word detected — greeting visitor")
                                await _send_tts_response(
                                    websocket, tts_processor, WELCOME_MESSAGE, client_id
                                )
                            else:
                                logger.info(f"[{client_id}] Wake word re-detected mid-session")
                            continue

                        if "audio_segment" in message:
                            if not session_started:
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

        async def brain():
            try:
                while True:
                    text = await text_queue.get()
                    manager.client_state[client_id] = "THINKING"
                    logger.info(f"[{client_id}] User said: '{text}'")

                    try:
                        reply_text = await route_query(client_id, text)
                        logger.info(f"[{client_id}] Reply: '{reply_text}'")
                    except Exception as e:
                        logger.error(f"route_query error: {e}", exc_info=True)
                        reply_text = "I'm sorry, could you please repeat that?"

                    if reply_text and reply_text.strip():
                        await _send_tts_response(
                            websocket, tts_processor, reply_text, client_id
                        )

                    text_queue.task_done()
                    manager.client_state[client_id] = "WAITING_FOR_PLAYBACK"

            except WebSocketDisconnect:
                logger.info("WebSocket closed during brain loop")
            except Exception as e:
                logger.error(f"Brain task error for {client_id}: {e}", exc_info=True)

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
        clear_session_state(client_id)