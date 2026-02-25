import asyncio
import base64
import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from managers.connection_manager import manager
from models.whisper_processor import WhisperProcessor
from models.ollama_processor import OllamaProcessor
from models.tts_processor import KokoroTTSProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time Receptionist AI interaction (audio-only)."""
    await manager.connect(websocket, client_id)

    # Get instances of processors
    whisper_processor = WhisperProcessor.get_instance()
    ollama_processor = OllamaProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    # Per-connection text queue
    text_queue: asyncio.Queue[str] = asyncio.Queue()

    try:
        # Send initial configuration confirmation
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

        async def send_keepalive():
            """Send periodic keepalive pings."""
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)  # Send ping every 10 seconds
                except Exception:
                    break

        async def listener():
            """Receive audio from client, run STT, and enqueue transcribed text."""
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)

                        # Handle complete audio segments from frontend
                        if "audio_segment" in message:
                            audio_data = base64.b64decode(message["audio_segment"])
                            logger.info(
                                f"Received audio-only segment: {len(audio_data)} bytes"
                            )

                            # Transcribe using faster-whisper (CPU)
                            transcribed_text = await whisper_processor.transcribe_audio(
                                audio_data
                            )
                            logger.info(
                                f"Listener transcription result: '{transcribed_text}'"
                            )

                            if transcribed_text in [
                                "NOISE_DETECTED",
                                "NO_SPEECH",
                                None,
                            ]:
                                continue

                            await text_queue.put(transcribed_text)

                        # Ignore non-audio payloads for the Receptionist AI

                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {e}")
                        await websocket.send_text(
                            json.dumps({"error": "Invalid JSON format"})
                        )
                    except KeyError as e:
                        logger.error(f"Missing key in message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Missing required field: {e}"})
                        )
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Processing error: {str(e)}"})
                        )
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed during listener loop")

        async def brain():
            """Consume transcribed text, get LLM response, synthesize TTS, and send audio."""
            try:
                while True:
                    text = await text_queue.get()
                    manager.client_state[client_id] = "THINKING"

                    # Get LLM response from Ollama
                    reply_text = await ollama_processor.get_response(text)
                    logger.info(f"Ollama reply: '{reply_text}'")

                    # Synthesize speech with Kokoro TTS (non-blocking)
                    audio, word_timings = await tts_processor.synthesize_initial_speech_with_timing(  # type: ignore[attr-defined]
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
                logger.info("WebSocket connection closed during brain loop")
            except Exception as e:
                logger.error(f"Brain task error for client {client_id}: {e}")

        # Run tasks concurrently
        listener_task = asyncio.create_task(listener())
        brain_task = asyncio.create_task(brain())
        keepalive_task = asyncio.create_task(send_keepalive())

        # Track tasks in manager for potential cancellation
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
                _ = task.result()
            except Exception as e:
                logger.error(f"Task finished with error: {e}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket session error for client {client_id}: {e}")
    finally:
        # Cleanup
        logger.info(f"Cleaning up resources for client {client_id}")
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)
