import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperProcessor:
    """Handles speech-to-text using faster-whisper model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing faster-whisper model (distil-large-v3, CPU int8)...")
        self.model = WhisperModel(
            "distil-large-v3",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
        logger.info("faster-whisper model ready for transcription")
        self.transcription_count = 0

    async def transcribe_audio(self, audio_bytes):
        """Transcribe audio bytes to text using faster-whisper with VAD."""
        try:
            # Convert 16-bit PCM audio bytes to float32 numpy array in [-1.0, 1.0]
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            loop = asyncio.get_event_loop()

            def _run_transcription():
                segments, _info = self.model.transcribe(
                    audio_array,
                    vad_filter=True,
                )
                # segments is a generator; materialize to list to safely iterate twice
                return " ".join(segment.text for segment in segments).strip()

            transcribed_text = await loop.run_in_executor(None, _run_transcription)
            self.transcription_count += 1

            logger.info(
                f"Transcription #{self.transcription_count}: '{transcribed_text}'"
            )

            # Check for noise/empty transcription
            if not transcribed_text or len(transcribed_text) < 3:
                return "NO_SPEECH"

            # Check for common noise indicators
            noise_indicators = ["thank you", "thanks for watching", "you", ".", ""]
            if transcribed_text.lower().strip() in noise_indicators:
                return "NOISE_DETECTED"

            return transcribed_text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
