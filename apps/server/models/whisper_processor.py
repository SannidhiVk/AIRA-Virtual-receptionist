import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info(
            "Initializing faster-whisper model (distil-large-v3, CUDA float16)..."
        )
        self.model = WhisperModel(
            "distil-large-v3",
            device="cuda",
            compute_type="float16",
        )
        # Threshold for raw audio volume (0.0 to 1.0)
        # If the average volume is below this, we don't even transcribe.
        self.MIN_ENERGY_THRESHOLD = 0.005

        # Common Whisper hallucinations during silence

        logger.info("faster-whisper model ready for transcription")
        self.transcription_count = 0

    async def transcribe_audio(self, audio_bytes):
        try:
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # LAYER 1 — RMS Filter
            rms = np.sqrt(np.mean(audio_array**2))
            if rms < self.MIN_ENERGY_THRESHOLD:
                logger.info(f"Skipping transcription: Audio too quiet (RMS: {rms:.5f})")
                return "NO_SPEECH"

            loop = asyncio.get_event_loop()

            def _run_transcription():

                segments, info = self.model.transcribe(
                    audio_array,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    beam_size=5,
                )

                # 🔴 ADD LANGUAGE GATE HERE
                if info.language_probability < 0.88:
                    logger.info(
                        f"Low language probability: {info.language_probability:.3f}"
                    )
                    return ""

                valid_text = []

                for segment in segments:

                    if segment.no_speech_prob > 0.6:
                        continue

                    if segment.avg_logprob < -1.0:
                        continue

                    if (segment.end - segment.start) < 0.5:
                        continue

                    valid_text.append(segment.text)

                return " ".join(valid_text).strip()

            transcribed_text = await loop.run_in_executor(None, _run_transcription)

            # Final safety check
            if not transcribed_text:
                return "NO_SPEECH"

            self.transcription_count += 1
            logger.info(
                f"Transcription #{self.transcription_count}: '{transcribed_text}'"
            )

            return transcribed_text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
