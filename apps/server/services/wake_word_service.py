import logging
from typing import Optional

import numpy as np
import openwakeword
from openwakeword.model import Model


logger = logging.getLogger(__name__)

# OpenWakeWord expects 80 ms frames at 16 kHz = 1280 samples (2560 bytes int16 PCM).
OWW_CHUNK_SAMPLES = 1280
OWW_CHUNK_BYTES = OWW_CHUNK_SAMPLES * 2


class WakeWordService:
    """In-memory OpenWakeWord wrapper for 16kHz/16-bit mono PCM chunks."""

    def __init__(self, wakeword: str = "hey_jarvis", threshold: float = 0.6):
        openwakeword.utils.download_models()
        self.model = Model(wakeword_models=[wakeword], inference_framework="onnx")
        self.model_name = list(self.model.models.keys())[0]
        self.wakeword = wakeword
        self.threshold = threshold
        logger.info(
            "WakeWordService initialized: requested='%s', loaded='%s', threshold=%.3f",
            self.wakeword,
            self.model_name,
            self.threshold,
        )

    def process_chunk(self, audio_chunk: bytes) -> tuple[bool, float]:
        """Return (triggered, score) for one 80ms PCM chunk."""
        if len(audio_chunk) != OWW_CHUNK_BYTES:
            logger.warning(
                "WakeWordService received %s bytes; expected %s. Skipping chunk.",
                len(audio_chunk),
                OWW_CHUNK_BYTES,
            )
            return False, 0.0

        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        prediction = self.model.predict(audio_data)
        score = float(prediction.get(self.model_name, 0.0))
        return score >= self.threshold, score

    def is_triggered(self, audio_chunk: bytes) -> bool:
        """Boolean-only helper used by state machine logic."""
        triggered, _ = self.process_chunk(audio_chunk)
        return triggered


_service_instance: Optional[WakeWordService] = None


def get_wake_word_service(wakeword: str = "hey_jarvis") -> WakeWordService:
    """Lazy singleton so the model is loaded once and kept in memory."""
    global _service_instance
    if _service_instance is None:
        _service_instance = WakeWordService(wakeword=wakeword)
    return _service_instance