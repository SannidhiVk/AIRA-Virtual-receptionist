import asyncio
import logging
import numpy as np
import torch

try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None
    print("⚠️ Kokoro TTS not installed. TTS disabled.")
logger = logging.getLogger(__name__)


class KokoroTTSProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if KPipeline is None:
            raise RuntimeError("Kokoro TTS not available")

        # Prefer CUDA if available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            logger.warning(
                "CUDA not available; Kokoro TTS will run on CPU instead of GPU."
            )

        logger.info(f"Initializing Kokoro TTS processor on device='{device}'...")

        try:
            # Initialize Kokoro TTS pipeline on the chosen device
            self.pipeline = KPipeline(lang_code="a", device=device)

            # Set voice
            self.default_voice = "af_sarah"

            logger.info("Kokoro TTS processor initialized successfully")
            # Counter
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            self.pipeline = None

    async def synthesize_initial_speech_with_timing(self, text):
        """Convert initial text to speech using Kokoro TTS data"""
        if not text or not self.pipeline:
            return None, []

        try:
            logger.info(f"Synthesizing initial speech for text: '{text}'")

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []
            all_word_timings = []
            time_offset = 0  # Track cumulative time for multiple segments

            # Use the executor to run the TTS pipeline with minimal splitting
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1,
                    split_pattern=None,  # No splitting for initial text to process faster
                ),
            )

            # Process all generated segments and extract NATIVE timing
            for i, result in enumerate(generator):
                # Extract the components as shown in your screenshot
                gs = result.graphemes  # str - the text graphemes
                ps = result.phonemes  # str - the phonemes
                audio = result.audio.cpu().numpy()  # numpy array
                tokens = result.tokens  # List[en.MToken] - THE TIMING GOLD!

                logger.info(
                    f"Segment {i}: {len(tokens)} tokens, audio shape: {audio.shape}"
                )

                # Extract word timing from native tokens with null checks
                for token in tokens:
                    # Check if timing data is available
                    if token.start_ts is not None and token.end_ts is not None:
                        word_timing = {
                            "word": token.text,
                            "start_time": (token.start_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                            "end_time": (token.end_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                        }
                        all_word_timings.append(word_timing)
                        logger.debug(
                            f"Word: '{token.text}' Start: {word_timing['start_time']:.1f}ms End: {word_timing['end_time']:.1f}ms"
                        )
                    else:
                        # Log when timing data is missing
                        logger.debug(
                            f"Word: '{token.text}' - No timing data available (start_ts: {token.start_ts}, end_ts: {token.end_ts})"
                        )

                # Add audio segment
                audio_segments.append(audio)

                # Update time offset for next segment
                if len(audio) > 0:
                    segment_duration = len(audio) / 24000  # seconds
                    time_offset += segment_duration

            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(
                    f"✨ Initial speech synthesis complete: {len(combined_audio)} samples, {len(all_word_timings)} word timings"
                )
                return combined_audio, all_word_timings
            return None, []

        except Exception as e:
            logger.error(f"Initial speech synthesis with timing error: {e}")
            return None, []

    async def synthesize_remaining_speech_with_timing(self, text):
        """Convert remaining text to speech using Kokoro TTS data"""
        if not text or not self.pipeline:
            return None, []

        try:
            logger.info(
                f"Synthesizing chunk speech for text: '{text[:50]}...' if len(text) > 50 else text"
            )

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []
            all_word_timings = []
            time_offset = 0  # Track cumulative time for multiple segments

            # Determine appropriate split pattern based on text length
            if len(text) < 100:
                split_pattern = None  # No splitting for very short chunks
            else:
                split_pattern = r"[.!?。！？]+"

            # Use the executor to run the TTS pipeline with optimized splitting
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text, voice=self.default_voice, speed=1, split_pattern=split_pattern
                ),
            )

            # Process all generated segments and extract NATIVE timing
            for i, result in enumerate(generator):
                # Extract the components with NATIVE timing
                gs = result.graphemes  # str
                ps = result.phonemes  # str
                audio = result.audio.cpu().numpy()  # numpy array
                tokens = result.tokens  # List[en.MToken] - THE TIMING GOLD!

                logger.info(
                    f"Chunk segment {i}: {len(tokens)} tokens, audio shape: {audio.shape}"
                )

                # Extract word timing from native tokens with null checks
                for token in tokens:
                    # Check if timing data is available
                    if token.start_ts is not None and token.end_ts is not None:
                        word_timing = {
                            "word": token.text,
                            "start_time": (token.start_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                            "end_time": (token.end_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                        }
                        all_word_timings.append(word_timing)
                        logger.debug(
                            f"Chunk word: '{token.text}' Start: {word_timing['start_time']:.1f}ms End: {word_timing['end_time']:.1f}ms"
                        )
                    else:
                        # Log when timing data is missing
                        logger.debug(
                            f"Chunk word: '{token.text}' - No timing data available (start_ts: {token.start_ts}, end_ts: {token.end_ts})"
                        )

                # Add audio segment
                audio_segments.append(audio)

                # Update time offset for next segment
                if len(audio) > 0:
                    segment_duration = len(audio) / 24000  # seconds
                    time_offset += segment_duration

            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(
                    f"✨ Chunk speech synthesis complete: {len(combined_audio)} samples, {len(all_word_timings)} word timings"
                )
                return combined_audio, all_word_timings
            return None, []

        except Exception as e:
            logger.error(f"Chunk speech synthesis with timing error: {e}")
            return None, []
