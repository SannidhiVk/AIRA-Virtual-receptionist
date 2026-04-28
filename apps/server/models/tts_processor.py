import asyncio
import logging
import numpy as np
import torch
import os

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

            # Set voice to the local file we downloaded
            # This safely gets the path to apps/server/voices/af_sarah.pt
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_voice_path = os.path.join(base_dir, "voices", "af_sarah.pt")

            if os.path.exists(local_voice_path):
                self.default_voice = local_voice_path
                logger.info(f"Loaded local voice file from: {local_voice_path}")
            else:
                self.default_voice = "af_sarah"
                logger.warning(
                    f"Local voice not found at {local_voice_path}. Falling back to internet download."
                )

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

            # TalkingHead expects wtimes/wdurations in milliseconds.
            # Kokoro token timestamps are in seconds (float), so convert with *1000.
            seconds_to_ms = 1000.0
            logged_first_word = False

            # Process all generated segments and extract native timing
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
                        start_time_ms = (token.start_ts * seconds_to_ms) + (
                            time_offset * 1000.0
                        )
                        end_time_ms = (token.end_ts * seconds_to_ms) + (
                            time_offset * 1000.0
                        )
                        word_timing = {
                            "word": token.text,
                            "start_time": start_time_ms,
                            "end_time": end_time_ms,
                        }
                        all_word_timings.append(word_timing)
                        if not logged_first_word:
                            logger.info(
                                "Initial timing check | word='%s' raw_start_ts=%s raw_end_ts=%s -> start_time_ms=%.2f end_time_ms=%.2f",
                                token.text,
                                token.start_ts,
                                token.end_ts,
                                start_time_ms,
                                end_time_ms,
                            )
                            logged_first_word = True
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

            # TalkingHead expects wtimes/wdurations in milliseconds.
            # Kokoro token timestamps are in seconds (float), so convert with *1000.
            seconds_to_ms = 1000.0
            logged_first_word = False

            # Process all generated segments and extract native timing
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
                        start_time_ms = (token.start_ts * seconds_to_ms) + (
                            time_offset * 1000.0
                        )
                        end_time_ms = (token.end_ts * seconds_to_ms) + (
                            time_offset * 1000.0
                        )
                        word_timing = {
                            "word": token.text,
                            "start_time": start_time_ms,
                            "end_time": end_time_ms,
                        }
                        all_word_timings.append(word_timing)
                        if not logged_first_word:
                            logger.info(
                                "Chunk timing check | word='%s' raw_start_ts=%s raw_end_ts=%s -> start_time_ms=%.2f end_time_ms=%.2f",
                                token.text,
                                token.start_ts,
                                token.end_ts,
                                start_time_ms,
                                end_time_ms,
                            )
                            logged_first_word = True
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
