import logging

logger = logging.getLogger(__name__)


async def process_audio_segment(*args, **kwargs):
    """Deprecated: audio processing is now handled directly in websocket_routes via queues."""
    logger.warning(
        "process_audio_segment is deprecated. Audio is now processed in websocket_routes via asyncio.Queue."
    )
