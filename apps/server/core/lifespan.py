import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from models.whisper_processor import WhisperProcessor
from models.groq_processor import GroqProcessor
from models.tts_processor import KokoroTTSProcessor
from receptionist.database import engine
from receptionist.models import Base
from receptionist.seed_data import seed_database
from services.face_recognition_service import cleanup_old_captures

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing models on startup...")
    try:
        # Initialize processors to load models
        whisper_processor = WhisperProcessor.get_instance()
        llm_processor = GroqProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()

        # Initialize receptionist database (SQLite)
        loop = asyncio.get_running_loop()

        def _init_db():
            Base.metadata.create_all(bind=engine)
            seed_database()

        await loop.run_in_executor(None, _init_db)
        deleted_captures = await loop.run_in_executor(None, cleanup_old_captures)
        logger.info(
            "Diagnostic capture cleanup completed. Deleted=%s", deleted_captures
        )

        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")
    # Close any remaining connections
    from managers.connection_manager import manager

    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
        manager.disconnect(client_id)
    logger.info("Server shutdown complete")
