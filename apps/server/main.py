import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import logger
from core.lifespan import lifespan

from services.query_router import route_query, clear_session_state

app = FastAPI(
    title="AlmostHuman Voice Assistant",
    description="CPU-optimized voice assistant with real-time speech recognition, conversational brain, and text-to-speech.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_text_for_client(client_id: str, text: str) -> str:
    """
    Dispatcher for incoming text from WebSocket / HTTP routes.
    """
    if not text:
        return ""

    try:
       
        reply_text = await route_query(client_id, text)

        return reply_text

    except Exception as e:
        logger.error("Dispatcher failed: %s", e)
        return "I'm sorry, I encountered an internal error. Please try again."


@app.get("/")
async def health():
    return {"status": "running"}


# API endpoints for frontend (e.g. if someone clicks 'Reset Session' on the UI)
@app.post("/reset_session/{client_id}")
async def reset_session(client_id: str):
    clear_session_state(client_id)
    return {"status": "cleared"}


# Include routes after dispatcher is defined
from routes.api_routes import router as api_router
from routes.websocket_routes import router as websocket_router

app.include_router(api_router)
app.include_router(websocket_router)


def main():
    logger.info("Starting AlmostHuman Voice Assistant server...")
    config = uvicorn.Config(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
