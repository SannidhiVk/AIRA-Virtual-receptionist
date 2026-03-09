import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ Correct imports (relative to apps/server)
from core.config import logger
from core.lifespan import lifespan
from routes.api_routes import router as api_router
from routes.websocket_routes import router as websocket_router

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
    ],  # <-- Change this line
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(websocket_router)


@app.get("/")
async def health():
    return {"status": "running"}


def main():
    logger.info("Starting AlmostHuman Voice Assistant server...")

    config = uvicorn.Config(
        app="main:app",  # important change
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
