import logging
from fastapi import APIRouter
from pydantic import BaseModel

from managers.connection_manager import manager
from main import process_text_for_client

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.get("/stats")
async def get_stats():
    """Get server statistics (voice-only)"""
    return manager.get_stats()


@router.post("/query")
async def handle_text_query(payload: QueryRequest):
    """
    Main text query endpoint.

    Instead of calling the LLM directly, this routes the query through
    the intent detector and database grounding workflow.
    """
    reply = await process_text_for_client("api", payload.query)
    return {"response": reply}
