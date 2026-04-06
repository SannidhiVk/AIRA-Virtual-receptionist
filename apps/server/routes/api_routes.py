import logging
from fastapi import APIRouter
from pydantic import BaseModel

from managers.connection_manager import manager
from services.query_router import route_query

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    client_id: str = "api"  # REST callers can optionally pass their own session ID


@router.get("/stats")
async def get_stats():
    """Get server statistics."""
    return manager.get_stats()


@router.post("/query")
async def handle_text_query(payload: QueryRequest):
    """
    Main text query endpoint.
    Routes through the full conversation/registration engine, same as WebSocket.
    """
    reply = await route_query(payload.client_id, payload.query)
    return {"response": reply}
