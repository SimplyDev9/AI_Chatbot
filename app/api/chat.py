from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator
from starlette.requests import Request

from app.chatbot import answer_query
from app.core.dependencies import require_permission
from app.core.limiter import limiter

router = APIRouter()

MAX_QUERY_LENGTH = 500


class ChatRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty_or_too_long(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be empty.")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.")
        return v


@router.post("/chat")
@limiter.limit("30/minute")
def chat(
        req: ChatRequest,
        request: Request,
        user=Depends(require_permission("chat")),
):
    result = answer_query(req.query)

    return {
        "query": req.query,
        "response": result["response"],
        "sources": result["sources"],
        # retrieved_context intentionally omitted — internal KB detail
    }