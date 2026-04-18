from fastapi import APIRouter
from pydantic import BaseModel
from app.chatbot import answer_query
from fastapi import APIRouter, Depends
from app.core.dependencies import require_permission

router = APIRouter()

class ChatRequest(BaseModel):
    query: str


@router.post("/chat")
def chat(
        req: ChatRequest,
        user = Depends(require_permission("chat"))   # 🔐 ADD THIS
):
    result = answer_query(req.query)

    return {
        "query": req.query,
        "response": result["response"],
        "sources": result["sources"],
        "retrieved_context": result["retrieved_context"],
    }