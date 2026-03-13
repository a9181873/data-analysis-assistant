from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os

# Ensure the project root is on the Python path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

class KnowledgeRequest(BaseModel):
    query: str
    n_results: int = 3

# Cache pattern simple memory mapping for agents associated with specific session IDs
AGENT_STORE: Dict[str, Any] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Sends a query to the LangChain Agent.
    Lazy-imports langchain_agent only when called.
    """
    try:
        from api.routers.data import SESSION_STORE
        from langchain_agent import create_agent_executor
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"AI dependencies not available: {e}. Please install langchain, langchain-ollama, etc."
        )

    df = SESSION_STORE.get(request.session_id)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded for this session. Please upload a dataset first."
        )

    try:
        if request.session_id not in AGENT_STORE:
            agent_executor = create_agent_executor(df)
            AGENT_STORE[request.session_id] = agent_executor
        else:
            agent_executor = AGENT_STORE[request.session_id]

        response = agent_executor.invoke({"input": request.message})
        return ChatResponse(response=response.get("output", "系統未能產出有效回答"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 代理處理錯誤: {str(e)}")


@router.post("/knowledge")
async def search_knowledge_base(request: KnowledgeRequest):
    """
    Direct endpoint for searching the Vector DB (RAG) manually.
    """
    try:
        from rag_manager import get_chroma_collection, query_rag_with_scores
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG dependencies not available: {e}"
        )

    try:
        _, collection = get_chroma_collection()
        results = query_rag_with_scores(request.query, collection, request.n_results)
        formatted_results = [
            {"document": doc, "distance": dist} for doc, dist in results
        ]
        return {"status": "success", "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
