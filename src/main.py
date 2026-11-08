from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.retriever import generate_rag_response

app = FastAPI(
    title="gour.ai Profile RAG API",
    description="Vector-backed profile assistant with Semantic Cache and LLM fallbacks.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., example="Hi Gour, what is your current project?")


class QueryResponse(BaseModel):
    answer: str
    provider: str
    context_used: str
    is_cached: bool


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "gour.ai RAG Pipeline"}


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        result = generate_rag_response(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
