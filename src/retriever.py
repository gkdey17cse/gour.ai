from google import genai
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.cache import SemanticCache
from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
)

# Initialize core models & clients
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
semantic_cache = SemanticCache(embedder=embedder, threshold=0.82)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# LLM Clients
groq_client = (
    OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def retrieve_context(query: str, top_k: int = 4) -> str:
    """Searches Qdrant vector database for top matching document chunks."""
    query_vector = embedder.encode(query).tolist()

    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
        )
        search_results = response.points
    else:
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k
        )

    if not search_results:
        return "No relevant profile documents found."

    context_blocks = [
        f"[{hit.payload.get('source', 'doc')}]: {hit.payload['text']}"
        for hit in search_results
    ]
    return "\n---\n".join(context_blocks)


def generate_rag_response(user_query: str) -> dict:
    """
    1. Checks Semantic Cache. If confidence >= 0.82, returns cached response instantly.
    2. Fallback to Qdrant Vector Retrieval + Groq LLM (or Gemini backup).
    """
    # 1. Semantic Cache Evaluation
    cached_result = semantic_cache.check_cache(user_query)
    if cached_result:
        print(
            f"[CACHE HIT] Score: {cached_result['similarity_score']} for query: '{user_query}'"
        )
        return {
            "answer": cached_result["answer"],
            "provider": cached_result["provider"],
            "context_used": "Deterministic answer returned from semantic cache.",
            "is_cached": True,
        }

    print(f"[CACHE MISS] Routing to RAG + LLM pipeline for query: '{user_query}'")

    # 2. Vector Retrieval & LLM Generation
    context = retrieve_context(user_query)
    system_prompt = (
        "You are gour.ai, an AI representative for Gour Krishna Dey. "
        "Answer the user's question accurately using ONLY the provided context below. "
        "Be concise, professional, and directly highlight candidate achievements.\n\n"
        f"Context:\n{context}"
    )

    # Groq API Attempt
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            return {
                "answer": response.choices[0].message.content,
                "provider": "Groq (llama-3.3-70b)",
                "context_used": context,
                "is_cached": False,
            }
        except Exception as e:
            print(f"Groq API Error: {e}. Falling back to Gemini API...")

    # Gemini Backup Attempt
    if gemini_client:
        try:
            full_prompt = f"{system_prompt}\n\nUser Question: {user_query}"
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=full_prompt
            )
            return {
                "answer": response.text,
                "provider": "Gemini Backup (gemini-2.5-flash)",
                "context_used": context,
                "is_cached": False,
            }
        except Exception as e:
            raise RuntimeError(f"Both Groq and Gemini API calls failed: {e}")

    raise RuntimeError("No working LLM API configured.")
