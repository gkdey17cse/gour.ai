from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from qdrant_client import QdrantClient

from src.cache import SemanticCache
from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    COLLECTION_NAME,
)

# Initialize core models & clients
semantic_cache = SemanticCache(threshold=0.82)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# LLM Clients
groq_client = (
    OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"api_version": "v1beta"},
)

# Try these Groq models in order — first one that succeeds wins.
# Update this list if Groq deprecates a model:
#   https://console.groq.com/docs/models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# Hard length constraint is enforced in the prompt AND via max_tokens below.
MAX_ANSWER_TOKENS = 280

SYSTEM_PROMPT_TEMPLATE = (
    "You are Dey.ai, an AI representative for Gour Krishna Dey. "
    "Answer the user's question using ONLY the context below.\n\n"
    "STRICT RULES:\n"
    "1. Answer in 2 to 4 short sentences. Hard cap: ~90 to ~120 words.\n"
    "2. Do NOT use bullet points, numbered lists, or markdown headers.\n"
    "3. Do NOT restate the question. Do NOT pad with generic praise.\n"
    "4. If the context does not contain the answer, say so in one sentence.\n"
    "5. Lead with the direct fact (numbers, dates, names) the user asked for.\n\n"
    "Context:\n{context}"
)


def retrieve_context(query: str, top_k: int = 4) -> str:
    """Searches Qdrant vector database for top matching document chunks."""
    query_vector = semantic_cache.get_embedding(query).tolist()

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


def _try_groq(system_prompt: str, user_query: str):
    """Try each Groq model in order; return (answer, model_name) or (None, None)."""
    if not groq_client:
        return None, None

    last_error = None
    for model in GROQ_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.1,
                max_tokens=MAX_ANSWER_TOKENS,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                print(f"[GROQ OK] model={model} tokens={len(answer.split())} words")
                return answer, model
        except Exception as e:
            last_error = e
            print(f"[GROQ ERROR] model={model} :: {type(e).__name__}: {e}")

    print(f"[GROQ] All models failed. Last error: {last_error}")
    return None, None


def _try_gemini(system_prompt: str, user_query: str):
    try:
        full_prompt = f"{system_prompt}\n\nUser Question: {user_query}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=MAX_ANSWER_TOKENS,
            ),
        )
        answer = (response.text or "").strip()
        if answer:
            return answer
        return None
    except Exception as e:
        print(f"[GEMINI ERROR] {type(e).__name__}: {e}")
        return None


def generate_rag_response(user_query: str) -> dict:
    """
    1. Checks Semantic Cache.
    2. Fallback to Qdrant Vector Retrieval + Groq LLM (with Gemini backup).
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
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    # Groq first (model fallback chain inside)
    groq_answer, groq_model = _try_groq(system_prompt, user_query)
    if groq_answer:
        return {
            "answer": groq_answer,
            "provider": f"Groq ({groq_model})",
            "context_used": context,
            "is_cached": False,
        }

    # Gemini backup
    gemini_answer = _try_gemini(system_prompt, user_query)
    if gemini_answer:
        return {
            "answer": gemini_answer,
            "provider": "Gemini Backup (gemini-2.5-flash)",
            "context_used": context,
            "is_cached": False,
        }

    raise RuntimeError("Both Groq and Gemini API calls failed. Check server logs.")