import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "profile_docs")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Qdrant configuration missing in environment variables.")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise ValueError("At least one LLM API Key (GROQ_API_KEY or GEMINI_API_KEY) must be provided.")