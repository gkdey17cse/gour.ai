# Dey.ai Profile RAG API Backend

A production-ready **Retrieval-Augmented Generation (RAG) API** that intelligently answers questions about professional background, technical skills, work experience, projects, and educational qualifications.

This backend powers **Dey.ai**, utilizing a multi-layered architecture:

1. **Semantic Cache Engine:** Instantly serves deterministic responses for frequent intent-based queries (e.g., greetings, contact info) using cosine similarity.
2. **Vector Retrieval:** Fetches relevant contextual chunks from **Qdrant Cloud**.
3. **LLM Generation:** Uses **Groq API (`llama-3.3-70b-versatile`)** for high-speed primary inference, with a built-in fallback to **Gemini API (`gemini-2.5-flash`)**.

## Tech Stack

| Component | Technology | 
| ----- | ----- | 
| Backend Framework | FastAPI | 
| Programming Language | Python 3.10+ | 
| Semantic Cache | `sentence-transformers`, Cosine Similarity | 
| Primary LLM | Groq API (`llama-3.3-70b-versatile`) | 
| Fallback LLM | Google Gemini API (`gemini-2.5-flash`) | 
| Vector Database | Qdrant Cloud | 
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | 
| ASGI Server | Uvicorn | 
| Deployment Platform | Railway / Render | 

## Project Structure

```text
profile-rag-api/
│
├── data/
│   ├── .gitkeep
│   ├── linkedin_posts.json
│   └── resume.pdf
│
├── src/
│   ├── __init__.py
│   ├── cache.py          # Semantic caching engine and intent definitions
│   ├── config.py         # Environment variables configuration
│   ├── ingest.py         # Document parsing and Qdrant vector indexing
│   ├── retriever.py      # Cache routing, vector search, and LLM generation
│   └── main.py           # FastAPI application and endpoints
│
├── test_api.py           # Local testing script
├── Dockerfile            # Containerization instructions
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

Follow the instructions below to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd profile-rag-api
```

### 2. Create a Virtual Environment

Create and activate a Python virtual environment.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file using the provided template:

```bash
cp .env.example .env
```

Update `.env` with your actual API credentials:

```ini
# Groq & Gemini API Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Cloud Configuration
QDRANT_CLUSTER_ENDPOINT=https://your-cluster-id.us-east-4-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Embedding Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Vector Database Collection
COLLECTION_NAME=profile_docs
```

> **Security:** Never commit your `.env` file or expose API credentials in public repositories.

### 5. Prepare and Ingest Data

Place your `resume.pdf` and `linkedin_posts.json` inside the `data/` directory.

Run the ingestion script from the project root to chunk your data, generate embeddings, and upload them to Qdrant Cloud:

```bash
python -m src.ingest
```

### 6. Start the API Server

Run the FastAPI application from the project root using Uvicorn:

```bash
uvicorn src.main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

**Interactive API Documentation:**
* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

## API Documentation

### 1. Health Check

**Endpoint:** `GET /health`

Checks whether the API service is running.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
    "status": "ok",
    "service": "Dey.ai RAG Pipeline"
}
```

### 2. Chat API

**Endpoint:** `POST /chat`

Accepts a natural-language query and routes it through the Semantic Cache. If a high-confidence match is found, it returns instantly. Otherwise, it retrieves context from Qdrant and generates a response via LLM.

**Request Body:**
```json
{
    "query": "Hi Gour, what are you currently working on?"
}
```

**Example Response (Cache Hit - Instant):**
```json
{
    "answer": "At EXL Services, Gour is currently working on enhancing a Python-based data lineage pipeline...",
    "provider": "Semantic Cache (Intent: current_project)",
    "context_used": "Deterministic answer returned from semantic cache.",
    "is_cached": true
}
```

**Example Response (Cache Miss - RAG Generation):**
```json
{
    "answer": "Gour Krishna Dey has experience working with Python in the context of data engineering and Agentic AI...",
    "provider": "Groq (llama-3.3-70b)",
    "context_used": "[resume.pdf]: Python, SAS, Agentic ADK...\n---\n[linkedin]: Excited to launch my new RAG pipeline...",
    "is_cached": false
}
```

## Deployment

The application is containerized and ready to be deployed on cloud platforms such as **Railway** or **Render**.

### Deployment Checklist

1. Push the project to a GitHub repository.
2. Create a new project on Railway or Render connected to the repository.
3. Configure the required environment variables (`GROQ_API_KEY`, `GEMINI_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY`).
4. The provided `Dockerfile` will automatically handle building and starting the application using Uvicorn.

**Docker Start Command (handled automatically):**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Security and Git Configuration

* `.env` files contain sensitive API credentials and must never be committed.
* Keep the `data/.gitkeep` file intact so the data folder structure is preserved in version control without committing personal documents.

## License

This project is built for the personal portfolio and professional profile of Gour Krishna Dey.