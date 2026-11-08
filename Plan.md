
# Profile RAG Pipeline Blueprint

## Phase 1: Environment & Project Setup

- [X] Directory structure initialization
- [X] Dependency management (`requirements.txt`)
- [ ] Environment variable configuration (`.env`)

## Phase 2: Data Processing & Vector Indexing (`src/ingest.py`)

- [ ] Resume PDF parser using `pypdf`
- [ ] LinkedIn / Document JSON parser
- [ ] Text chunking (400 token chunks, 50 token overlap)
- [ ] Embedding generation via `sentence-transformers` (`all-MiniLM-L6-v2`)
- [ ] Indexing vector payload into Qdrant Cloud collection (`profile_docs`)

## Phase 3: Retrieval Engine & LLM Pipeline (`src/retriever.py`)

- [ ] Vector similarity search client initialization
- [ ] Context formatting module
- [ ] System prompt engineering (Profile persona grounding)
- [ ] Groq LPU API integration (`llama-3.3-70b-versatile`)

## Phase 4: API Endpoint Layer (`src/main.py`)

- [ ] FastAPI instance setup with CORS middleware
- [ ] POST `/chat` query endpoint implementation
- [ ] Healthcheck endpoint GET `/health`
- [ ] Input validation & response formatting

## Phase 5: Containerization & Deployment

- [ ] Dockerfile creation & local verification
- [ ] Deployment to Railway / Render
- [ ] Production endpoint validation & API test suite
