import json
import uuid
from pathlib import Path
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from src.config import QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY, COLLECTION_NAME

# New Gemini embedding model (text-embedding-004 was retired on 14 Jan 2026)
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Initialize Gemini client and Qdrant client
client = genai.Client(api_key=GEMINI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def extract_resume_chunks(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        print(f"Warning: {pdf_path} not found. Skipping PDF ingestion.")
        return []
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
    chunks = chunk_text(full_text)
    return [{"text": c, "source": "resume.pdf"} for c in chunks]


def extract_linkedin_chunks(json_path: Path) -> list[dict]:
    if not json_path.exists():
        print(f"Warning: {json_path} not found. Skipping JSON ingestion.")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    chunks = []
    for idx, post in enumerate(posts):
        content = post.get("text") or post.get("content", "")
        if content:
            chunks.append(
                {"text": content, "source": "linkedin", "post_id": post.get("id", str(idx))}
            )
    return chunks


def run_ingestion():
    data_dir = Path("data")
    doc_chunks = []
    doc_chunks.extend(extract_resume_chunks(data_dir / "resume.pdf"))
    doc_chunks.extend(extract_linkedin_chunks(data_dir / "linkedin_posts.json"))

    if not doc_chunks:
        print("No documents found to ingest. Ensure files are placed in data/")
        return

    # Delete existing collection to recreate it cleanly with the new embedding model
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)

    print(f"Creating Qdrant collection: {COLLECTION_NAME} with dimension {EMBEDDING_DIM}")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = []
    for item in doc_chunks:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=item["text"],
            config={"output_dimensionality": EMBEDDING_DIM},
        )
        vector = result.embeddings[0].values

        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"text": item["text"], "source": item["source"]},
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully uploaded {len(points)} vector embeddings to Qdrant!")


if __name__ == "__main__":
    run_ingestion()