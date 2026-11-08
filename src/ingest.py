import json
import uuid
from pathlib import Path
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL_NAME, COLLECTION_NAME

# Initialize embedding model and Qdrant client
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits raw text into overlapping token/character chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def extract_resume_chunks(pdf_path: Path) -> list[dict]:
    """Extracts text from PDF resume and returns chunk objects."""
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
    """Extracts text from LinkedIn posts JSON file."""
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
                {
                    "text": content,
                    "source": "linkedin",
                    "post_id": post.get("id", str(idx)),
                }
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

    print(f"Total chunks extracted: {len(doc_chunks)}")

    # Ensure collection exists in Qdrant
    if not qdrant.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # Encode texts and push to Qdrant
    points = []
    for item in doc_chunks:
        vector = embedder.encode(item["text"]).tolist()
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
