import io
import pdfplumber
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import embeddings, settings, logger


def extract_text(file_bytes: bytes, filename: str) -> str:
    lower = filename.lower()
    text = ""

    if lower.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    elif lower.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
    elif lower.endswith((".txt", ".md")):
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type for '{filename}'. Use PDF, PNG, JPG, TXT, or MD.")

    return text


def process_and_upsert(file_bytes: bytes, filename: str, user_id: int, document_id: int) -> int:
    text = extract_text(file_bytes, filename)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    # user_id in the metadata is what makes per-user isolation possible -
    # agent.py's retriever filters on this field, so one user's documents
    # are never returned for another user's question, not just hidden by
    # the UI. document_id lets a specific upload's chunks be found and
    # removed later (see main.py's DELETE /documents/{id}) without
    # accidentally touching a different upload that happens to share the
    # same filename.
    docs = [
        Document(page_content=c, metadata={"source": filename, "user_id": user_id, "document_id": document_id})
        for c in chunks
    ]

    QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
    )
    logger.info("Indexed %d chunks from %s for user %d (document_id=%d)", len(docs), filename, user_id, document_id)
    return len(docs)


def delete_document_chunks(qdrant_client, document_id: int, user_id: int) -> None:
    """
    Removes every vector chunk belonging to one document. Filtered on both
    document_id and user_id (not document_id alone) as defense in depth -
    the caller (main.py) already checks ownership before calling this, but
    a second independent check here means a bug in that ownership check
    can't turn into "delete anyone's document just by guessing an id."
    """
    qdrant_client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=Filter(must=[
            FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id)),
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
        ]),
    )
    logger.info("Deleted vector chunks for document_id=%d (user %d)", document_id, user_id)
