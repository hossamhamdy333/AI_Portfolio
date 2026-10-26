import io
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import embeddings, settings, logger


def extract_pages(file_bytes: bytes, filename: str) -> list[tuple[int | None, str]]:
    """Returns (page_number, text) pairs. Only PDFs have real page numbers,
    other file types come back as one entry with page None."""
    lower = filename.lower()

    if lower.endswith(".pdf"):
        pages = []
        pdf = pdfium.PdfDocument(file_bytes)
        try:
            for number in range(1, len(pdf) + 1):
                extracted = pdf[number - 1].get_textpage().get_text_range().replace("\r\n", "\n")
                if extracted.strip():
                    pages.append((number, extracted))
        finally:
            pdf.close()
        return pages
    if lower.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(file_bytes))
        return [(None, pytesseract.image_to_string(image))]
    if lower.endswith((".txt", ".md")):
        return [(None, file_bytes.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type for '{filename}'. Use PDF, PNG, JPG, TXT, or MD.")


def extract_text(file_bytes: bytes, filename: str) -> str:
    return "".join(text + "\n" for _, text in extract_pages(file_bytes, filename))


def process_and_upsert(file_bytes: bytes, filename: str, user_id: int, document_id: int) -> int:
    pages = extract_pages(file_bytes, filename)

    if not any(text.strip() for _, text in pages):
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # user_id in the metadata is what makes per-user isolation possible -
    # agent.py's retriever filters on this field, so one user's documents
    # are never returned for another user's question, not just hidden by
    # the UI. document_id lets a specific upload's chunks be found and
    # removed later (see main.py's DELETE /documents/{id}) without
    # accidentally touching a different upload that happens to share the
    # same filename. Each page is split on its own, so a chunk never
    # crosses a page and its page number is always exact.
    docs = []
    for page_number, text in pages:
        for chunk in splitter.split_text(text):
            metadata = {"source": filename, "user_id": user_id, "document_id": document_id}
            if page_number is not None:
                metadata["page"] = page_number
            docs.append(Document(page_content=chunk, metadata=metadata))

    QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        timeout=30,
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
