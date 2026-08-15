import io
import pdfplumber
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
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


def process_and_upsert(file_bytes: bytes, filename: str) -> int:
    text = extract_text(file_bytes, filename)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": filename}) for c in chunks]

    PineconeVectorStore.from_documents(docs, embeddings, index_name=settings.PINECONE_INDEX_NAME)
    logger.info("Indexed %d chunks from %s", len(docs), filename)
    return len(docs)
