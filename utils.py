import logging
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_pdf_chunks(file_path, chunk_size=700, overlap=120):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    # Support both filesystem paths and in-memory file-like objects
    pages_text: list[tuple[int, str]] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                raise ValueError("PDF has no pages")
            for idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                # Basic cleanup: collapse whitespace and dashes broken by line breaks
                cleaned = " ".join(page_text.split())
                cleaned = cleaned.replace("- ", "")
                cleaned = cleaned.strip()
                pages_text.append((idx, cleaned))
    except Exception as exc:
        logger.exception("Failed to read PDF: %s", exc)
        raise

    # Concatenate text to check emptiness
    all_text = "\n".join(t for _, t in pages_text).strip()
    if not all_text:
        raise ValueError("PDF appears to be empty or contains no extractable text")

    # Sentence-aware recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )

    documents: list[Document] = []
    for page_num, text in pages_text:
        if not text:
            continue
        parts = splitter.split_text(text)
        for part in parts:
            if not part.strip():
                continue
            documents.append(
                Document(page_content=part, metadata={"page": page_num})
            )

    if not documents:
        raise ValueError("No chunks produced from PDF text")

    logger.info("Created %d chunks (size=%d, overlap=%d)", len(documents), chunk_size, overlap)
    return documents

def get_embedding_model():
    # Multilingual embeddings for TR/EN robustness
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True}
    )

def call_openrouter(prompt: str, temperature: float = 0.2, max_tokens: int = 512, model: str = "deepseek/deepseek-chat-v3.1:free") -> str:
    """Call OpenRouter's OpenAI-compatible API to generate a response.

    Requires env var OPENROUTER_API_KEY.
    """
    # Load variables from .env if present
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            # Optional but recommended by OpenRouter
            "HTTP-Referer": "https://localhost",  # or your deployed URL
            "X-Title": "PDF RAG Chatbot"
        }
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, concise RAG assistant. Paraphrase; do not copy verbatim."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()
