import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

def load_pdf_chunks(file_path, chunk_size=400, overlap=50):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_local_llm():
    hf_model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_length=512
    )
    return HuggingFacePipeline(pipeline=pipe)
