from langchain.vectorstores import FAISS
from langchain.schema import Document
from utils import load_pdf_chunks, get_embedding_model, call_openrouter
import logging

def build_rag(pdf_files):
    # Allow single or multiple uploaded PDFs
    if not isinstance(pdf_files, (list, tuple)):
        pdf_files = [pdf_files]

    all_chunks = []
    for f in pdf_files:
        all_chunks.extend(load_pdf_chunks(f))
    # load_pdf_chunks now returns Documents with page metadata
    docs = all_chunks

    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

    def ask(query: str, k: int = 4):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 24, "lambda_mult": 0.6}
        )
        retrieved = retriever.get_relevant_documents(query)
        context = "\n\n".join(d.page_content for d in retrieved)
        prompt = (
            "You are a helpful, concise RAG assistant. Answer strictly in the same language as the question. "
            "Use the context to answer in 2-3 sentences. Paraphrase; do not copy. If insufficient, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        answer = call_openrouter(prompt)
        return {"result": answer, "source_documents": retrieved}

    return ask
