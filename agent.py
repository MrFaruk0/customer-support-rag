from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from utils import load_pdf_chunks, get_embedding_model, get_local_llm

def build_rag(pdf_file):
    pdf_chunks = load_pdf_chunks(pdf_file)
    docs = [Document(page_content=chunk) for chunk in pdf_chunks]

    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

    llm = get_local_llm()

    # Prompt input variable MUST be "query"
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Aşağıdaki bağlamı kullanarak soruyu açık ve doğru şekilde cevapla.\n\nBağlam: {context}\n\nSoru: {query}\nCevap:"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa
