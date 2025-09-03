import streamlit as st
from agent import build_rag

st.title("PDF Soru-Cevap Uygulaması")

pdf_file = st.file_uploader("PDF dosyası yükle", type="pdf")

if pdf_file is not None:
    st.success("PDF yüklendi!")
    qa = build_rag(pdf_file)

    user_input = st.text_input("Sorunuzu yazın:")
    if user_input:
        # Input key kesinlikle "query" olmalı
        result = qa({"query": user_input})
        st.write("**Cevap:**", result["result"])
