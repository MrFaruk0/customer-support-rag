import streamlit as st
from agent import build_rag
import logging

logging.basicConfig(level=logging.INFO)

st.title("PDF Q&A Application")

pdf_files = st.file_uploader("Upload PDF files (multiple)", type="pdf", accept_multiple_files=True)

if pdf_files:
    try:
        st.success(f"{len(pdf_files)} PDF files uploaded!")
        qa = build_rag(pdf_files)
    except Exception as exc:
        st.error(f"RAG build error: {exc}")
        st.stop()

    user_input = st.text_input("Ask your question (TR/EN):")
    lang_choice = st.selectbox(
        "Response language",
        options=["Auto", "Turkish", "English"],
        index=0
    )
    if user_input:
        if len(user_input) > 400:
            st.warning("Question is too long. Please write a shorter and more concise question.")
        else:
            try:
                # Determine response language
                def detect_lang(text):
                    tr_chars = set("çğıöşüÇĞİÖŞÜ")
                    if any(ch in tr_chars for ch in text):
                        return "tr"
                    # crude heuristic: if most tokens are ASCII letters, treat as English
                    ascii_letters = sum(c.isalpha() and ord(c) < 128 for c in text)
                    letters = sum(c.isalpha() for c in text)
                    if letters and ascii_letters / max(1, letters) > 0.9:
                        return "en"
                    return "tr"

                target_lang = (
                    "tr" if lang_choice == "Turkish" else
                    "en" if lang_choice == "English" else
                    detect_lang(user_input)
                )

                if target_lang == "en":
                    prefix = "Answer in English in 2-3 concise sentences as a direct summary.\n\n"
                else:
                    prefix = "Cevabı Türkçe, 2-3 cümle, doğrudan ve öz bir özet olarak ver.\n\n"

                prefixed_query = prefix + user_input
                result = qa(prefixed_query)
                answer = (result.get("result") or "").strip()
                if not answer:
                    answer = "No answer generated. Please rephrase the question."
                st.markdown("**Answer:**")
                st.write(answer)

                src_docs = result.get("source_documents") or []
                if src_docs:
                    with st.expander("Source chunks"):
                        for i, d in enumerate(src_docs, 1):
                            page = d.metadata.get("page") if hasattr(d, "metadata") else None
                            title = f"Source {i} (Page {page})" if page else f"Source {i}"
                            st.markdown(title)
                            st.text(d.page_content[:1000])
                else:
                    st.caption("No source documents returned.")
            except KeyError:
                st.error("Expected keys not found in chain. Please try again.")
            except Exception as exc:
                st.error(f"Query execution error: {exc}")
