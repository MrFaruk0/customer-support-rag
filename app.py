import streamlit as st
from agent import build_rag
import logging

logging.basicConfig(level=logging.INFO)

st.title("PDF Soru-Cevap Uygulaması")

pdf_files = st.file_uploader("PDF dosyaları yükle (birden fazla)", type="pdf", accept_multiple_files=True)

if pdf_files:
    try:
        st.success(f"{len(pdf_files)} PDF yüklendi!")
        qa = build_rag(pdf_files)
    except Exception as exc:
        st.error(f"RAG kurulurken hata oluştu: {exc}")
        st.stop()

    user_input = st.text_input("Sorunuzu yazın (TR/EN):")
    lang_choice = st.selectbox(
        "Yanıt dili",
        options=["Auto", "Türkçe", "English"],
        index=0
    )
    if user_input:
        if len(user_input) > 400:
            st.warning("Soru çok uzun. Lütfen daha kısa ve net bir soru yazın.")
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
                    "tr" if lang_choice == "Türkçe" else
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
                    answer = "Bağlamı özetleyecek bir yanıt üretilemedi. Lütfen soruyu yeniden ifade edin."
                st.markdown("**Cevap:**")
                st.write(answer)

                src_docs = result.get("source_documents") or []
                if src_docs:
                    with st.expander("Kaynak parçalar"):
                        for i, d in enumerate(src_docs, 1):
                            page = d.metadata.get("page") if hasattr(d, "metadata") else None
                            title = f"Kaynak {i} (Sayfa {page})" if page else f"Kaynak {i}"
                            st.markdown(title)
                            st.text(d.page_content[:1000])
                else:
                    st.caption("Herhangi bir kaynak döndürülmedi.")
            except KeyError:
                st.error("Zincirden beklenen anahtarlar bulunamadı. Lütfen tekrar deneyin.")
            except Exception as exc:
                st.error(f"Sorgu çalıştırılırken hata: {exc}")
