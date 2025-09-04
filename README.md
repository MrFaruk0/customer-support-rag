# PDF-RAG-CHATBOT (LOCAL)

Çalıştırma talimatları:

1. Gerekli paketleri kurun:

```
pip install -r requirements.txt
```

2. Uygulamayı başlatın:

```
streamlit run app.py
```

Özellikler:
- Streamlit arayüzü ile PDF yükleme ve soru sorma
- pdfplumber ile metin çıkarma ve parçalama (chunking)
- LangChain ile FAISS vektör veritabanı ve RetrievalQA zinciri
- OpenRouter API (ücretsiz model: meta-llama/llama-3.1-8b-instruct:free) ile güçlü yanıtlar
- Hata yönetimi: boş PDF, çok uzun soru, eksik anahtarlar

OpenRouter kurulumu:
- `.env` dosyası oluşturun ve anahtarı ekleyin:
```
OPENROUTER_API_KEY=your_key_here
```

Notlar:
- OpenRouter, OpenAI uyumlu bir API sunar. Doğru taban URL `https://openrouter.ai/api/v1` şeklindedir. Tarayıcıda bu URL doğrudan bir model değildir; kod içinde istemci `base_url` olarak kullanılır.
