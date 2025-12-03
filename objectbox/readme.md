# 📄 ObjectBox RAG Demo (Groq + LangChain 1.x)

A lightweight Retrieval-Augmented Generation (RAG) application built using:

- **LangChain 1.x (LCEL)**
- **Groq Llama 3.3 70B**
- **ObjectBox VectorStore**
- **HuggingFace BGE Embeddings**
- **Streamlit UI**

The app loads PDF documents, indexes them into ObjectBox using embeddings, and answers user questions using Groq with context-aware retrieval.

---

## 🚀 Features

- PDF ingestion with `PyPDFDirectoryLoader`
- CPU-friendly embeddings: `BAAI/bge-small-en-v1.5`
- ObjectBox vectorstore for fast similarity search
- Modern RAG chain using LCEL (LangChain 1.x)
- Clean Streamlit interface with real-time inference
- Automatic indexing with caching for speed

---

## 1. Install dependencies
cd objectbox
pip install -r requirements.txt

2. Set your Groq API Key

Create a .env file:

GROQ_API_KEY=your_api_key_here

3. Start the app
streamlit run app.py

🛠 Tech Stack
| Component    | Library            |
| ------------ | ------------------ |
| UI           | Streamlit          |
| LLM          | Groq Llama 3.3 70B |
| Embeddings   | HuggingFace BGE    |
| Vector Store | ObjectBox          |
| Framework    | LangChain 1.x LCEL |
