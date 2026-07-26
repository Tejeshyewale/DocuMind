# DocuMind — Multi-Document RAG Q&A !!

Upload one or more PDFs, ask questions in plain language, and get answers grounded
in the documents themselves — with exact source citations (`filename.pdf, page X`)
instead of a generic LLM guess.

> 🔗 **Live demo:** https://tejesh-rag-assistant417.streamlit.app/

**Stack:** Python · LangChain · Groq (Llama 3.3) · Local HuggingFace Embeddings · FAISS · Streamlit

## Features

- **Multi-document Q&A** — upload several PDFs in one session and ask questions across all of them at once.
- **Source citations** — every answer shows exactly which file and page it came from.
- **Smart caching** — each PDF is hashed (MD5); re-uploading a previously processed file loads the cached FAISS index instead of re-embedding it.
- **Session controls** — "Clear Chat" wipes conversation history without touching loaded documents; "Reset Session" clears everything.
- **Transparent retrieval** — an expandable panel shows the exact chunks retrieved for each answer, useful for explaining the pipeline.
- **Graceful error handling** — corrupted or invalid PDFs show a friendly warning instead of crashing the app.
- **Zero-cost to run** — Groq's free API (no card required) for generation, and a local HuggingFace embedding model (no API key at all) for embeddings.

## How it works

```
PDF Upload(s)
   ↓
PyPDFLoader (extract text per page, tag with source filename)
   ↓
RecursiveCharacterTextSplitter (chunk into ~1000-char overlapping pieces)
   ↓
Local HuggingFace embedding model (chunk -> vector, runs on your machine, no API key)
   ↓
FAISS (per-document index, cached on disk by file hash; merged into a session-wide index)
   ↓
User question -> top-k relevant chunks retrieved across all loaded documents
   ↓
Groq's Llama 3.3 model answers using ONLY those retrieved chunks as context
   ↓
Answer + "filename.pdf, page X" citations shown in Streamlit
```

This is Retrieval-Augmented Generation (RAG): instead of asking the LLM to answer
from memory, you hand it the relevant slice of *your* documents, which cuts down
hallucination and lets you cite an exact source.

## Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Tejeshyewale/DocuMind-.git
   cd DocuMind-
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   First install downloads the local embedding model (~100–200MB) — only happens once.

4. **Get a free Groq API key:** https://console.groq.com/keys (no credit card required).

5. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your GROQ_API_KEY
   ```

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`.

## Folder structure

```
DocuMind/
├── app.py                    # Streamlit UI — sidebar uploads, chat, session controls
├── requirements.txt
├── .env.example
├── .streamlit/config.toml     # Theme/UI configuration
├── utils/
│   ├── pdf_loader.py           # PDF -> per-page text, tagged with source filename
│   ├── embeddings.py           # Local HuggingFace embedding model
│   ├── vector_store.py         # Chunking + FAISS build/load/merge + MD5 caching
│   └── rag_pipeline.py         # Retrieval + Groq answer generation + citations
├── data/                      # (gitignored) place to drop sample PDFs locally
└── faiss_index/               # (gitignored) cached FAISS indexes, keyed by file hash
```

## Roadmap

- [x] Multiple PDF upload + cross-document search
- [x] Persist FAISS index per-document (skip re-embedding on re-upload)
- [x] Polished, responsive Streamlit UI
- [ ] Move the RAG logic behind a FastAPI backend, keep Streamlit as a thin client
- [ ] Add simple auth (so it's not a fully open demo)
- [ ] Containerize with Docker, deploy backend + frontend separately
- [ ] Evaluate retrieval quality with RAGAS to back up accuracy claims

## Notes

- Groq currently serves Llama 3.3 70B for free with generous rate limits — no payment
  info needed anywhere in this project.
- Embeddings run locally via `sentence-transformers`, so there's no per-query cost or
  external API call for that step...!!

  STAR the repo
