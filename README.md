# LLM-Powered Document Question Answering System (RAG)

## Overview
A Retrieval-Augmented Generation (RAG) based intelligent document question answering system. Users can upload documents (PDF/Text) and receive context-aware answers grounded in the source material.

## Architecture
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Orchestration**: LangChain
- **Embeddings**: SentenceTransformers
- **Vector Store**: FAISS
- **LLM**: OpenAI / Vertex AI

## Setup

### Backend
1. Navigate to `backend/`
2. Create virtual environment: `python -m venv venv`
3. Activate: `.\venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `uvicorn app.main:app --reload`

### Frontend
1. Navigate to `frontend/`
2. Create virtual environment (optional, or reuse backend's if appropriate, but keeping separate is cleaner)
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

### Website Link 
https://document-system.streamlit.app/

## Features
- Document Ingestion (PDF/Text)
- Text Chunking & Cleaning
- Vector Similarity Search
- Context-Aware Answer Generation
