from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings
import os
from typing import List
from langchain_core.documents import Document

class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vector_store = None
        self._load_index()

    def _load_index(self):
        if os.path.exists(settings.VECTOR_STORE_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    settings.VECTOR_STORE_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # Local dev env, this is acceptable
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None
        else:
            self.vector_store = None

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
            
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        self.vector_store.save_local(settings.VECTOR_STORE_PATH)

    def search(self, query: str, k: int = 3) -> List[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
