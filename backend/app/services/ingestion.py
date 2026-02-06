from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import io

class IngestionService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_content: bytes, filename: str) -> List[Document]:
        pdf_reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean text (basic)
        text = self._clean_text(text)
        
        # Create chunks
        chunks = self.text_splitter.create_documents(
            texts=[text], 
            metadatas=[{"source": filename}]
        )
        return chunks

    def process_text(self, file_content: str, filename: str) -> List[Document]:
        text = self._clean_text(file_content)
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[{"source": filename}]
        )
        return chunks

    def _clean_text(self, text: str) -> str:
        # Basic cleaning: remove excessive whitespace
        return " ".join(text.split())
