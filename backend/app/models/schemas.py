from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    model_provider: str = "gemini" # gemini or openai

class SourceDocument(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

class UploadResponse(BaseModel):
    filename: str
    status: str
    chunks: int
