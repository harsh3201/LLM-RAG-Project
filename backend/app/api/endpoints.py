from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, SourceDocument, UploadResponse
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStoreService
from app.services.llm import LLMService
from typing import List

router = APIRouter()

# Initialize Services (Singleton pattern for simplicity)
ingestion_service = IngestionService()
vector_store_service = VectorStoreService()
llm_service = LLMService()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    content = await file.read()
    
    try:
        if file.filename.endswith(".pdf"):
            chunks = ingestion_service.process_pdf(content, file.filename)
        else:
            # Assume text
            text_content = content.decode("utf-8", errors="ignore")
            chunks = ingestion_service.process_text(text_content, file.filename)
            
        vector_store_service.add_documents(chunks)
        
        return UploadResponse(
            filename=file.filename,
            status="Processed and indexed successfully",
            chunks=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Retrieve context
        docs = vector_store_service.search(request.query, k=request.top_k)
        
        # Generate Answer
        answer = llm_service.generate_response(request.query, docs, provider=request.model_provider)
        
        # Format sources
        sources = [
            SourceDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in docs
        ]
        
        return QueryResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
