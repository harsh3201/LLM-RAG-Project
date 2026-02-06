import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(None, override=True) # Try default first
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv(env_path, override=True) # Try explicit path if missing

class Settings:
    PROJECT_NAME: str = "RAG Document Q&A"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_PATH = os.path.join(DATA_DIR, "faiss_index")
    
    # Model Config
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo" # Default, can be configured
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

settings = Settings()

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
