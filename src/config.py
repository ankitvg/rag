import os
from dotenv import load_dotenv
from chromadb.config import Settings

# Load environment variables
load_dotenv()

class Config:
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
    DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', '1000'))
    DEFAULT_OVERLAP = int(os.getenv('DEFAULT_OVERLAP', '200'))
    
    CHROMA_SETTINGS = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )

    @classmethod
    def get_ollama_embeddings_url(cls):
        return f"{cls.OLLAMA_HOST}/api/embeddings"