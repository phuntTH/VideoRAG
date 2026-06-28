import os
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    
    STORAGE_DIR: Path = BASE_DIR.parent / "storage"
    AUDIO_CACHE_DIR: Path = BASE_DIR.parent / "storage" / "audio_cache"
    FAISS_INDICES_DIR: Path = BASE_DIR.parent / "storage" / "faiss_indices"
    BM25_INDICES_DIR: Path = BASE_DIR.parent / "storage" / "bm25_indices"
    
    model_config = ConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

settings.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.FAISS_INDICES_DIR.mkdir(parents=True, exist_ok=True)
settings.BM25_INDICES_DIR.mkdir(parents=True, exist_ok=True)