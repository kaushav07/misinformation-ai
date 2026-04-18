"""
Central configuration — reads from .env automatically.
All services import settings from here, never from os.environ directly.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Anthropic
    anthropic_api_key: str = ""

    # Google APIs
    google_fact_check_api_key: str = ""
    serper_api_key: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    similarity_threshold: float = 0.85
    max_claims_per_request: int = 5

    # Models
    nli_model: str = "facebook/bart-large-mnli"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    whisper_model: str = "openai/whisper-base"
    ai_detect_model: str = "roberta-base-openai-detector"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
