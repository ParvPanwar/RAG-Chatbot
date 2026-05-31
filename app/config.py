from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Social Video Comparison API"
    app_env: str = "development"
    ollama_base_url: str = "https://your-ollama-server.example.com"
    ollama_chat_model: str = "gpt-oss:20b"
    ollama_api_key: str = ""
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    instagram_cookie_file: str = ""
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "video_transcripts_local"
    rag_top_k: int = 6
    max_chunks_per_video: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
