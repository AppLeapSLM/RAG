from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/appleap_rag"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "phi4"

    # Chunking (legacy — used by naive fallback only)
    chunk_size: int = 512
    chunk_overlap: int = 200

    # Parsing (Unstructured.io)
    parsing_strategy: str = "auto"  # "auto", "fast", "hi_res", "ocr_only"

    # Chunking (Unstructured.io / content-type-aware router)
    chunking_strategy: str = "auto"  # "auto", "by_title", "by_similarity", "naive"
    # Unstructured groups by headings up to this limit (Parse Big)
    chunk_max_characters: int = 30000
    chunk_new_after_n_chars: int = 29000
    chunk_combine_under_n_chars: int = 800
    # Semantic chunking — the real chunk size ceiling (Store Small)
    similarity_threshold: float = 0.75
    similarity_max_characters: int = 3000

    # File upload
    max_upload_size_mb: int = 50

    # Retrieval
    top_k: int = 5
    neighbor_window: int = 0  # pull ±N adjacent chunks (0 = disabled)
    max_context_chars: int = 40000  # ~10K tokens hard cap sent to LLM

    # Embedding dimension (Nomic produces 768-dim vectors)
    embedding_dim: int = 768

    # Google Drive connector
    google_drive_credentials_path: str = ""

    model_config = {"env_prefix": "APPLEAP_"}


settings = Settings()
