from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/appleap_rag"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "phi4"
    llm_num_ctx: int = 16384  # Phi-4's full window; Ollama default is 2048

    # Parsing (Unstructured.io — used only as file parser, not for chunking)
    parsing_strategy: str = "auto"  # "auto", "fast", "hi_res", "ocr_only"

    # Chunking (recursive character splitter)
    chunk_size: int = 3000          # hard max characters per chunk
    chunk_overlap: int = 200        # characters of overlap between consecutive chunks

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
