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

    # File upload (corpus — admin-only, CLI/connectors)
    max_upload_size_mb: int = 200

    # Chat attachments (conversation-scoped, user-facing)
    max_chat_upload_mb: int = 5
    inline_attachment_threshold_kb: int = 20  # ≤ this inlined, > this chunked

    # Admin token for /ingest/file (empty = no gate, dev only)
    admin_token: str = ""

    # Retrieval
    top_k: int = 5
    neighbor_window: int = 0  # pull ±N adjacent chunks (0 = disabled)
    max_context_chars: int = 40000  # ~10K tokens hard cap sent to LLM

    # Embedding dimension (Nomic produces 768-dim vectors)
    embedding_dim: int = 768

    # Nango (self-hosted) — webhook receiver verifies HMAC against this secret.
    # Empty = signature check is skipped (dev only — DO NOT use in production).
    nango_signing_secret: str = ""

    # Auth: HS256 JWT signing key. Empty = auth refuses to issue/validate tokens
    # (every protected endpoint returns 503). Generate with: openssl rand -hex 64
    jwt_secret: str = ""
    jwt_lifetime_hours: int = 24
    # bcrypt rounds. 12 ≈ 250ms on a 2 GHz core. Bumping is safe; lowering is not.
    bcrypt_rounds: int = 12
    # Minimum password length on user creation.
    password_min_length: int = 12

    model_config = {"env_prefix": "APPLEAP_"}


settings = Settings()
