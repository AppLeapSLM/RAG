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
    top_k: int = 5  # DEPRECATED for RAG selection — superseded by dynamic
                    # rerank selection below. Still the request-model default;
                    # no longer slices the reranked pool.
    neighbor_window: int = 0  # pull ±N adjacent chunks (0 = disabled)
    max_context_chars: int = 40000  # ~10K tokens hard cap sent to LLM

    # Dynamic rerank selection (relative-to-top band). Instead of a fixed
    # top_k slice, keep every chunk whose cross-encoder score is within
    # `rerank_relative_delta` (logit space) of the top-scoring chunk, bounded
    # to [rerank_floor, rerank_max_k]. This lets an enumeration question that
    # has many near-equally-relevant docs keep them all, while a precise
    # single-answer question keeps a tight set.
    #   - floor = 5 == the old fixed top_k, so the selected set is always a
    #     superset of the old top-5 → recall can only stay equal or improve
    #     (no regression on the existing baseline).
    #   - max_k caps over-fetch; max_context_chars (above) is the downstream
    #     physical guard during context assembly.
    #   - delta is in LOGIT units (the reranker's sigmoid output is inverted to
    #     logits before banding — see reranker._to_logit). Tuned on the eval:
    #     1.5 keeps precise single-answer questions tight at the floor (an exact
    #     match sits ~6+ logits above near-duplicates) while still capturing
    #     genuine enumeration clusters (e.g. 8 near-equal incident docs).
    rerank_floor: int = 5
    rerank_max_k: int = 15  # cap on the band. Bounds the latency/over-fetch tail
                            # on near-duplicate-heavy queries (where widening is
                            # mostly wasted). Verified: covers ENUM gold within
                            # rank 15 except ENUM-04's 12th doc (tie-break at 18
                            # → 11/12); CROSS-02 safe. Raise to 18 to keep ENUM-04
                            # whole if the latency tail is acceptable.
    rerank_relative_delta: float = 1.5

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
