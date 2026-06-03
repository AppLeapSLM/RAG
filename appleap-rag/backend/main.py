import asyncio
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import (
    AuthedUser,
    DUMMY_HASH,
    encode_token,
    get_user_conversation,
    hash_password,
    require_admin,
    require_user,
    verify_password,
)
from backend.chunking.chunker import chunk_parsed_document_async
from backend.chunking.dispatch import process_file
from backend.config import settings
from backend.db.connection import async_session, engine, get_session
from backend.db.models import Base, Chunk, Conversation, ConversationInlineAttachment, Document, Message, User
from backend.embedding.embedder import embed_batch
from backend.generation import stream_registry
from backend.generation.llm import generate_answer_stream, rewrite_query
from backend.parsing.parser import parse_text
from backend.retrieval.vector_search import search
from backend.webhooks.nango import router as nango_webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables + pgvector extension on startup
    async with engine.begin() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.run_sync(Base.metadata.create_all)

        # Migrate documents table: add source_type + conversation_id for chat attachments
        await conn.execute(__import__("sqlalchemy").text("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS source_type VARCHAR(32) NOT NULL DEFAULT 'corpus'
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS conversation_id VARCHAR
                REFERENCES conversations(id) ON DELETE CASCADE
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            CREATE INDEX IF NOT EXISTS idx_documents_conv_id
            ON documents(conversation_id)
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'ck_documents_source_type_conv_id'
                ) THEN
                    ALTER TABLE documents ADD CONSTRAINT ck_documents_source_type_conv_id
                    CHECK (
                        (source_type = 'corpus' AND conversation_id IS NULL)
                        OR (source_type = 'attachment' AND conversation_id IS NOT NULL)
                    );
                END IF;
            END $$
        """))

        # Full-text search: add tsvector column + GIN index + auto-populate trigger
        await conn.execute(__import__("sqlalchemy").text("""
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
            ON chunks USING gin(search_vector)
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            CREATE OR REPLACE FUNCTION chunks_search_vector_update() RETURNS trigger AS $$
            BEGIN
                NEW.search_vector :=
                    setweight(to_tsvector('english', NEW.content), 'A') ||
                    setweight(to_tsvector('english', coalesce(NEW.metadata->>'section', '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(NEW.metadata->>'title', '')), 'C');
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            DROP TRIGGER IF EXISTS chunks_search_vector_trigger ON chunks
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            CREATE TRIGGER chunks_search_vector_trigger
            BEFORE INSERT OR UPDATE ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_search_vector_update()
        """))

        # GIN index on chunks.metadata for ACL pre-filtering at query time
        # (filter on metadata->'acl' against user identity + cached groups).
        await conn.execute(__import__("sqlalchemy").text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
            ON chunks USING gin(metadata jsonb_path_ops)
        """))

        # Auth rollout: one-shot legacy wipe.
        # Detected by: conversations table exists but owner_user_id column does NOT.
        # This runs once on the first deploy carrying auth, then never again.
        # Cascades through messages, conversation_inline_attachments, and
        # documents(source_type='attachment').
        await conn.execute(__import__("sqlalchemy").text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'conversations'
                ) AND NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'conversations' AND column_name = 'owner_user_id'
                ) THEN
                    DELETE FROM conversations;
                    ALTER TABLE conversations
                        ADD COLUMN owner_user_id VARCHAR
                        REFERENCES users(id) ON DELETE SET NULL;
                    CREATE INDEX IF NOT EXISTS idx_conversations_owner_user_id
                        ON conversations(owner_user_id);
                END IF;
            END $$
        """))

        # Partial index on jobs.run_after for the worker hot path:
        # SELECT ... WHERE locked_at IS NULL AND run_after <= now() ORDER BY run_after.
        await conn.execute(__import__("sqlalchemy").text("""
            CREATE INDEX IF NOT EXISTS idx_jobs_run_after_unlocked
            ON jobs(run_after)
            WHERE locked_at IS NULL
        """))

        # Streaming-generation columns. Added in V12 alongside SSE /query.
        # Existing messages predate the column, so default everything to
        # 'completed' on backfill.
        await conn.execute(__import__("sqlalchemy").text("""
            ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'completed'
        """))
        await conn.execute(__import__("sqlalchemy").text("""
            ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS error_text TEXT
        """))

        # Crash recovery: any assistant message still marked 'streaming' from
        # a previous process is orphaned (in-memory stream registry is gone).
        # Mark them as errored so clients see a clear signal instead of an
        # eternal spinner.
        await conn.execute(__import__("sqlalchemy").text("""
            UPDATE messages
            SET status = 'error',
                error_text = COALESCE(error_text, 'Generation interrupted by server restart')
            WHERE status = 'streaming'
        """))
    yield
    await engine.dispose()


app = FastAPI(title="AppLeap RAG", version="0.2.0", lifespan=lifespan)
app.include_router(nango_webhook_router)

ACTIVITY_STAMP = Path("/tmp/appleap-last-request")


@app.middleware("http")
async def track_activity(request: Request, call_next):
    """Touch a stamp file on every request so the idle-shutdown script knows we're active."""
    try:
        ACTIVITY_STAMP.touch()
    except OSError:
        pass
    return await call_next(request)


# ── Request / Response schemas ──────────────────────────────────────


class IngestRequest(BaseModel):
    title: str
    content: str
    source: str = "manual"
    metadata: dict = {}


class IngestResponse(BaseModel):
    document_id: str
    chunks_stored: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = settings.top_k
    conversation_id: str | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict]
    # The id of an assistant message currently being generated server-side
    # for this conversation, or None. Frontend uses this to reconnect to a
    # live stream when the user navigates back to a thinking conversation.
    streaming_message_id: str | None = None


class ConversationUpdate(BaseModel):
    title: str


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "user"  # 'admin' or 'user'


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    active: bool


ALLOWED_EXTENSIONS = {
    # Prose / office (via Unstructured.io)
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".md", ".markdown", ".txt", ".html", ".htm", ".rst", ".xml",
    ".eml", ".rtf",
    # Tabular — row-per-chunk pipe format (.xlsx/.xls via Excel path, .csv/.tsv via CSV path)
    ".csv", ".tsv", ".xlsx", ".xls",
    # IaC / config (via tree-sitter)
    ".tf", ".tfvars", ".hcl", ".yaml", ".yml", ".json", ".pp",
    ".conf",  # misc text config files — fall through to prose path
    # Code (via tree-sitter)
    ".py", ".pyi", ".go", ".rb", ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx", ".sh", ".bash",
}

# Extensionless files commonly carried at repo roots. Matched by exact basename
# (case-sensitive) since these conventions are case-specific.
ALLOWED_FILENAMES = {
    "Puppetfile", "Dockerfile", "Makefile", "Gemfile", "Rakefile", "Vagrantfile",
}


def require_admin_token(x_admin_token: str | None = Header(default=None)):
    """Gate for admin-only endpoints (corpus ingestion + first-user bootstrap).

    If `admin_token` setting is empty, the gate is open (dev convenience).
    Otherwise the caller must send a matching `X-Admin-Token` header.
    """
    if not settings.admin_token:
        return
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Unauthorized")


# ── Auth endpoints ──────────────────────────────────────────────────


@app.post("/auth/login", response_model=LoginResponse)
async def auth_login(
    req: LoginRequest,
    session: AsyncSession = Depends(get_session),
):
    """Exchange email + password for a Bearer JWT.

    Constant-time response: if the email isn't registered, we still run a
    bcrypt verify against a dummy hash so the response time matches a real
    failed-password attempt. Defeats email-enumeration via timing.
    """
    from sqlalchemy import select as sa_select

    if not settings.jwt_secret:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured on this server",
        )

    email_norm = req.email.strip().lower()

    row = await session.execute(
        sa_select(User).where(User.email == email_norm)
    )
    user = row.scalar_one_or_none()

    # Always run bcrypt verify (against the real hash if user exists, else
    # against DUMMY_HASH). Both arms take ~250ms.
    target_hash = user.hashed_password if user else DUMMY_HASH
    password_ok = verify_password(req.password, target_hash)

    # Uniform response on ANY failure mode (no user, wrong password,
    # disabled account). Don't disclose which one.
    if not user or not password_ok or not user.active:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user.last_login_at = datetime.now(timezone.utc)
    await session.commit()

    token, expires_in = encode_token(user_id=user.id, email=user.email, role=user.role)
    return LoginResponse(access_token=token, expires_in=expires_in)


@app.post(
    "/auth/users",
    response_model=UserResponse,
    dependencies=[Depends(require_admin_token)],
)
async def auth_create_user(
    req: CreateUserRequest,
    session: AsyncSession = Depends(get_session),
):
    """Admin-only user creation. Gated by X-Admin-Token (the existing admin
    secret is the bootstrap path; once a real admin user exists, we'll add
    a Bearer-admin path too).
    """
    from sqlalchemy import select as sa_select

    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")
    if len(req.password) < settings.password_min_length:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {settings.password_min_length} characters",
        )
    if len(req.password.encode("utf-8")) > 72:
        # bcrypt silently truncates beyond 72 bytes; reject up front so the
        # full password is what gets verified.
        raise HTTPException(
            status_code=400,
            detail="Password must be at most 72 bytes",
        )

    email_norm = req.email.strip().lower()
    if "@" not in email_norm or " " in email_norm:
        raise HTTPException(status_code=400, detail="Invalid email")

    existing = await session.execute(
        sa_select(User).where(User.email == email_norm)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=email_norm,
        hashed_password=hash_password(req.password),
        role=req.role,
        active=True,
    )
    session.add(user)
    await session.commit()
    return UserResponse(id=user.id, email=user.email, role=user.role, active=user.active)


@app.get("/auth/me", response_model=UserResponse)
async def auth_me(user: AuthedUser = Depends(require_user)):
    """Echo the authenticated user back. Useful for the frontend to
    verify the cached token is still valid.

    Note: returns claims from the JWT, not a fresh DB lookup. If you've been
    disabled (active=false), this still echoes active=true until your token
    expires (≤ jwt_lifetime_hours).
    """
    return UserResponse(id=user.id, email=user.email, role=user.role, active=True)


# ── Endpoints ───────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, session: AsyncSession = Depends(get_session)):
    """Ingest a text document: parse → chunk → embed → store in pgvector."""
    # 1. Parse text into document model
    parsed_doc = parse_text(
        req.content, extra_metadata={**req.metadata, "title": req.title}
    )

    # 2. Chunk using content-type-aware pipeline
    chunks = await chunk_parsed_document_async(parsed_doc)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest")

    # 3. Create document record
    doc = Document(source=req.source, title=req.title, metadata_=req.metadata)
    session.add(doc)
    await session.flush()

    # 4. Embed all chunks in one batch call
    chunk_texts = [c["text"] for c in chunks]
    embeddings = await embed_batch(chunk_texts)

    # 5. Store chunks with embeddings
    for i, (chunk_data, emb) in enumerate(zip(chunks, embeddings)):
        chunk = Chunk(
            document_id=doc.id,
            content=chunk_data["text"],
            chunk_index=i,
            embedding=emb,
            metadata_={
                **chunk_data["metadata"],
                "element_types": chunk_data["element_types"],
            },
        )
        session.add(chunk)

    await session.commit()
    return IngestResponse(document_id=doc.id, chunks_stored=len(chunks))


def _classify_extension(filename: str) -> str:
    """Return the canonical extension, handling .tf.json as a two-segment suffix."""
    lower = filename.lower()
    if lower.endswith(".tf.json"):
        return ".tf.json"
    return Path(filename).suffix.lower()


def _is_file_supported(filename: str) -> bool:
    """Accept files by either known extension or known extensionless basename
    (e.g. Puppetfile, Dockerfile)."""
    if _classify_extension(filename) in ALLOWED_EXTENSIONS:
        return True
    return Path(filename).name in ALLOWED_FILENAMES


@app.post(
    "/ingest/file",
    response_model=IngestResponse,
    dependencies=[Depends(require_admin_token)],
)
async def ingest_file(
    file: UploadFile = File(...),
    source: str = Form(default="upload"),
    metadata_json: str = Form(default="{}"),
    session: AsyncSession = Depends(get_session),
):
    """Corpus ingest (admin-only): parse via Unstructured → chunk → embed → store."""
    # 1. Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = _classify_extension(file.filename)
    if not _is_file_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type '{ext or file.filename}' is not supported. "
                f"Supported types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                f" or filenames: {', '.join(sorted(ALLOWED_FILENAMES))}"
            ),
        )

    try:
        extra_metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata_json")

    # 2. Save to temp file (Unstructured needs a file path)
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. Please upload a file less than "
                f"{settings.max_upload_size_mb}MB."
            ),
        )

    # .tf.json needs a two-segment suffix so tempfile preserves it for classification
    suffix = ".tf.json" if file.filename.lower().endswith(".tf.json") else ext
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, content)
        os.close(tmp_fd)

        # 3. Parse + chunk via dispatch (routes prose → Unstructured, structured → tree-sitter)
        meta_in = {**extra_metadata, "title": file.filename, "source": source}
        try:
            chunks, doc_meta = await process_file(
                tmp_path, meta_in, display_name=file.filename
            )
        except ValueError as exc:
            logger.info("ingest_file no-content file=%s err=%s", file.filename, exc)
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from this file. If it's an image-only PDF or scan, try converting to text/markdown first.",
            ) from exc
        if not chunks:
            raise HTTPException(
                status_code=400, detail="No content extracted from file"
            )

        # 4. Create document record
        doc = Document(
            source=source,
            title=file.filename,
            metadata_={
                **extra_metadata,
                **doc_meta,
            },
        )
        session.add(doc)
        await session.flush()

        # 5. Embed all chunk texts in one batch
        chunk_texts = [c["text"] for c in chunks]
        embeddings = await embed_batch(chunk_texts)

        # 6. Store chunks with embeddings
        for i, (chunk_data, emb) in enumerate(zip(chunks, embeddings)):
            chunk = Chunk(
                document_id=doc.id,
                content=chunk_data["text"],
                chunk_index=i,
                embedding=emb,
                metadata_={
                    **chunk_data["metadata"],
                    "element_types": chunk_data["element_types"],
                },
            )
            session.add(chunk)

        await session.commit()
        return IngestResponse(document_id=doc.id, chunks_stored=len(chunks))

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _persist_loop(state: stream_registry.StreamState) -> None:
    """Snapshot accumulated_text to DB every ~1.5s while generation runs.
    Exits when state.done_event is set (the main flow does the final save
    with the terminal status). The 1.5s cadence means a server crash loses
    at most ~1.5s of generated tokens."""
    last_persisted_len = 0
    while True:
        try:
            await asyncio.wait_for(state.done_event.wait(), timeout=1.5)
            return  # done — main flow handles the final save
        except asyncio.TimeoutError:
            pass
        text = state.accumulated_text
        if len(text) == last_persisted_len:
            continue
        try:
            async with async_session() as s:
                row = await s.get(Message, state.message_id)
                if row is not None:
                    row.content = text
                    await s.commit()
            last_persisted_len = len(text)
        except Exception:
            # Don't crash the loop on a transient DB error; the next tick
            # will retry, and the final-save in _run_generation is the
            # ultimate source of truth.
            logger.exception("persist_loop write failed message=%s", state.message_id)


async def _run_generation(
    state: stream_registry.StreamState,
    question: str,
    chunks: list,
    history: list[dict],
    inline_attachments: list,
    rewritten_query: str,
) -> None:
    """Background task: stream tokens from Ollama into the registry, then
    persist the final answer to the assistant message row. Runs independently
    of the HTTP request that started it — survives client disconnect."""
    persist_task = asyncio.create_task(_persist_loop(state))
    try:
        async for delta in generate_answer_stream(
            question, chunks, history=history, inline_attachments=inline_attachments,
        ):
            stream_registry.push_delta(state, delta)

        async with async_session() as s:
            row = await s.get(Message, state.message_id)
            if row is not None:
                row.content = state.accumulated_text
                row.status = "completed"
                if rewritten_query != question:
                    row.metadata_ = {"rewritten_query": rewritten_query}
                await s.commit()
        stream_registry.finish(state)
    except Exception as exc:
        logger.exception("generation_failed message=%s", state.message_id)
        try:
            async with async_session() as s:
                row = await s.get(Message, state.message_id)
                if row is not None:
                    row.content = state.accumulated_text  # save partial
                    row.status = "error"
                    row.error_text = str(exc)
                    await s.commit()
        except Exception:
            logger.exception("error_persist_failed message=%s", state.message_id)
        stream_registry.finish(state, error=str(exc))
    finally:
        # done_event is set by finish(); persist_loop exits on its own.
        try:
            await persist_task
        except Exception:
            logger.exception("persist_task failed message=%s", state.message_id)
        # Keep terminal state in the registry briefly so late subscribers
        # can get the synthesized done event, then drop it.
        await asyncio.sleep(30)
        stream_registry.remove(state.message_id)


async def _stream_clarification(
    conversation_id: str,
    message_id: str,
    text: str,
) -> AsyncIterator[str]:
    """SSE generator for a clarifying question. The text was produced by the
    rewrite call and is already complete — we pace delivery at ~Phi-4's
    native rate (~40 tok/s) so the UX is indistinguishable from a real
    generation stream."""
    yield _sse("start", {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "sources": [],
    })
    chunk_size = 4  # ~1 token's worth of characters
    delay = 0.025   # ~40 tok/s
    for i in range(0, len(text), chunk_size):
        yield _sse("delta", {"text": text[i:i + chunk_size]})
        await asyncio.sleep(delay)
    yield _sse("done", {})


async def _stream_existing(state: stream_registry.StreamState) -> AsyncIterator[str]:
    """SSE generator that resumes an in-flight stream. Sends a snapshot of
    accumulated text first, then drains new deltas until done/error."""
    text, status, q = stream_registry.subscribe_with_snapshot(state)
    try:
        yield _sse("snapshot", {
            "conversation_id": state.conversation_id,
            "message_id": state.message_id,
            "sources": state.sources,
            "text": text,
            "status": status,
        })
        while True:
            event = await q.get()
            if event["type"] == "delta":
                yield _sse("delta", {"text": event["text"]})
            elif event["type"] == "done":
                err = event.get("error")
                if err:
                    yield _sse("error", {"error": err})
                else:
                    yield _sse("done", {})
                return
    finally:
        stream_registry.unsubscribe(state, q)


async def _stream_fresh(
    state: stream_registry.StreamState,
) -> AsyncIterator[str]:
    """SSE generator for a brand-new /query stream. Emits `start` (with
    conversation_id, message_id, sources) first, then deltas as the
    background producer pushes them."""
    text, status, q = stream_registry.subscribe_with_snapshot(state)
    try:
        yield _sse("start", {
            "conversation_id": state.conversation_id,
            "message_id": state.message_id,
            "sources": state.sources,
        })
        # If anything was generated before subscribe (race-tolerant — usually
        # empty since we subscribe before the producer task gets to run),
        # flush it as a delta so the client renders it.
        if text:
            yield _sse("delta", {"text": text})
        if status != "streaming":
            err = state.error_text if status == "error" else None
            if err:
                yield _sse("error", {"error": err})
            else:
                yield _sse("done", {})
            return
        while True:
            event = await q.get()
            if event["type"] == "delta":
                yield _sse("delta", {"text": event["text"]})
            elif event["type"] == "done":
                err = event.get("error")
                if err:
                    yield _sse("error", {"error": err})
                else:
                    yield _sse("done", {})
                return
    finally:
        stream_registry.unsubscribe(state, q)


@app.post("/query")
async def query(
    req: QueryRequest,
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Stream a grounded answer back as Server-Sent Events.

    Event sequence on success:
      event: start    data: {conversation_id, message_id, sources}
      event: delta    data: {text}             (repeated)
      event: done     data: {}

    On error mid-stream:
      event: error    data: {error: "..."}

    The HTTP request can disconnect at any time; generation continues
    server-side and is persisted. Reconnect via GET /messages/{id}/stream.

    Ownership: if `conversation_id` is provided, the caller MUST own it.
    """
    from sqlalchemy import select as sa_select

    # 1. Resolve or create conversation (ownership enforced)
    if req.conversation_id:
        conv = await get_user_conversation(req.conversation_id, user, session)
    else:
        conv = Conversation(title="New Chat", owner_user_id=user.id)
        session.add(conv)
        await session.flush()

    # 2. Load conversation history (all messages, chronological)
    rows = (
        await session.execute(
            sa_select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.created_at)
        )
    ).scalars().all()
    history = [{"role": m.role, "content": m.content} for m in rows]

    # 2b. Load inline attachments for this conversation (prepended to context).
    inline_rows = (
        await session.execute(
            sa_select(ConversationInlineAttachment)
            .where(ConversationInlineAttachment.conversation_id == conv.id)
            .order_by(ConversationInlineAttachment.created_at)
        )
    ).scalars().all()
    inline_attachments = list(inline_rows)
    if inline_attachments:
        total_inline_bytes = sum(a.size_bytes for a in inline_attachments)
        # 40960 bytes ≈ 10K tokens — start of the danger zone for 16K num_ctx
        # when combined with history + retrieved context. See
        # project_inline_attachment_overflow memory.
        if total_inline_bytes > 40 * 1024:
            logger.warning(
                "inline_attachments large conv=%s count=%d total_bytes=%d",
                conv.id, len(inline_attachments), total_inline_bytes,
            )

    # 2c. Filenames of chunked attachments for this conv. Titles only — the
    #     full chunks still go through retrieval. We pass these into the
    #     rewriter so it can resolve "this doc" / "the attached file" to a
    #     real filename before retrieval runs.
    chunked_attachment_rows = (
        await session.execute(
            sa_select(Document)
            .where(Document.source_type == "attachment")
            .where(Document.conversation_id == conv.id)
            .order_by(Document.created_at)
        )
    ).scalars().all()
    attachment_filenames = (
        [a.filename for a in inline_attachments]
        + [d.title for d in chunked_attachment_rows]
    )

    # 3. Decide: answer the follow-up, or ask a clarifying question?
    #    Rewrite fires when there's history OR attached files — both
    #    introduce references the model needs to resolve before retrieval.
    #    Cold path with neither stays LLM-free.
    mode, payload = await rewrite_query(
        req.question, history, attachment_filenames=attachment_filenames,
    )

    if mode == "clarify":
        # No retrieval, no background generation. Persist the user message
        # and an assistant message containing the clarification, then
        # fake-stream the clarification text back so the UX matches a
        # normal answer.
        clarification = payload
        assistant_msg_id = str(uuid.uuid4())
        user_msg = Message(
            conversation_id=conv.id,
            role="user",
            content=req.question,
        )
        assistant_msg = Message(
            id=assistant_msg_id,
            conversation_id=conv.id,
            role="assistant",
            content=clarification,
            status="completed",
            model_used=settings.llm_model,
            sources=[],
            metadata_={"clarification": True},
        )
        session.add(user_msg)
        session.add(assistant_msg)
        if conv.title == "New Chat":
            conv.title = req.question[:100]
        conv.updated_at = datetime.now(timezone.utc)
        await session.commit()

        return StreamingResponse(
            _stream_clarification(conv.id, assistant_msg_id, clarification),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    search_query = payload

    # 4. Retrieve relevant chunks
    results = await search(
        search_query,
        session,
        top_k=req.top_k,
        conversation_id=conv.id,
        user_email=user.email,
    )

    # 5. Build source metadata
    sources = [
        {
            "chunk_id": r.id,
            "document_id": r.document_id,
            "content_preview": r.content[:200],
            "chunk_index": r.chunk_index,
        }
        for r in results
    ]

    # 6. Persist user message + empty assistant message (status='streaming')
    #    immediately so the row exists for late subscribers / resume.
    assistant_msg_id = str(uuid.uuid4())
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=req.question,
    )
    assistant_msg = Message(
        id=assistant_msg_id,
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="streaming",
        model_used=settings.llm_model,
        sources=sources,
    )
    session.add(user_msg)
    session.add(assistant_msg)

    if conv.title == "New Chat":
        conv.title = req.question[:100]
    conv.updated_at = datetime.now(timezone.utc)
    await session.commit()

    # 7. Register the stream + spawn background generation. The task is
    #    intentionally NOT tied to the request lifecycle — it survives
    #    client disconnect so the answer always completes.
    state = stream_registry.register(assistant_msg_id, conv.id, sources)
    asyncio.create_task(_run_generation(
        state=state,
        question=req.question,
        chunks=results,
        history=history,
        inline_attachments=inline_attachments,
        rewritten_query=search_query,
    ))

    return StreamingResponse(
        _stream_fresh(state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx-style buffering
        },
    )


@app.get("/messages/{message_id}/stream")
async def stream_message(
    message_id: str,
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Reconnect to an in-flight stream, or replay a finished message.

    Ownership: 404 unless the message's conversation belongs to the caller.

    If the message is still being generated server-side, subscribe to live
    deltas. Otherwise emit a single snapshot + done/error so the frontend
    has one code path."""
    msg = await session.get(Message, message_id)
    if msg is None:
        raise HTTPException(status_code=404, detail="Message not found")

    conv = await session.get(Conversation, msg.conversation_id)
    if conv is None or conv.owner_user_id != user.id:
        raise HTTPException(status_code=404, detail="Message not found")

    state = stream_registry.get(message_id)
    if state is not None:
        return StreamingResponse(
            _stream_existing(state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    # Not in registry — generation is over (clean, errored, or lost to
    # restart). Replay from DB.
    async def _replay() -> AsyncIterator[str]:
        yield _sse("snapshot", {
            "conversation_id": msg.conversation_id,
            "message_id": msg.id,
            "sources": msg.sources or [],
            "text": msg.content,
            "status": msg.status,
        })
        if msg.status == "error":
            yield _sse("error", {"error": msg.error_text or "Generation failed"})
        else:
            yield _sse("done", {})

    return StreamingResponse(
        _replay(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ── Conversation endpoints ─────────────────────────────────────────


@app.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """List the caller's conversations, most recent first. Other users'
    conversations are NEVER returned, regardless of role."""
    from sqlalchemy import func, select as sa_select

    # Subquery: message count per conversation
    msg_count = (
        sa_select(
            Message.conversation_id,
            func.count(Message.id).label("msg_count"),
        )
        .group_by(Message.conversation_id)
        .subquery()
    )

    rows = (
        await session.execute(
            sa_select(Conversation, msg_count.c.msg_count)
            .outerjoin(msg_count, Conversation.id == msg_count.c.conversation_id)
            .where(Conversation.owner_user_id == user.id)
            .order_by(Conversation.updated_at.desc())
        )
    ).all()

    return [
        ConversationSummary(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            message_count=count or 0,
        )
        for conv, count in rows
    ]


@app.post("/conversations")
async def create_conversation(
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Create an empty conversation owned by the caller. Useful when the UI
    needs a conv_id before sending a message (e.g., to attach a file first)."""
    conv = Conversation(title="New Chat", owner_user_id=user.id)
    session.add(conv)
    await session.flush()
    await session.commit()
    return {"id": conv.id, "title": conv.title}


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Load a full conversation with all messages. 404 if not owned."""
    from sqlalchemy import select as sa_select

    conv = await get_user_conversation(conversation_id, user, session)

    rows = (
        await session.execute(
            sa_select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
    ).scalars().all()

    messages = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "status": m.status,
            "error_text": m.error_text,
            "model_used": m.model_used,
            "sources": m.sources,
            "created_at": m.created_at.isoformat(),
        }
        for m in rows
    ]

    in_flight = stream_registry.get_streaming_for_conversation(conv.id)
    streaming_message_id = in_flight.message_id if in_flight is not None else None

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        messages=messages,
        streaming_message_id=streaming_message_id,
    )


@app.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    req: ConversationUpdate,
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Rename a conversation. 404 if not owned."""
    conv = await get_user_conversation(conversation_id, user, session)

    conv.title = req.title
    conv.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return {"id": conv.id, "title": conv.title}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Delete a conversation and all its messages. 404 if not owned."""
    from sqlalchemy import delete as sa_delete

    # Ownership check (raises 404 if not owned).
    await get_user_conversation(conversation_id, user, session)

    await session.execute(
        sa_delete(Message).where(Message.conversation_id == conversation_id)
    )
    await session.execute(
        sa_delete(Conversation).where(Conversation.id == conversation_id)
    )
    await session.commit()
    return {"deleted": conversation_id}


# ── Chat attachments ───────────────────────────────────────────────


@app.post("/conversations/{conversation_id}/attachments")
async def upload_attachment(
    conversation_id: str,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Upload a file scoped to one conversation. 404 if not owned.

    ≤ `inline_attachment_threshold_kb` → stored as inline (full text prepended
    to every turn's context). Otherwise → parsed, chunked, embedded, and
    stored as a conversation-scoped document that participates in retrieval
    only for this conversation.
    """
    conv = await get_user_conversation(conversation_id, user, session)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = _classify_extension(file.filename)
    if not _is_file_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type '{ext or file.filename}' is not supported. "
                f"Supported types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                f" or filenames: {', '.join(sorted(ALLOWED_FILENAMES))}"
            ),
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > settings.max_chat_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. Please upload a file less than "
                f"{settings.max_chat_upload_mb}MB."
            ),
        )

    suffix = ".tf.json" if file.filename.lower().endswith(".tf.json") else ext
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, content)
        os.close(tmp_fd)

        try:
            chunks, doc_meta = await process_file(
                tmp_path,
                {"title": file.filename, "source": "chat_upload"},
                display_name=file.filename,
            )
        except ValueError as exc:
            # parse_file raises ValueError when neither Unstructured nor the
            # pdftotext fallback can extract text (typically image-only PDFs).
            logger.info("chat_upload no-content conv=%s file=%s err=%s",
                        conversation_id, file.filename, exc)
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from this file. If it's an image-only PDF or scan, try converting to text/markdown first.",
            ) from exc
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any content from the file.",
            )

        extracted_text = "\n\n".join(c["text"] for c in chunks)
        text_bytes = len(extracted_text.encode("utf-8"))
        inline_limit = settings.inline_attachment_threshold_kb * 1024

        if text_bytes <= inline_limit:
            attachment = ConversationInlineAttachment(
                conversation_id=conversation_id,
                filename=file.filename,
                mime_type=file.content_type or "application/octet-stream",
                text_content=extracted_text,
                size_bytes=text_bytes,
            )
            session.add(attachment)
            await session.commit()
            logger.info(
                "inline_attachment conv=%s file=%s bytes=%d",
                conversation_id, file.filename, text_bytes,
            )
            return {
                "attachment_id": attachment.id,
                "filename": attachment.filename,
                "mode": "inline",
                "size_bytes": attachment.size_bytes,
                "chunks": None,
            }

        # Chunked path: create conversation-scoped Document + embed Chunks
        doc = Document(
            source="chat_upload",
            title=file.filename,
            source_type="attachment",
            conversation_id=conversation_id,
            metadata_={"title": file.filename, **doc_meta},
        )
        session.add(doc)
        await session.flush()

        chunk_texts = [c["text"] for c in chunks]
        embeddings = await embed_batch(chunk_texts)
        for i, (chunk_data, emb) in enumerate(zip(chunks, embeddings)):
            chunk = Chunk(
                document_id=doc.id,
                content=chunk_data["text"],
                chunk_index=i,
                embedding=emb,
                metadata_={
                    **chunk_data["metadata"],
                    "element_types": chunk_data["element_types"],
                },
            )
            session.add(chunk)

        await session.commit()
        logger.info(
            "chunked_attachment conv=%s file=%s bytes=%d chunks=%d",
            conversation_id, file.filename, text_bytes, len(chunks),
        )
        return {
            "attachment_id": doc.id,
            "filename": file.filename,
            "mode": "chunked",
            "size_bytes": text_bytes,
            "chunks": len(chunks),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/conversations/{conversation_id}/attachments")
async def list_attachments(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """List all attachments (inline + chunked) for a conversation. 404 if not owned."""
    from sqlalchemy import select as sa_select

    await get_user_conversation(conversation_id, user, session)

    inline_rows = (
        await session.execute(
            sa_select(ConversationInlineAttachment)
            .where(ConversationInlineAttachment.conversation_id == conversation_id)
            .order_by(ConversationInlineAttachment.created_at)
        )
    ).scalars().all()

    chunked_rows = (
        await session.execute(
            sa_select(Document)
            .where(Document.source_type == "attachment")
            .where(Document.conversation_id == conversation_id)
            .order_by(Document.created_at)
        )
    ).scalars().all()

    items = []
    for a in inline_rows:
        items.append({
            "attachment_id": a.id,
            "filename": a.filename,
            "mode": "inline",
            "size_bytes": a.size_bytes,
            "created_at": a.created_at.isoformat(),
        })
    for d in chunked_rows:
        items.append({
            "attachment_id": d.id,
            "filename": d.title,
            "mode": "chunked",
            "size_bytes": None,
            "created_at": d.created_at.isoformat(),
        })
    return items


@app.delete("/conversations/{conversation_id}/attachments/{attachment_id}")
async def delete_attachment(
    conversation_id: str,
    attachment_id: str,
    session: AsyncSession = Depends(get_session),
    user: AuthedUser = Depends(require_user),
):
    """Remove a single attachment (inline or chunked) from a conversation. 404 if not owned."""
    from sqlalchemy import delete as sa_delete

    await get_user_conversation(conversation_id, user, session)

    inline = await session.get(ConversationInlineAttachment, attachment_id)
    if inline and inline.conversation_id == conversation_id:
        await session.delete(inline)
        await session.commit()
        return {"deleted": attachment_id, "mode": "inline"}

    doc = await session.get(Document, attachment_id)
    if (
        doc
        and doc.source_type == "attachment"
        and doc.conversation_id == conversation_id
    ):
        await session.execute(
            sa_delete(Chunk).where(Chunk.document_id == attachment_id)
        )
        await session.delete(doc)
        await session.commit()
        return {"deleted": attachment_id, "mode": "chunked"}

    raise HTTPException(status_code=404, detail="Attachment not found")


# ── Common endpoints ───────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def ui():
    """Serve the test UI."""
    ui_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return FileResponse(ui_path, media_type="text/html")
