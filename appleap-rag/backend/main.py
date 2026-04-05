import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.chunking.chunker import chunk_parsed_document_async
from backend.config import settings
from backend.connectors.base import SyncResult, SyncStatus

try:
    from backend.connectors.google_drive import GoogleDriveConnector

    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False
from backend.db.connection import async_session, engine, get_session
from backend.db.models import Base, Chunk, Document
from backend.embedding.embedder import embed_batch
from backend.generation.llm import generate_answer
from backend.parsing.parser import parse_file, parse_text
from backend.retrieval.vector_search import search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables + pgvector extension on startup
    async with engine.begin() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.run_sync(Base.metadata.create_all)

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
    yield
    await engine.dispose()


app = FastAPI(title="AppLeap RAG", version="0.2.0", lifespan=lifespan)

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


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".md", ".txt", ".html", ".htm", ".rst", ".csv", ".json", ".xml",
}


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


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source: str = Form(default="upload"),
    metadata_json: str = Form(default="{}"),
    session: AsyncSession = Depends(get_session),
):
    """Ingest a file: parse via Unstructured → chunk → embed → store."""
    # 1. Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
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
            status_code=400,
            detail=f"File too large. Max: {settings.max_upload_size_mb}MB",
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        os.write(tmp_fd, content)
        os.close(tmp_fd)

        # 3. Parse with Unstructured (sync, run in thread)
        parsed_doc = await asyncio.to_thread(
            parse_file, tmp_path, {**extra_metadata, "title": file.filename}
        )

        # 4. Chunk (async — by_similarity needs embeddings)
        chunks = await chunk_parsed_document_async(parsed_doc)
        if not chunks:
            raise HTTPException(
                status_code=400, detail="No content extracted from file"
            )

        # 5. Create document record
        doc = Document(
            source=source,
            title=file.filename,
            metadata_={
                **extra_metadata,
                "filetype": parsed_doc.filetype,
                "original_filename": parsed_doc.filename,
                "num_elements": len(parsed_doc.elements),
            },
        )
        session.add(doc)
        await session.flush()

        # 6. Embed all chunk texts in one batch
        chunk_texts = [c["text"] for c in chunks]
        embeddings = await embed_batch(chunk_texts)

        # 7. Store chunks with embeddings
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


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, session: AsyncSession = Depends(get_session)):
    """Answer a question using retrieved context from pgvector + Phi-4."""
    # 1. Retrieve relevant chunks (with neighbor expansion)
    results = await search(req.question, session, top_k=req.top_k)

    # 2. Generate answer from context (or general knowledge if no results)
    answer = await generate_answer(req.question, results)

    # 3. Return answer + source metadata
    sources = [
        {
            "chunk_id": r.id,
            "document_id": r.document_id,
            "content_preview": r.content[:200],
            "chunk_index": r.chunk_index,
        }
        for r in results
    ]
    return QueryResponse(answer=answer, sources=sources)


# ── Google Drive connector ─────────────────────────────────────────

# In-memory sync state (single-worker; for multi-worker use DB/Redis)
_gdrive_sync: SyncResult | None = None
_gdrive_sync_lock = asyncio.Lock()


async def _ingest_connector_file(
    connector_file,
    session: AsyncSession,
) -> int:
    """Parse, chunk, embed, and store a single connector file.

    Returns the number of chunks stored.
    """
    parsed_doc = await asyncio.to_thread(
        parse_file,
        connector_file.file_path,
        {
            **connector_file.metadata,
            "title": connector_file.filename,
            "source": connector_file.source,
            "permissions": connector_file.permissions,
        },
    )

    chunks = await chunk_parsed_document_async(parsed_doc)
    if not chunks:
        return 0

    doc = Document(
        source=connector_file.source,
        title=connector_file.filename,
        metadata_={
            **connector_file.metadata,
            "permissions": connector_file.permissions,
            "source_id": connector_file.source_id,
            "filetype": parsed_doc.filetype,
            "original_filename": parsed_doc.filename,
            "num_elements": len(parsed_doc.elements),
        },
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
                "permissions": connector_file.permissions,
            },
        )
        session.add(chunk)

    await session.commit()
    return len(chunks)


async def _run_google_drive_sync() -> None:
    """Background task: authenticate, list, download, ingest all files."""
    global _gdrive_sync

    connector = GoogleDriveConnector()

    try:
        await connector.authenticate()
    except Exception as e:
        _gdrive_sync.status = SyncStatus.FAILED
        _gdrive_sync.errors.append(f"Authentication failed: {e}")
        _gdrive_sync.finished_at = datetime.now(timezone.utc)
        return

    try:
        files = await connector.list_files()
    except Exception as e:
        _gdrive_sync.status = SyncStatus.FAILED
        _gdrive_sync.errors.append(f"Listing files failed: {e}")
        _gdrive_sync.finished_at = datetime.now(timezone.utc)
        return

    _gdrive_sync.files_found = len(files)

    # Check which files are already ingested (by drive_file_id) to support
    # incremental sync — skip files whose last_modified hasn't changed.
    already_ingested: dict[str, str] = {}  # drive_file_id -> last_modified
    async with async_session() as session:
        from sqlalchemy import select as sa_select

        rows = (
            await session.execute(
                sa_select(Document.metadata_)
                .where(Document.source == "google_drive")
            )
        ).scalars().all()
        for meta in rows:
            if meta and meta.get("drive_file_id"):
                already_ingested[meta["drive_file_id"]] = meta.get("last_modified", "")

    for cf in files:
        file_id = cf.source_id
        new_modified = cf.metadata.get("last_modified", "")

        # Skip if already ingested and not modified since
        if file_id in already_ingested and already_ingested[file_id] == new_modified:
            _gdrive_sync.files_skipped += 1
            logger.info("Skipping unchanged file: %s", cf.filename)
            continue

        try:
            cf = await connector.download_file(cf)

            async with async_session() as session:
                # If file was previously ingested but now modified, remove old version
                if file_id in already_ingested:
                    await _delete_document_by_source_id(session, file_id)

                chunks_stored = await _ingest_connector_file(cf, session)

            _gdrive_sync.files_ingested += 1
            _gdrive_sync.chunks_stored += chunks_stored
            logger.info(
                "Ingested: %s (%d chunks)", cf.filename, chunks_stored
            )

        except Exception as e:
            _gdrive_sync.files_failed += 1
            _gdrive_sync.errors.append(f"{cf.filename}: {e}")
            logger.error("Failed to ingest %s: %s", cf.filename, e)

        finally:
            # Clean up temp file
            if cf.file_path:
                try:
                    os.unlink(cf.file_path)
                except OSError:
                    pass

    _gdrive_sync.status = SyncStatus.COMPLETED
    _gdrive_sync.finished_at = datetime.now(timezone.utc)
    logger.info(
        "Google Drive sync complete: %d ingested, %d skipped, %d failed",
        _gdrive_sync.files_ingested,
        _gdrive_sync.files_skipped,
        _gdrive_sync.files_failed,
    )


async def _delete_document_by_source_id(session: AsyncSession, drive_file_id: str) -> None:
    """Remove a previously ingested Drive file (document + chunks) for re-sync."""
    from sqlalchemy import delete as sa_delete, select as sa_select

    # Find the document
    result = await session.execute(
        sa_select(Document.id).where(
            Document.source == "google_drive",
            Document.metadata_["drive_file_id"].astext == drive_file_id,
        )
    )
    doc_id = result.scalar_one_or_none()
    if not doc_id:
        return

    await session.execute(sa_delete(Chunk).where(Chunk.document_id == doc_id))
    await session.execute(sa_delete(Document).where(Document.id == doc_id))
    await session.commit()


@app.post("/connectors/google-drive/sync")
async def google_drive_sync():
    """Trigger a Google Drive sync. Runs in the background.

    Pulls all accessible files, parses, chunks, embeds, and stores them.
    Skips files that haven't changed since last sync (incremental).
    """
    global _gdrive_sync

    if not _GDRIVE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Google Drive dependencies not installed. "
            "Run: pip install 'appleap-rag[google-drive]'",
        )

    async with _gdrive_sync_lock:
        if _gdrive_sync and _gdrive_sync.status == SyncStatus.RUNNING:
            raise HTTPException(
                status_code=409,
                detail="A sync is already running. Check GET /connectors/google-drive/status",
            )

        _gdrive_sync = SyncResult(status=SyncStatus.RUNNING)

    # Fire and forget — sync runs in background
    asyncio.create_task(_run_google_drive_sync())

    return {"message": "Google Drive sync started", "status": "running"}


@app.get("/connectors/google-drive/status")
async def google_drive_status():
    """Check the current/last Google Drive sync status."""
    if _gdrive_sync is None:
        return {"status": "idle", "message": "No sync has been run yet"}
    return _gdrive_sync.to_dict()


# ── Common endpoints ───────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def ui():
    """Serve the test UI."""
    ui_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return FileResponse(ui_path, media_type="text/html")
