import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.chunking.chunker import chunk_parsed_document_async
from backend.config import settings
from backend.db.connection import engine, get_session
from backend.db.models import Base, Chunk, Document
from backend.embedding.embedder import embed_batch
from backend.generation.llm import generate_answer
from backend.parsing.parser import parse_file, parse_text
from backend.retrieval.vector_search import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables + pgvector extension on startup
    async with engine.begin() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.run_sync(Base.metadata.create_all)
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
    parsed_doc = parse_text(req.content, extra_metadata=req.metadata)

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
        parsed_doc = await asyncio.to_thread(parse_file, tmp_path, extra_metadata)

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
    # 1. Retrieve relevant chunks
    results = await search(req.question, session, top_k=req.top_k)

    if not results:
        return QueryResponse(answer="No relevant documents found.", sources=[])

    # 2. Generate answer from context
    context_texts = [r.content for r in results]
    answer = await generate_answer(req.question, context_texts)

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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def ui():
    """Serve the test UI."""
    ui_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return FileResponse(ui_path, media_type="text/html")
