# AppLeap RAG

An Enterprise AI platform for IT Operations, built for SMBs and small enterprises. The system ingests data from customer tools, stores it in a per-customer vector database, and provides AI-powered answers grounded in the customer's own data.

**Hard constraint: 100% on-premise.** Zero customer data leaves the customer's infrastructure. Every component — LLM, embeddings, parsing, database, connectors — runs locally. No cloud APIs, no telemetry, no external calls at runtime.

---

## End-to-End Pipeline

### 1. Data Ingestion

Customers connect their tools (Google Drive, GitHub, Jira). Lightweight Python connectors authenticate via OAuth or API keys and pull raw data — files from Drive, issues from GitHub, tickets from Jira.

### 2. Parsing

Raw files (PDF, DOCX, PPTX, HTML, Markdown, etc.) are fed into **Unstructured.io**, which converts them into a uniform list of typed elements — titles, narrative text, tables, list items, code snippets. Each element carries metadata: page number, section heading, filename, source. This solves the problem of losing structure and metadata during ingestion.

### 3. Chunking

A content-type-aware router selects the right chunking strategy for each document:

| Strategy | When Used | How It Works |
|----------|-----------|--------------|
| **by_title** (default) | Documents with headings — runbooks, docs, markdown, Jira | Groups elements under their nearest heading. Keeps sections together. |
| **by_similarity** (fallback) | Unstructured prose with no headings | Embeds elements via Nomic, cuts where cosine similarity between consecutive elements drops below threshold. |
| **naive** | Single-element documents | Simple character-based splitting. |

Auto-detection: titles present → `by_title`. No titles → `by_similarity`. Single element → `naive`.

Chunk defaults: max 1500 chars, soft target 1200 chars, 200 char overlap, combine small chunks under 500 chars.

### 4. Embedding

Each chunk is embedded using **Nomic-embed-text** (768-dimension vectors) running locally via **Ollama**. Batch embedding is used for efficiency.

### 5. Storage

Embeddings and metadata are stored in **PostgreSQL + pgvector**. Two tables:
- `documents` — id, source, title, metadata, created_at
- `chunks` — id, document_id, content, chunk_index, embedding (768-dim vector), metadata, created_at

### 6. Retrieval

When a user asks a question, the query is embedded with Nomic and a cosine similarity search is run against pgvector to find the top-K most relevant chunks (default K=5).

### 7. Answer Generation

Retrieved chunks are assembled into a context prompt and passed to **Phi-4** (14.7B parameters, Q4_K_M quantization) running locally via Ollama. The model generates a grounded response — answering only from the provided context.

---

## Architecture Diagram

```
Customer connects tools (Google Drive, GitHub, Jira, ...)
        |
        v
+-----------------------------------------------+
|           CONNECTOR LAYER                      |
|                                                |
|  Google Drive ---> |                           |
|  GitHub ---------> | Lightweight Python        |
|  Jira -----------> | connectors (custom)       |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|           PARSING & CHUNKING                   |
|                                                |
|  Unstructured.io (all file types)              |
|  Content-type-aware chunking router            |
|  + connector metadata (ACLs, author, project)  |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|           EMBEDDING & STORAGE                  |
|                                                |
|  Nomic-embed-text (768-dim) via Ollama         |
|  PostgreSQL + pgvector                         |
|  Per-customer isolation                        |
+-----------------------------------------------+
        |
        v
   User asks question
        |
        v
   Retrieval (cosine similarity over pgvector)
        |
        v
   Phi-4 (14.7B) via Ollama generates grounded response
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Phi-4 (14.7B, Q4_K_M) via Ollama | Strong reasoning at manageable size. Runs on a single L4 GPU. |
| Embeddings | Nomic-embed-text (768-dim) via Ollama | Local, no API dependency. Runs alongside Phi-4 in the same Ollama instance. |
| Database | PostgreSQL 18 + pgvector 0.8.2 | Battle-tested. Vectors, metadata filtering, and real SQL in one database. |
| Parsing | Unstructured.io [all-docs] (local, open-source) | Handles PDF, DOCX, PPTX, HTML, Markdown. Metadata is first-class output. |
| Backend | FastAPI (Python 3.13) | Async-native, auto-generated API docs, lightweight. |
| Deployment | docker-compose, air-gapped capable | Single `docker-compose up` on customer hardware. |

---

## Infrastructure

### Primary: L4 GPU VM (Google Cloud)

| Spec | Value |
|------|-------|
| Instance | `appleap-dev` (us-central1-a) |
| Machine type | g2-standard-8 |
| CPU / RAM | 8 vCPU / 32 GB |
| GPU | 1x NVIDIA L4 (24 GB VRAM) |
| Storage | 100 GB SSD |
| OS | Ubuntu 22.04 LTS |
| NVIDIA Driver | 570.211.01, CUDA 12.8 |

**Performance on L4:**
- Phi-4 generation: ~27 tokens/sec
- Phi-4 prompt processing: ~300 tokens/sec
- Nomic embedding: ~300 tokens/sec
- Model cold load: ~4s, warm: 119ms

### Secondary: Local Windows (development only)

| Spec | Value |
|------|-------|
| OS | Windows 11 Home |
| Python | 3.13 |
| PostgreSQL | 18 + pgvector 0.8.2 |
| GPU | None — LLM inference is too slow for practical use |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Ingest raw text — parse, chunk, embed, store |
| `POST` | `/ingest/file` | Upload a file — Unstructured parses it, then chunk, embed, store |
| `POST` | `/query` | Ask a question — retrieves relevant chunks, generates answer via Phi-4 |
| `GET` | `/health` | Health check |

---

## Project Structure

```
appleap-rag/
├── backend/
│   ├── main.py              # FastAPI app — all endpoints
│   ├── config.py             # Settings via env vars (APPLEAP_ prefix)
│   ├── chunking/
│   │   └── chunker.py        # Content-type-aware router (by_title, by_similarity, naive)
│   ├── connectors/           # Google Drive, GitHub, Jira (stubs, in progress)
│   │   ├── base.py
│   │   ├── google_drive.py
│   │   ├── github.py
│   │   └── jira.py
│   ├── db/
│   │   ├── connection.py      # Async SQLAlchemy engine + session
│   │   └── models.py          # Document + Chunk ORM models
│   ├── embedding/
│   │   └── embedder.py        # Nomic via Ollama (single + batch)
│   ├── generation/
│   │   └── llm.py             # Phi-4 via Ollama
│   ├── parsing/
│   │   ├── base.py            # ElementType enum, ParsedElement, ParsedDocument
│   │   └── parser.py          # Unstructured.io partition()
│   └── retrieval/
│       └── vector_search.py   # Cosine distance search over pgvector
├── eval/                      # Evaluation framework (planned)
├── frontend/                  # UI (planned)
├── tests/
├── pyproject.toml
├── requirements.txt
└── CLAUDE.md
```

---

## Running the Server

On the L4 VM:

```bash
# Install as editable package (first time only)
cd ~/appleap-rag
pip install -e ".[parsing]"

# Start Ollama + models
ollama serve &
ollama pull phi4
ollama pull nomic-embed-text

# Start the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://<vm-ip>:8000/docs`.

---

## Roadmap

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Local Brain — end-to-end RAG pipeline | Done |
| 2 | Live Connector + Unstructured.io integration | In Progress |
| 3 | Security — ACLs, user login, retrieval filtering | Planned |
| 4 | Hybrid Search — BM25 + vector, re-ranking | Planned |
| 5 | Hardening — evaluation framework, metrics, load testing | Planned |
| 6 | Ship It — docker-compose, single deployable unit | Planned |
