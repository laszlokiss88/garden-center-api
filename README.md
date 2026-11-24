# Garden Center API

Garden Center API is the backend for a document-based knowledge assistant. It
currently supports PDF upload, text extraction, token-aware chunking, and
document lifecycle management, Ollama embeddings, and pgvector semantic search.

## Technology stack

- Python 3.14 (standard CPython)
- FastAPI and Uvicorn
- PostgreSQL 17 and pgvector
- SQLAlchemy 2 (async), psycopg 3, and Alembic
- pydantic-settings
- PyMuPDF and tiktoken
- Ollama with `nomic-embed-text` embeddings
- uv dependency management
- pytest, HTTPX, and Ruff
- Docker and Docker Compose

## Prerequisites

Install Docker Desktop on Windows 11 and enable its WSL 2 backend. No host
installation of Python, pip, or uv is needed.

## Development

From PowerShell in the project directory, build and start the complete stack:

```powershell
docker compose up --build -d
```

The API waits for PostgreSQL to become healthy and automatically runs
`alembic upgrade head` before Uvicorn starts. The local `src` directory is
mounted with hot reload enabled.

- Health endpoint: <http://localhost:8000/health>
- Readiness endpoint: <http://localhost:8000/ready>
- Swagger UI: <http://localhost:8000/docs>

`/health` is a simple process liveness check; `/ready` verifies database
connectivity.

### Ollama model setup

Ollama starts with the stack, but models are not downloaded automatically.
Pull the configured 768-dimensional embedding model once:

```powershell
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama list
```

The model remains in the `ollama_models` Docker volume across restarts.

## Document API

After pulling the Ollama model, upload and synchronously process a PDF from
PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/documents `
  -F "file=@C:\path\to\document.pdf;type=application/pdf"
```

List documents and retrieve one document with its extracted chunks:

```powershell
curl.exe http://localhost:8000/api/v1/documents
curl.exe http://localhost:8000/api/v1/documents/{document-id}
```

Delete a document, its stored PDF, chunks, and related embeddings:

```powershell
curl.exe -X DELETE http://localhost:8000/api/v1/documents/{document-id}
```

Duplicate PDF content is rejected with HTTP 409 based on its SHA-256 hash.
Uploaded PDFs are stored in the `document_uploads` Docker volume; only metadata
and extracted chunks are stored in PostgreSQL.

A `201 Created` upload response means extraction, chunking, Ollama embedding,
and vector persistence all completed successfully, so the document is already
searchable. Its processing status moves through `processing` to `completed`.
If Ollama is unavailable, missing the model, times out, or returns invalid
vectors, the request returns HTTP 503 and the persisted document is marked
`failed` rather than incorrectly reported as completed. After fixing Ollama,
retry the document through the manual indexing endpoint described below.

### PDF limitations

- Only files named `.pdf` with `Content-Type: application/pdf` are accepted.
- The default maximum upload size is 10 MiB.
- PDFs must contain extractable text. Scanned-image PDFs and OCR are not
  supported.
- Password-protected, malformed, and empty PDFs are rejected.
- Processing is synchronous; background workers are intentionally out of scope.

### Configuration

Copy `.env.example` to `.env` to override the Compose development defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Compose PostgreSQL URL | Async SQLAlchemy connection URL |
| `DOCUMENT_STORAGE_PATH` | `/data/uploads` | PDF storage inside the API container |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | Maximum accepted PDF size |
| `CHUNK_SIZE_TOKENS` | `500` | Maximum tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `75` | Overlap between adjacent page chunks |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama API URL used by the API container |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model identifier |
| `EMBEDDING_DIMENSIONS` | `768` | Required model vector dimensions |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks sent per Ollama request |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Embedding request timeout |
| `SEARCH_TOP_K` | `5` | Default maximum search results |
| `SEARCH_SIMILARITY_THRESHOLD` | `0.5` | Default minimum cosine similarity |

## Embedding indexing and semantic search

Index all chunks that do not yet have an embedding for the configured model:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/embeddings/index
```

Index only one existing document:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/embeddings/index?document_id={document-id}"
```

Repeated indexing is idempotent and reports already indexed chunks as skipped.
It also recovers `failed` documents by embedding only their missing chunks and
marking them `completed` after successful persistence.
Search only considers embeddings whose provider, model, and dimensions match
the configured provider:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/search `
  -H "Content-Type: application/json" `
  -d '{"query":"How often should roses be watered?","top_k":5,"similarity_threshold":0.5}'
```

Each result includes the document ID, filename, source page, chunk index, text,
and cosine similarity score.

### Real Ollama verification

Automated tests inject a deterministic provider and do not require the model.
After pulling it, verify the real endpoint separately from PowerShell:

```powershell
$body = @{ model = "nomic-embed-text"; input = @("healthy tomato plants") } |
  ConvertTo-Json
$result = Invoke-RestMethod -Method Post `
  -Uri http://localhost:11434/api/embed `
  -ContentType "application/json" `
  -Body $body
$result.embeddings[0].Count
```

The final command should output `768`. Then use the indexing and search commands
above for an end-to-end application check.

## Tests and code quality

The full suite includes PostgreSQL integration tests, so keep Compose running:

```powershell
docker compose exec api python -m pytest
```

Run only database-independent unit tests:

```powershell
docker compose exec api python -m pytest -m "not integration"
```

Run lint and formatting checks:

```powershell
docker compose exec api ruff check .
docker compose exec api ruff format --check .
```

## Database migrations

Apply all migrations manually when needed:

```powershell
docker compose exec api alembic upgrade head
```

Create a migration after changing SQLAlchemy metadata:

```powershell
docker compose exec api alembic revision --autogenerate -m "describe change"
```

## Stopping the stack

Stop containers while preserving database data, uploaded PDFs, and Ollama models:

```powershell
docker compose down
```

To intentionally remove all persistent volumes:

```powershell
docker compose down --volumes
```

## Project structure

```text
garden-center-api/
|-- src/garden_center/
|   |-- config.py
|   |-- database.py
|   |-- document_processing.py
|   |-- document_service.py
|   |-- embeddings.py
|   |-- search_service.py
|   |-- models.py
|   |-- schemas.py
|   |-- main.py
|   |-- routes/documents.py
|   `-- routes/search.py
|-- migrations/versions/
|   |-- 20261124_01_enable_pgvector.py
|   `-- 20261124_02_add_documents.py
|-- tests/
|-- alembic.ini
|-- Dockerfile
|-- compose.yaml
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Planned components

Later phases may add RAG orchestration, answer generation, and LangGraph. Those
features are intentionally outside Phase 4.
