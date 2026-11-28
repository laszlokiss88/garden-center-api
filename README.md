
# Garden Center API

A document-grounded gardening assistant for garden-center staff and home gardeners. It extracts and indexes text-based PDF guides, retrieves relevant passages using PostgreSQL/pgvector, and answers questions with source references through a simple LangGraph workflow.

The project is a Python/FastAPI RAG backend with **fully local Ollama embeddings and answer generation** by default. Anthropic Claude is an optional generation provider; no paid API key is needed for the local demo.

## Quick start

**Prerequisite:** Docker Desktop with Docker Compose. Run from the repository root:

```powershell
docker compose up --build -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen3:1.7b
```

Compose starts PostgreSQL 17/pgvector, Ollama, and the API, runs database migrations, and persists data and models in Docker volumes. Initial downloads require internet access.

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready
```

Swagger UI: <http://localhost:8000/docs>. `/health` checks application liveness; `/ready` checks database connectivity.

### Try the sample data

Upload the included redistributable gardening handbook. Upload and indexing are synchronous; HTTP 201 means indexing completed successfully.

```powershell
curl.exe -X POST http://localhost:8000/api/v1/documents `
  -F "file=@sample_data/flower_gardening_handbook.pdf;type=application/pdf"
```

Ask a question:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/questions `
  -H "Content-Type: application/json" `
  -d '{"question":"Why should bulb leaves remain after flowering?"}'
```

The response contains a grounded `answer`, generation `provider`/`model`, and `sources` with document, page, chunk, excerpt, and similarity. Source passages are provided as context; not every returned passage is necessarily used in the wording of the answer.

To check the no-match fallback:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/questions `
  -H "Content-Type: application/json" `
  -d '{"question":"How do I replace a car transmission?"}'
```

If no chunk meets the relevance threshold, the API returns an insufficient-information answer, empty `sources`, and does not call the generation model.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | Database readiness |
| `POST` | `/api/v1/documents` | Upload and index a PDF |
| `GET` | `/api/v1/documents` | Paginated documents |
| `GET` | `/api/v1/documents/{id}` | Document metadata and chunks |
| `DELETE` | `/api/v1/documents/{id}` | Delete document and index |
| `POST` | `/api/v1/questions` | Document-grounded Q&A |

Duplicate document content is rejected with HTTP 409. Input validation, structured errors, request limits, and model timeout handling are implemented.

The assignment does not require a document-management API, but these endpoints make the prototype easier to operate and verify: documents can be uploaded reproducibly, listed with their processing status, inspected with their extracted chunks, and deleted together with their stored file and index data. Listing is paginated.

## Architecture

**Ingestion:** validate PDF → extract page text (PyMuPDF) → token-aware chunking (`tiktoken`) → store the PDF and document metadata → generate local embeddings (`nomic-embed-text`) → persist the searchable index in PostgreSQL/pgvector.

**Question answering:** embed question → search compatible embeddings by cosine similarity → apply relevance threshold and context budget → LangGraph routes to answer generation or no-match fallback → return answer with sources.

Responsibilities are separated across `api`, `documents`, `rag`, `ai`, `database`, `storage`, `core`, and `observability`. The `database` package owns SQLAlchemy models and session infrastructure, while `storage` provides the file-storage contract and local filesystem implementation. Provider interfaces keep storage-, Ollama-, and Anthropic-specific behavior out of the document and RAG workflows.

See [Architecture and design decisions](docs/architecture.md) for component boundaries, data flow, indexing consistency, concurrency, scaling tradeoffs, and operational limitations.

## Configuration and models

The default local mode works without a `.env` file. To override settings:

```powershell
Copy-Item .env.example .env
```

| Setting | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `OLLAMA_LLM_MODEL` | `qwen3:1.7b` | Local generation model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Local embedding model (768 dimensions) |
| `SEARCH_TOP_K` | `3` | Maximum retrieved passages |
| `SEARCH_SIMILARITY_THRESHOLD` | `0.6` | Minimum passage similarity |
| `LLM_TIMEOUT_SECONDS` | `120` | Generation timeout |
| `RAG_CONTEXT_TOKEN_BUDGET` | `4096` | Estimated input/output context budget |
| `QUESTIONS_RATE_LIMIT_PER_MINUTE` | `15` | Per-IP question limit |
| `OTEL_TRACING_ENABLED` | `false` | Console OpenTelemetry tracing |

See [`.env.example`](.env.example) for all settings, including database credentials, upload limits, concurrency caps, and provider-specific options. Do not commit `.env` or API keys.

### Optional Claude mode

Add to `.env` and recreate the API container:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-sonnet-4-5
```

```powershell
docker compose up -d --force-recreate api
```

Only generation uses Claude; embeddings remain local in Ollama. Claude requires internet access and may incur charges.

### Local model requirements

Ollama must have both models pulled as shown in Quick start. `qwen3:1.7b` is intended for a small local demo, not high-throughput production use. Allow **several GB of available memory** for the Ollama models and runtime; actual requirements vary by host and model loading. CPU-only inference works but can be considerably slower than GPU execution, especially on cold startup or longer contexts. Increase the generation timeout if needed.

## Tests

Tests run inside Docker; core tests use fake providers and do not require paid API access or downloaded Ollama models. Database integration tests use Compose PostgreSQL.

```powershell
docker compose exec api python -m pytest
docker compose exec api python -m pytest -m "not integration"
docker compose exec api ruff check .
docker compose exec api ruff format --check .
```

The live demo above can be used as a manual local integration check; a separate RAG benchmark is not required for this assignment.

## Concurrency, limitations, and next steps

Request-scoped database sessions and graph invocation state prevent cross-request mixing. Async database/model calls, timeouts, per-IP rate limits, and process-local generation limits protect the demo from overload. These limits are **not shared across multiple API workers or instances**.

For larger deployments, the first priorities would be shared rate limiting, a bounded indexing worker/queue, shared persistent storage, and monitoring model latency, failures, and throughput. See [architecture.md](docs/architecture.md) for the rationale and scaling plan; these are **future improvements**, not implemented distributed infrastructure.

Current limitations: text-based PDFs only (no OCR or encrypted PDFs), default 10 MiB upload cap, synchronous indexing, no user accounts or conversation history, heuristic similarity filtering, and potentially slow CPU-based local generation. Interrupted indexing may require manual recovery; incomplete documents are excluded from retrieval. Tracing is optional and disabled by default.

## Operations

```powershell
docker compose exec api alembic upgrade head  # manual migration, if needed
docker compose down                         # stop; keep persistent data
docker compose down --volumes               # delete persistent data too
```
