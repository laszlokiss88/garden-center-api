# Garden Center API

Garden Center API is the backend of a flower-gardening knowledge assistant. It
accepts text-based PDF guides, extracts and chunks their content, creates local
Ollama embeddings, and retrieves relevant passages with PostgreSQL/pgvector.

The current version implements document ingestion and semantic retrieval. It
does **not** yet generate natural-language answers or run a LangGraph workflow;
search results are the retrieved, source-attributed passages. See
[Architecture](docs/architecture.md) for the design, trade-offs, and assignment
gap analysis.

## Quick start

Docker Desktop with Docker Compose is the only host prerequisite. From the
repository root:

```powershell
docker compose up --build -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama list
```

Compose starts PostgreSQL 17 with pgvector, Ollama, and the API. The API waits
for PostgreSQL, runs `alembic upgrade head`, and then starts Uvicorn. PostgreSQL
data, uploaded PDFs, and Ollama models use persistent Docker volumes.

Verify the services:

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready
```

- Swagger UI: <http://localhost:8000/docs>
- `/health` is process liveness and does not access the database.
- `/ready` verifies database connectivity.

### Reproducible demo

Upload the included, redistributable gardening handbook. Upload is synchronous:
a `201 Created` response means extraction, chunking, embedding, and vector
persistence all succeeded.

```powershell
curl.exe -X POST http://localhost:8000/api/v1/documents `
  -F "file=@sample_data/flower_gardening_handbook.pdf;type=application/pdf"
```

Save the returned `id`, then run a semantic search:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/search `
  -H "Content-Type: application/json" `
  -d '{"query":"How much sunlight do roses need?","top_k":5,"similarity_threshold":0.5}'
```

Other useful demo queries:

- `Why should bulb leaves remain after flowering?`
- `Can I plant lilies where cats have access?`
- `How should I water flower pots during summer?`
- `How can I attract pollinators all season?`
- `What should I do about yellow leaves?`

The search response contains the document ID, filename, page number, chunk
index, text, and cosine similarity. It does not synthesize an answer. Queries
outside the handbook, such as current shop prices or tomorrow's weather, should
normally return no result when an appropriate similarity threshold is used.

## Processing and retrieval workflow

1. Validate the upload as a non-empty, text-based PDF within the configured
   size limit.
2. Store the original PDF in the `document_uploads` volume.
3. Extract normalized text and page numbers with PyMuPDF.
4. Split text deterministically with tiktoken, using configurable chunk size
   and overlap.
5. Persist document metadata and chunks in PostgreSQL.
6. Generate `nomic-embed-text` vectors through Ollama in batches and persist
   them with their provider, model, and dimensions.
7. Embed search text in the same embedding space and order compatible chunks
   by pgvector cosine distance.

Embeddings are generated automatically during upload. If extraction, indexing,
or persistence fails, the document, chunks, vectors, and stored PDF are removed
so the same PDF can be retried. Searches cannot mix embeddings created by
different provider/model/dimension combinations.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API liveness |
| `GET` | `/ready` | Database readiness |
| `POST` | `/api/v1/documents` | Upload, process, and index a PDF |
| `GET` | `/api/v1/documents?limit=20&offset=0` | Paginated document metadata |
| `GET` | `/api/v1/documents/{id}` | Metadata and extracted chunks |
| `DELETE` | `/api/v1/documents/{id}` | Delete PDF, chunks, and embeddings |
| `POST` | `/api/v1/search` | Retrieve semantically similar chunks |

The upload response includes metadata, processing status, and chunk count, with
a `Location` header for the detail endpoint. Duplicate PDF content is rejected
with HTTP 409 using its SHA-256 hash. The document list returns `items`, `total`,
`limit`, and `offset`; `limit` defaults to 20 and is capped at 100. PostgreSQL
computes chunk counts without loading chunk text.

### PDF limitations

- Filename extension and content type must identify a PDF.
- Maximum upload size defaults to 10 MiB.
- Scanned PDFs, OCR, encrypted PDFs, and background processing are unsupported.
- Failed uploads are removed and can be submitted again after the dependency
  recovers.

## Configuration

Copy the safe template only when overriding defaults:

```powershell
Copy-Item .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_DB` | `garden_center` | PostgreSQL database |
| `POSTGRES_USER` | `garden_center` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `garden_center` | Local development password |
| `DATABASE_URL` | Compose PostgreSQL URL | SQLAlchemy/psycopg connection URL |
| `DOCUMENT_STORAGE_PATH` | `/data/uploads` | PDF path inside the API container |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | Upload limit |
| `MAX_PDF_PAGES` | `500` | Maximum pages processed per PDF |
| `MAX_EXTRACTED_CHARACTERS` | `2000000` | Extracted-text safety limit |
| `CHUNK_SIZE_TOKENS` | `500` | Maximum tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `75` | Adjacent chunk overlap |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `768` | Expected vector dimensions |
| `EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding request |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Ollama request timeout |
| `SEARCH_TOP_K` | `5` | Default result limit |
| `SEARCH_SIMILARITY_THRESHOLD` | `0.5` | Default minimum similarity |

### Claude and Qwen3

Claude/Qwen3 switching is **not implemented in the current codebase**. There is
no generation-provider abstraction, Anthropic client, Qwen3 model configuration,
or answer endpoint. Setting provider-related environment variables would have no
effect. The only runtime model is the local Ollama `nomic-embed-text` embedding
model. A future assignment phase would need to add a generation interface and
both implementations before a configuration-only switch could be documented.

## Tests and quality checks

All Python commands run inside Docker. Automated tests use a deterministic fake
embedding provider, so they do not require a downloaded Ollama model or paid
service. Integration tests use the real Compose PostgreSQL database.

```powershell
docker compose exec api python -m pytest
docker compose exec api python -m pytest -m "not integration"
docker compose exec api ruff check .
docker compose exec api ruff format --check .
```

To verify real Ollama integration, pull the model, upload the sample PDF, and run
the search request from the quick start.

## Operations

Apply migrations manually if required:

```powershell
docker compose exec api alembic upgrade head
```

Stop services while retaining all data:

```powershell
docker compose down
```

Delete services and persistent data intentionally:

```powershell
docker compose down --volumes
```
