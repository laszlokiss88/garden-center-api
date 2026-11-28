## Project
Garden Center API is a Python/FastAPI RAG backend developed for a PwC Senior Backend Developer take-home assignment.

Read `docs/Feladatkiiras_Backend_Developer_RAG.pdf` for requirements and follow the existing architecture and conventions.

## Development principles
- Prefer simple, maintainable solutions over unnecessary complexity.
- Follow Python conventions, type hints, and existing project patterns.
- Keep API, application logic, RAG orchestration, and infrastructure integrations separated.
- Use dependency injection and provider abstractions where appropriate.
- Avoid unnecessary database queries, full-table loading, and in-memory processing of large datasets.
- Consider async behavior, concurrency, and resource cleanup.
- Do not introduce new frameworks or infrastructure without a clear need.

## Testing
- Focus on important functionality and meaningful regression risks.
- Do not add tests for every edge case or trivial implementation detail.
- Do not target a specific coverage percentage.
- Run existing tests and Ruff checks after changes.

## Workflow
- Use Docker Compose for development and verification.
- Keep README and architecture documentation consistent with implemented behavior.
- Do not modify files outside this repository.
- Do not commit or push unless explicitly requested.
