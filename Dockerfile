FROM ghcr.io/astral-sh/uv:0.9.18-python3.14-bookworm-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY README.md ./
COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
COPY tests ./tests

RUN uv sync --locked
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

EXPOSE 8000

CMD ["uvicorn", "garden_center.main:app", "--host", "0.0.0.0", "--port", "8000"]
