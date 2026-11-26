"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import SQLAlchemyError

from garden_center.config import get_settings
from garden_center.database import check_database_connection, dispose_database
from garden_center.rate_limiting import limiter, rate_limit_exceeded_handler
from garden_center.routes.documents import router as documents_router
from garden_center.routes.questions import router as questions_router
from garden_center.telemetry import configure_tracing, shutdown_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release pooled database connections when the application stops."""
    yield
    try:
        await dispose_database()
    finally:
        shutdown_tracing(app)


app = FastAPI(title="Garden Center API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.include_router(documents_router)
app.include_router(questions_router)
configure_tracing(app, get_settings().otel_tracing_enabled)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report whether the API is running."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Report whether the application is ready to serve requests."""
    try:
        await check_database_connection()
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from None
    return {"status": "ready"}
