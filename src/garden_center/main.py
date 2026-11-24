"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from garden_center.database import check_database_connection, dispose_database
from garden_center.routes.documents import router as documents_router
from garden_center.routes.search import router as search_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Release pooled database connections when the application stops."""
    yield
    await dispose_database()


app = FastAPI(title="Garden Center API", version="0.1.0", lifespan=lifespan)
app.include_router(documents_router)
app.include_router(search_router)


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
