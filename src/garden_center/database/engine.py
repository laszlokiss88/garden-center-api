"""Reusable asynchronous SQLAlchemy infrastructure."""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from garden_center.core.config import get_settings
from garden_center.database.models import Base

__all__ = ["Base", "get_database_session"]

engine: AsyncEngine = create_async_engine(
    get_settings().database_url,
    pool_pre_ping=True,
)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_database_session() -> AsyncGenerator[AsyncSession]:
    """Yield a database session and always close it afterward."""
    async with async_session_factory() as session:
        yield session


async def check_database_connection() -> None:
    """Execute a minimal query to verify database connectivity."""
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def dispose_database() -> None:
    """Close all connections held by the engine pool."""
    await engine.dispose()
