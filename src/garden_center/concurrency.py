"""Process-local concurrency controls for expensive provider operations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from anyio import CapacityLimiter, WouldBlock

from garden_center.llm import LLMProvider, LLMProviderError

_LIMITERS: dict[tuple[str, str, int], CapacityLimiter] = {}


def _limiter_for(provider: LLMProvider) -> CapacityLimiter | None:
    limit = provider.max_concurrent_generations
    if limit is None:
        return None
    key = (provider.provider_name, provider.model, limit)
    limiter = _LIMITERS.get(key)
    if limiter is None:
        limiter = CapacityLimiter(limit)
        _LIMITERS[key] = limiter
    return limiter


@asynccontextmanager
async def llm_generation_slot(provider: LLMProvider) -> AsyncIterator[None]:
    """Acquire generation capacity immediately or reject without queueing."""
    limiter = _limiter_for(provider)
    if limiter is None:
        yield
        return
    try:
        limiter.acquire_nowait()
    except WouldBlock:
        raise LLMProviderError(
            "generation_capacity_unavailable",
            "Answer generation is temporarily at capacity.",
        ) from None
    try:
        yield
    finally:
        limiter.release()
