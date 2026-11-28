"""RAG-specific attributes for active OpenTelemetry spans."""

import logging
from collections.abc import Mapping

from openinference.instrumentation.langchain import get_current_span
from opentelemetry import trace

logger = logging.getLogger(__name__)


def _set_current_span_attributes(attributes: Mapping[str, object]) -> None:
    """Attach safe metadata to the active span without affecting application work."""
    try:
        span = get_current_span() or trace.get_current_span()
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:
        logger.exception("Could not attach OpenTelemetry span attributes")


def record_retrieval_attributes(retrieved_chunks: int) -> None:
    """Record retrieval attributes on the active LangGraph node span."""
    _set_current_span_attributes({"rag.retrieval.retrieved_chunks": retrieved_chunks})


def record_context_attributes(
    *,
    retrieved_chunks: int,
    included_chunks: int,
    estimated_tokens: int,
    token_budget: int,
    truncated: bool,
) -> None:
    """Record context-budget attributes on the active LangGraph node span."""
    _set_current_span_attributes(
        {
            "rag.context.retrieved_chunks": retrieved_chunks,
            "rag.context.included_chunks": included_chunks,
            "rag.context.estimated_tokens": estimated_tokens,
            "rag.context.token_budget": token_budget,
            "rag.context.budget_utilization_pct": estimated_tokens / token_budget * 100,
            "rag.context.chunks_excluded": retrieved_chunks > included_chunks,
            "rag.context.truncated": truncated,
        }
    )


def record_llm_metadata(provider: str, model: str) -> None:
    """Record provider metadata on the active generation node span."""
    _set_current_span_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
        }
    )


def record_llm_usage(input_tokens: int | None, output_tokens: int | None) -> None:
    """Record provider-reported token usage on the active generation node span."""
    _set_current_span_attributes(
        {
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        }
    )
