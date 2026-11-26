"""Central OpenTelemetry tracing setup for the FastAPI application."""

import logging
from collections.abc import Mapping

from fastapi import FastAPI
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)
_TRACER_PROVIDER_STATE_KEY = "tracer_provider"
_LANGCHAIN_INSTRUMENTOR_STATE_KEY = "langchain_instrumentor"


def _set_current_span_attributes(attributes: Mapping[str, object]) -> None:
    """Attach safe metadata to the active span without affecting application work."""
    try:
        span = get_current_span() or trace.get_current_span()
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:
        logger.exception("Could not attach OpenTelemetry span attributes")


def record_retrieval_metrics(retrieved_chunks: int) -> None:
    """Record retrieval metrics on the active LangGraph node span."""
    _set_current_span_attributes({"rag.retrieval.retrieved_chunks": retrieved_chunks})


def record_context_metrics(
    *,
    retrieved_chunks: int,
    included_chunks: int,
    estimated_tokens: int,
    token_budget: int,
    truncated: bool,
) -> None:
    """Record context-budget metrics on the active LangGraph node span."""
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


def configure_tracing(app: FastAPI, enabled: bool) -> None:
    """Configure console tracing when explicitly enabled."""
    if not enabled:
        return

    provider: TracerProvider | None = None
    langchain_instrumentor: LangChainInstrumentor | None = None
    try:
        provider = TracerProvider(
            resource=Resource.create({"service.name": "garden-center-api"})
        )
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            exclude_spans=["receive", "send"],
        )
        langchain_instrumentor = LangChainInstrumentor()
        langchain_instrumentor.instrument(
            tracer_provider=provider,
            config=TraceConfig(
                hide_inputs=True,
                hide_outputs=True,
                hide_llm_invocation_parameters=True,
            ),
        )
        setattr(app.state, _TRACER_PROVIDER_STATE_KEY, provider)
        setattr(
            app.state,
            _LANGCHAIN_INSTRUMENTOR_STATE_KEY,
            langchain_instrumentor,
        )
        logger.info("OpenTelemetry console tracing enabled")
    except Exception:
        logger.exception("Could not enable OpenTelemetry tracing")
        if langchain_instrumentor is not None:
            try:
                langchain_instrumentor.uninstrument()
            except Exception:
                logger.exception("Could not remove failed LangGraph instrumentation")
        if provider is not None:
            try:
                provider.shutdown()
            except Exception:
                logger.exception("Could not shut down failed tracer provider")


def shutdown_tracing(app: FastAPI) -> None:
    """Flush spans and release tracing resources when configured."""
    provider = getattr(app.state, _TRACER_PROVIDER_STATE_KEY, None)
    if provider is None:
        return
    langchain_instrumentor = getattr(app.state, _LANGCHAIN_INSTRUMENTOR_STATE_KEY, None)
    if langchain_instrumentor is not None:
        try:
            langchain_instrumentor.uninstrument()
        except Exception:
            logger.exception("Could not remove LangGraph tracing instrumentation")
    try:
        FastAPIInstrumentor.uninstrument_app(app)
    except Exception:
        logger.exception("Could not remove FastAPI tracing instrumentation")
    try:
        provider.shutdown()
    except Exception:
        logger.exception("Could not shut down OpenTelemetry tracing")
