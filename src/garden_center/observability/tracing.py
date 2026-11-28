"""OpenTelemetry tracing lifecycle for the FastAPI application."""

import logging

from fastapi import FastAPI
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)
_TRACER_PROVIDER_STATE_KEY = "tracer_provider"
_LANGCHAIN_INSTRUMENTOR_STATE_KEY = "langchain_instrumentor"


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
