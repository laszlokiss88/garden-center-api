"""Shared test isolation for process-local application controls."""

import os

import pytest

# Production tracing is configured when the application module is imported. Tests
# install their own in-memory instrumentation explicitly, so local .env settings
# must not initialize global instrumentation during test collection.
os.environ["OTEL_TRACING_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def reset_process_local_rate_limits() -> None:
    from garden_center.api.rate_limiting import limiter

    limiter.reset()
