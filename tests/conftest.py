"""Shared test isolation for process-local application controls."""

import pytest

from garden_center.rate_limiting import limiter


@pytest.fixture(autouse=True)
def reset_process_local_rate_limits() -> None:
    limiter.reset()
