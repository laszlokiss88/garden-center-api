"""Focused tests for liveness and database readiness endpoints."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.exc import SQLAlchemyError

import garden_center.main as main_module


def _request(path: str) -> Response:
    async def send() -> Response:
        transport = ASGITransport(app=main_module.app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    return asyncio.run(send())


def test_health_is_independent_of_database() -> None:
    response = _request("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.integration
def test_ready_with_database() -> None:
    response = _request("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_when_database_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable() -> None:
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(main_module, "check_database_connection", unavailable)

    response = _request("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "database_unavailable",
            "message": "Database unavailable.",
        }
    }
