"""Process-local request rate limiting for expensive API operations."""

from fastapi import Request, status
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from garden_center.config import get_settings

# get_remote_address uses the ASGI peer address and does not trust forwarded headers.
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")


def questions_rate_limit() -> str:
    return f"{get_settings().questions_rate_limit_per_minute}/minute"


def document_upload_rate_limit() -> str:
    return f"{get_settings().document_upload_rate_limit_per_minute}/minute"


async def rate_limit_exceeded_handler(
    _: Request, __: RateLimitExceeded
) -> JSONResponse:
    """Return the API's standard error shape for rejected requests."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": {
                "code": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
            }
        },
    )
