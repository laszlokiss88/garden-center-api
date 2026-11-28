"""Helpers and handlers for consistent API error responses."""

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """Build an HTTP exception using the API's standard error shape."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


async def request_validation_error_handler(
    _: Request, __: RequestValidationError
) -> JSONResponse:
    """Return the standard error shape for invalid request data."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": {
                "code": "invalid_request",
                "message": "The request contains invalid or missing data.",
            }
        },
    )
