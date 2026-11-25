"""HTTP routes for embedding indexing and semantic search."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.config import Settings, get_settings
from garden_center.database import get_database_session
from garden_center.dependencies import get_embedding_provider
from garden_center.embeddings import EmbeddingProvider, EmbeddingProviderError
from garden_center.schemas import (
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from garden_center.search_service import semantic_search

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["semantic-search"])
SessionDependency = Annotated[AsyncSession, Depends(get_database_session)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
ProviderDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]


def _provider_error(exc: EmbeddingProviderError) -> HTTPException:
    logger.warning("Embedding provider error [%s]: %s", exc.code, exc.message)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": exc.code, "message": exc.message},
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={503: {"model": ErrorResponse}},
)
async def search_documents(
    request: SearchRequest,
    session: SessionDependency,
    settings: SettingsDependency,
    provider: ProviderDependency,
) -> SearchResponse:
    """Search indexed chunks in the configured embedding space."""
    try:
        results = await semantic_search(
            session,
            provider,
            request.query,
            request.top_k or settings.search_top_k,
            (
                request.similarity_threshold
                if request.similarity_threshold is not None
                else settings.search_similarity_threshold
            ),
        )
    except EmbeddingProviderError as exc:
        raise _provider_error(exc) from exc
    except SQLAlchemyError:
        logger.exception("Semantic search failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "database_unavailable",
                "message": "Semantic search is temporarily unavailable.",
            },
        ) from None
    return SearchResponse(
        provider=provider.provider_name,
        model=provider.model,
        dimensions=provider.dimensions,
        results=[
            SearchResultResponse(
                document_id=result.document_id,
                filename=result.filename,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                text=result.text,
                similarity=result.similarity,
            )
            for result in results
        ],
    )
