"""HTTP endpoint for grounded document question answering."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.config import Settings, get_settings
from garden_center.database import get_database_session
from garden_center.dependencies import get_embedding_provider, get_llm_provider
from garden_center.embeddings import EmbeddingProvider, EmbeddingProviderError
from garden_center.llm import LLMProvider, LLMProviderError
from garden_center.rag_service import answer_question
from garden_center.schemas import (
    AnswerContextResponse,
    ErrorResponse,
    QuestionRequest,
    QuestionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["question-answering"])
SessionDependency = Annotated[AsyncSession, Depends(get_database_session)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
EmbeddingDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
LLMDependency = Annotated[LLMProvider, Depends(get_llm_provider)]


def _provider_error(exc: LLMProviderError) -> HTTPException:
    status_code = (
        status.HTTP_504_GATEWAY_TIMEOUT
        if exc.code == "generation_timeout"
        else status.HTTP_502_BAD_GATEWAY
        if exc.code == "invalid_generation_response"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    logger.warning("Generation provider error [%s]: %s", exc.code, exc.message)
    return HTTPException(
        status_code=status_code,
        detail={"code": exc.code, "message": exc.message},
    )


@router.post(
    "/questions",
    response_model=QuestionResponse,
    responses={
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def ask_question(
    request: QuestionRequest,
    session: SessionDependency,
    settings: SettingsDependency,
    embedding_provider: EmbeddingDependency,
    llm_provider: LLMDependency,
) -> QuestionResponse:
    """Answer a question using only retrieved document excerpts."""
    try:
        result = await answer_question(
            session,
            embedding_provider,
            llm_provider,
            request.question,
            settings.search_top_k,
            settings.search_similarity_threshold,
            settings.rag_context_token_budget,
            settings.llm_max_tokens,
        )
    except EmbeddingProviderError as exc:
        logger.warning("Question retrieval failed [%s]: %s", exc.code, exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except LLMProviderError as exc:
        raise _provider_error(exc) from exc
    except SQLAlchemyError:
        logger.exception("Question retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "database_unavailable",
                "message": "Question answering is temporarily unavailable.",
            },
        ) from None

    context = [
        AnswerContextResponse(
            document_id=source.document_id,
            filename=source.filename,
            page_number=source.page_number,
            chunk_index=source.chunk_index,
            text=source.text,
            similarity=source.similarity,
        )
        for source in result["results"]
    ]
    return QuestionResponse(
        answer=result["answer"],
        provider=llm_provider.provider_name if context else None,
        model=llm_provider.model if context else None,
        context=context,
    )
