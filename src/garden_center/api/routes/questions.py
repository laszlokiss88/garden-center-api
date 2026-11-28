"""HTTP endpoint for grounded document question answering."""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError

from garden_center.ai.embeddings import EmbeddingProviderError
from garden_center.ai.llm import LLMProviderError
from garden_center.api.dependencies import (
    EmbeddingDependency,
    LLMDependency,
    SessionDependency,
    SettingsDependency,
)
from garden_center.api.errors import api_error
from garden_center.api.rate_limiting import limiter, questions_rate_limit
from garden_center.api.schemas import (
    AnswerSourceResponse,
    ErrorResponse,
    QuestionRequest,
    QuestionResponse,
)
from garden_center.rag.service import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["question-answering"])


def _provider_error(exc: LLMProviderError) -> HTTPException:
    status_code = (
        status.HTTP_504_GATEWAY_TIMEOUT
        if exc.code == "generation_timeout"
        else status.HTTP_502_BAD_GATEWAY
        if exc.code == "invalid_generation_response"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    logger.warning("Generation provider error [%s]: %s", exc.code, exc.message)
    return api_error(status_code, exc.code, exc.message)


@router.post(
    "/questions",
    response_model=QuestionResponse,
    responses={
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
@limiter.limit(questions_rate_limit)
async def ask_question(
    request: Request,
    question_request: QuestionRequest,
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
            question_request.question,
            settings.search_top_k,
            settings.search_similarity_threshold,
            settings.rag_context_token_budget,
            settings.llm_max_tokens,
        )
    except EmbeddingProviderError as exc:
        logger.warning("Question retrieval failed [%s]: %s", exc.code, exc.message)
        raise api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, exc.code, exc.message
        ) from exc
    except LLMProviderError as exc:
        raise _provider_error(exc) from exc
    except SQLAlchemyError:
        logger.exception("Question retrieval failed")
        raise api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "Question answering is temporarily unavailable.",
        ) from None

    sources = [
        AnswerSourceResponse(
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
        provider=llm_provider.provider_name if sources else None,
        model=llm_provider.model if sources else None,
        sources=sources,
    )
