"""Pydantic response schemas for the document API."""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class DocumentChunkResponse(BaseModel):
    """Persisted document chunk returned by the detail endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    text_content: str
    source_page: int
    token_count: int


class DocumentResponse(BaseModel):
    """Document metadata returned by create and list operations."""

    id: uuid.UUID
    filename: str
    content_type: str
    file_size: int
    content_hash: str
    processing_status: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int


class DocumentDetailResponse(DocumentResponse):
    """Document metadata together with extracted chunks."""

    chunks: list[DocumentChunkResponse]


class DocumentListResponse(BaseModel):
    """A page of document metadata."""

    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class ErrorDetail(BaseModel):
    """Machine-readable API error information."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Structured error response."""

    detail: ErrorDetail


class QuestionRequest(BaseModel):
    """A grounded question for the indexed document collection."""

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    ]


class AnswerSourceResponse(BaseModel):
    """A retrieved excerpt provided to the LLM as grounding context."""

    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    text: str
    similarity: float


class QuestionResponse(BaseModel):
    """A grounded answer and the excerpts provided to the LLM."""

    answer: str
    provider: str | None
    model: str | None
    sources: list[AnswerSourceResponse]
