"""LangGraph workflow for grounded document question answering."""

from typing import Literal, TypedDict

import tiktoken
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.embeddings import EmbeddingProvider
from garden_center.llm import SYSTEM_PROMPT, LLMProvider
from garden_center.search_service import SearchResult, semantic_search
from garden_center.telemetry import (
    record_context_metrics,
    record_llm_metadata,
    record_llm_usage,
    record_retrieval_metrics,
)

INSUFFICIENT_INFORMATION_ANSWER = (
    "The available documents do not provide enough information to answer this question."
)
_TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
_CHAT_FORMAT_TOKEN_ALLOWANCE = 16
_SOURCE_SEPARATOR = "\n\n"


class QuestionState(TypedDict):
    """State passed between the question-answering graph nodes."""

    question: str
    results: list[SearchResult]
    has_relevant_sources: bool
    answer: str


class ContextSelection(TypedDict):
    """Selected retrieval results and token-budget metadata."""

    results: list[SearchResult]
    estimated_tokens: int
    truncated: bool


def _source_excerpts(results: list[SearchResult]) -> list[str]:
    return [result.text for result in results]


def _select_context_results(
    question: str,
    results: list[SearchResult],
    token_budget: int,
    output_token_allowance: int,
) -> ContextSelection:
    """Select the most relevant excerpts that fit the estimated model budget."""
    prompt_prefix = f"Question:\n{question}\n\nSource excerpts:\n"
    used_tokens = (
        len(_TOKEN_ENCODING.encode(SYSTEM_PROMPT))
        + len(_TOKEN_ENCODING.encode(prompt_prefix))
        + _CHAT_FORMAT_TOKEN_ALLOWANCE
        + output_token_allowance
    )
    selected: list[SearchResult] = []

    for result in sorted(results, key=lambda item: item.similarity, reverse=True):
        separator_tokens = (
            len(_TOKEN_ENCODING.encode(_SOURCE_SEPARATOR)) if selected else 0
        )
        remaining = token_budget - used_tokens - separator_tokens
        if remaining <= 0:
            break
        if result.token_count <= remaining:
            selected.append(result)
            used_tokens += separator_tokens + result.token_count
            continue

        break

    return {
        "results": selected,
        "estimated_tokens": used_tokens,
        "truncated": False,
    }


async def answer_question(
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    question: str,
    top_k: int,
    similarity_threshold: float,
    context_token_budget: int,
    output_token_allowance: int,
) -> QuestionState:
    """Run retrieval, relevance routing, and grounded answer generation."""

    async def retrieve(state: QuestionState) -> dict[str, object]:
        results = await semantic_search(
            session,
            embedding_provider,
            state["question"],
            top_k,
            similarity_threshold,
        )
        record_retrieval_metrics(len(results))
        selection = _select_context_results(
            state["question"],
            results,
            context_token_budget,
            output_token_allowance,
        )
        record_context_metrics(
            retrieved_chunks=len(results),
            included_chunks=len(selection["results"]),
            estimated_tokens=selection["estimated_tokens"],
            token_budget=context_token_budget,
            truncated=selection["truncated"],
        )
        return {"results": selection["results"]}

    def check_relevance(state: QuestionState) -> dict[str, object]:
        return {"has_relevant_sources": bool(state["results"])}

    def relevance_route(state: QuestionState) -> Literal["generate", "insufficient"]:
        return "generate" if state["has_relevant_sources"] else "insufficient"

    async def generate(state: QuestionState) -> dict[str, object]:
        record_llm_metadata(llm_provider.provider_name, llm_provider.model)
        generation = await llm_provider.generate(
            state["question"], _source_excerpts(state["results"])
        )
        record_llm_usage(
            generation.usage.input_tokens,
            generation.usage.output_tokens,
        )
        return {"answer": generation.answer}

    def insufficient(_: QuestionState) -> dict[str, object]:
        return {"answer": INSUFFICIENT_INFORMATION_ANSWER}

    graph = StateGraph(QuestionState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check_relevance", check_relevance)
    graph.add_node("generate", generate)
    graph.add_node("insufficient", insufficient)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "check_relevance")
    graph.add_conditional_edges(
        "check_relevance",
        relevance_route,
        {"generate": "generate", "insufficient": "insufficient"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("insufficient", END)

    workflow = graph.compile(name="garden_center_rag")
    return await workflow.ainvoke(
        {
            "question": question,
            "results": [],
            "has_relevant_sources": False,
            "answer": "",
        }
    )
