"""LangGraph workflow for grounded document question answering."""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.embeddings import EmbeddingProvider
from garden_center.llm import LLMProvider
from garden_center.search_service import SearchResult, semantic_search

INSUFFICIENT_INFORMATION_ANSWER = (
    "The available documents do not provide enough information to answer "
    "this question."
)


class QuestionState(TypedDict):
    """State passed between the question-answering graph nodes."""

    question: str
    results: list[SearchResult]
    has_relevant_sources: bool
    answer: str


def _source_excerpts(results: list[SearchResult]) -> list[str]:
    return [result.text for result in results]


async def answer_question(
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    question: str,
    top_k: int,
    similarity_threshold: float,
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
        return {"results": results}

    def check_relevance(state: QuestionState) -> dict[str, object]:
        return {"has_relevant_sources": bool(state["results"])}

    def relevance_route(state: QuestionState) -> Literal["generate", "insufficient"]:
        return "generate" if state["has_relevant_sources"] else "insufficient"

    async def generate(state: QuestionState) -> dict[str, object]:
        answer = await llm_provider.generate(
            state["question"], _source_excerpts(state["results"])
        )
        return {"answer": answer}

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

    workflow = graph.compile()
    return await workflow.ainvoke(
        {
            "question": question,
            "results": [],
            "has_relevant_sources": False,
            "answer": "",
        }
    )
