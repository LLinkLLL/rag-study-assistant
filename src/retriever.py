from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_min_relevance_score


def retrieve_documents_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = 4,
) -> list[Document]:
    """Retrieve top-k chunks and attach Chroma relevance scores to metadata."""
    if not query.strip():
        return []

    scored_results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    documents: list[Document] = []

    for document, score in scored_results:
        document.metadata = {
            **(document.metadata or {}),
            "relevance_score": round(float(score), 4),
        }
        documents.append(document)

    return documents


def filter_documents_by_relevance(
    documents: list[Document],
    min_relevance_score: float | None = None,
) -> list[Document]:
    """Keep only chunks above the relevance threshold."""
    threshold = get_min_relevance_score() if min_relevance_score is None else min_relevance_score
    return [
        document
        for document in documents
        if float(document.metadata.get("relevance_score", 0.0)) >= threshold
    ]


def retrieve_documents(
    vector_store: Chroma,
    query: str,
    k: int = 4,
    min_relevance_score: float | None = None,
) -> list[Document]:
    """Retrieve top-k chunks and keep only chunks above the relevance threshold."""
    return filter_documents_by_relevance(
        retrieve_documents_with_scores(vector_store, query, k=k),
        min_relevance_score=min_relevance_score,
    )
