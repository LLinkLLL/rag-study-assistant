from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_chat_model, get_min_relevance_score
from src.prompts import QA_PROMPT


INSUFFICIENT_CONTEXT_MESSAGE = (
    "I cannot find enough information in the uploaded materials to answer that."
)


def format_documents_for_prompt(documents: list[Document]) -> str:
    """Format retrieved documents with explicit source labels."""
    formatted_chunks = []
    for doc in documents:
        metadata = doc.metadata or {}
        filename = metadata.get("filename", "Unknown file")
        page_number = metadata.get("page_number", "Unknown page")
        chunk_id = metadata.get("chunk_id", "Unknown chunk")
        relevance_score = metadata.get("relevance_score")
        score_text = f", relevance {relevance_score:.2f}" if isinstance(relevance_score, float) else ""
        source = f"[Source: {filename}, page {page_number}, chunk {chunk_id}{score_text}]"
        formatted_chunks.append(f"{source}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)


def has_enough_context(
    documents: list[Document],
    minimum_characters: int = 20,
    require_relevance_score: bool = False,
    min_relevance_score: float | None = None,
) -> bool:
    """Guardrail before calling the LLM."""
    total_characters = sum(len(doc.page_content.strip()) for doc in documents)
    if total_characters < minimum_characters:
        return False

    if not require_relevance_score:
        return True

    threshold = get_min_relevance_score() if min_relevance_score is None else min_relevance_score
    return any(
        isinstance(doc.metadata.get("relevance_score"), float)
        and doc.metadata["relevance_score"] >= threshold
        for doc in documents
    )


def build_evidence_note(documents: list[Document], min_relevance_score: float | None = None) -> str:
    """Describe evidence strength so the model can answer cautiously."""
    if not documents:
        return "No source excerpts were retrieved."

    threshold = get_min_relevance_score() if min_relevance_score is None else min_relevance_score
    scores = [
        float(doc.metadata["relevance_score"])
        for doc in documents
        if isinstance(doc.metadata.get("relevance_score"), float)
    ]

    if not scores:
        return "Source excerpts were retrieved, but no relevance scores are available."

    best_score = max(scores)
    if best_score < threshold:
        return (
            "Source excerpts were retrieved, but all relevance scores are below the current threshold. "
            "Answer only if the excerpts directly support the answer, and mention that the source evidence is limited."
        )

    if len(documents) == 1:
        return "Only one source excerpt was retrieved. Mention that the answer is based on limited evidence."

    return "Relevant source excerpts were retrieved. Answer using only those excerpts."


def answer_question(
    question: str,
    documents: list[Document],
    model_name: str | None = None,
    min_relevance_score: float | None = None,
) -> str:
    """Generate a source-grounded answer from retrieved chunks."""
    if not has_enough_context(documents):
        return INSUFFICIENT_CONTEXT_MESSAGE

    prompt = ChatPromptTemplate.from_template(QA_PROMPT)
    llm = ChatOpenAI(model=model_name or get_chat_model(), temperature=0)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "question": question,
            "evidence_note": build_evidence_note(documents, min_relevance_score),
            "context": format_documents_for_prompt(documents),
        }
    )
