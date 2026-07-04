from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_chat_model
from src.prompts import SUMMARY_PROMPT
from src.qa_chain import format_documents_for_prompt, has_enough_context


def batch_documents_by_characters(documents: list[Document], max_characters: int = 6000) -> list[list[Document]]:
    """Group documents into prompt-sized batches for map-reduce summarization."""
    batches: list[list[Document]] = []
    current_batch: list[Document] = []
    current_size = 0

    for document in documents:
        document_size = len(document.page_content)
        if current_batch and current_size + document_size > max_characters:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(document)
        current_size += document_size

    if current_batch:
        batches.append(current_batch)

    return batches


def summarize_documents(
    documents: list[Document],
    topic: str = "",
    style: str = "Study notes",
    model_name: str | None = None,
) -> str:
    """Generate a grounded summary from selected source chunks."""
    if not has_enough_context(documents):
        return "There is not enough material to produce a reliable summary."

    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    llm = ChatOpenAI(model=model_name or get_chat_model(), temperature=0.2)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "topic": topic or "General summary of the uploaded materials",
            "style": style,
            "context": format_documents_for_prompt(documents),
        }
    )


def summarize_full_document(
    documents: list[Document],
    style: str = "Study notes",
    model_name: str | None = None,
) -> str:
    """Summarize all indexed chunks with a simple map-reduce workflow."""
    if not has_enough_context(documents):
        return "There is not enough material to produce a reliable summary."

    model = model_name or get_chat_model()
    partial_summaries: list[Document] = []

    for batch_number, batch in enumerate(batch_documents_by_characters(documents), start=1):
        partial_summary = summarize_documents(
            batch,
            topic=f"Full-document summary, batch {batch_number}",
            style=style,
            model_name=model,
        )
        partial_summaries.append(
            Document(
                page_content=partial_summary,
                metadata={"filename": "intermediate summary", "page_number": "multiple", "chunk_id": batch_number},
            )
        )

    return summarize_documents(
        partial_summaries,
        topic="Final full-document summary across all uploaded PDF chunks",
        style=style,
        model_name=model,
    )
