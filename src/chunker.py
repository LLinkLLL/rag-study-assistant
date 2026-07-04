from __future__ import annotations

from hashlib import sha256
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split page text into overlapping LangChain Document chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for page in pages:
        chunks = splitter.split_text(page["text"])

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_hash = sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
            stable_chunk_id = sha256(
                f"{page['filename']}|{page['page_number']}|{chunk_index}|{chunk_hash}".encode("utf-8")
            ).hexdigest()[:16]

            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "filename": page["filename"],
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "chunk_hash": chunk_hash,
                        "chunk_id": stable_chunk_id,
                    },
                )
            )

    return documents
