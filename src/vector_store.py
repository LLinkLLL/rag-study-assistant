from __future__ import annotations

import shutil
from hashlib import sha256
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import get_embedding_model


COLLECTION_NAME = "course_materials"


def get_embeddings() -> OpenAIEmbeddings:
    """Create the OpenAI embedding model used by ChromaDB."""
    return OpenAIEmbeddings(model=get_embedding_model())


def get_vector_store(persist_directory: str = "chroma_db") -> Chroma:
    """Create or connect to a persistent ChromaDB vector store."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def build_stable_chunk_id(document: Document) -> str:
    """Build a repeatable Chroma ID from source metadata and chunk content."""
    metadata = document.metadata or {}
    filename = str(metadata.get("filename", "unknown"))
    page_number = str(metadata.get("page_number", "unknown"))
    chunk_index = str(metadata.get("chunk_index", metadata.get("chunk_id", "unknown")))
    chunk_hash = str(metadata.get("chunk_hash") or sha256(document.page_content.encode("utf-8")).hexdigest()[:16])
    raw_id = f"{filename}|page:{page_number}|chunk:{chunk_index}|hash:{chunk_hash}"

    return sha256(raw_id.encode("utf-8")).hexdigest()


def add_documents_to_store(vector_store: Chroma, documents: list[Document]) -> list[str]:
    """Add new chunk documents to ChromaDB while skipping duplicates."""
    if not documents:
        return []

    ids = [build_stable_chunk_id(doc) for doc in documents]
    existing_ids = set(vector_store.get(ids=ids).get("ids", []))
    seen_ids: set[str] = set()
    new_pairs = []

    for doc, doc_id in zip(documents, ids):
        if doc_id in existing_ids or doc_id in seen_ids:
            continue

        seen_ids.add(doc_id)
        new_pairs.append((doc, doc_id))

    if not new_pairs:
        return []

    new_documents, new_ids = zip(*new_pairs)
    vector_store.add_documents(list(new_documents), ids=list(new_ids))
    return list(new_ids)


def count_documents(vector_store: Chroma) -> int:
    """Return the number of chunks currently stored in ChromaDB."""
    return vector_store._collection.count()


def get_all_documents(vector_store: Chroma, limit: int | None = 20) -> list[Document]:
    """Fetch stored documents for broad summaries when no topic is supplied."""
    get_kwargs = {"include": ["documents", "metadatas"]}
    if limit is not None:
        get_kwargs["limit"] = limit

    result = vector_store.get(**get_kwargs)
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    return [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(documents, metadatas)
        if text
    ]


def clear_vector_store(persist_directory: str = "chroma_db", vector_store: Chroma | None = None) -> None:
    """Clear the ChromaDB collection and remove local persistence files if possible."""
    if vector_store is not None:
        try:
            vector_store.delete_collection()
        except Exception:
            # The collection may not exist yet, which is fine for a reset action.
            pass

    path = Path(persist_directory)
    if path.exists():
        try:
            shutil.rmtree(path)
        except PermissionError:
            # On Windows, Chroma may briefly hold SQLite files open. The deleted
            # collection is enough to reset the app; the old files can be removed
            # after the process exits.
            pass
