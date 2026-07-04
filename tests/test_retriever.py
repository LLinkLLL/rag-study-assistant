import unittest

from langchain_core.documents import Document

from src.retriever import retrieve_documents


class FakeVectorStore:
    def similarity_search_with_relevance_scores(self, query, k=4):
        return [
            (Document(page_content="high relevance", metadata={"filename": "a.pdf"}), 0.91),
            (Document(page_content="low relevance", metadata={"filename": "b.pdf"}), 0.12),
        ][:k]


class RetrieverTests(unittest.TestCase):
    def test_retrieve_documents_filters_by_relevance_threshold(self):
        documents = retrieve_documents(
            FakeVectorStore(),
            "What is RAG?",
            k=2,
            min_relevance_score=0.5,
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "high relevance")
        self.assertEqual(documents[0].metadata["relevance_score"], 0.91)

    def test_retrieve_documents_returns_empty_for_blank_query(self):
        documents = retrieve_documents(FakeVectorStore(), "   ", k=2)

        self.assertEqual(documents, [])


if __name__ == "__main__":
    unittest.main()
