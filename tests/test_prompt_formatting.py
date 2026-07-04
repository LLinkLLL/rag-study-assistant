import unittest

from langchain_core.documents import Document

from src.qa_chain import format_documents_for_prompt


class PromptFormattingTests(unittest.TestCase):
    def test_format_documents_for_prompt_includes_source_and_score(self):
        documents = [
            Document(
                page_content="A source-grounded answer should cite retrieved chunks.",
                metadata={
                    "filename": "week1.pdf",
                    "page_number": 3,
                    "chunk_id": "abc123",
                    "relevance_score": 0.876,
                },
            )
        ]

        formatted = format_documents_for_prompt(documents)

        self.assertIn("week1.pdf", formatted)
        self.assertIn("page 3", formatted)
        self.assertIn("chunk abc123", formatted)
        self.assertIn("relevance 0.88", formatted)
        self.assertIn("A source-grounded answer", formatted)


if __name__ == "__main__":
    unittest.main()
