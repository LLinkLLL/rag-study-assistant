import unittest

from src.chunker import chunk_pages


class ChunkerTests(unittest.TestCase):
    def test_chunk_pages_preserves_source_metadata(self):
        pages = [
            {
                "filename": "lecture.pdf",
                "page_number": 2,
                "text": "Retrieval augmented generation uses source documents. " * 20,
            }
        ]

        chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        first_metadata = chunks[0].metadata
        self.assertEqual(first_metadata["filename"], "lecture.pdf")
        self.assertEqual(first_metadata["page_number"], 2)
        self.assertEqual(first_metadata["chunk_index"], 1)
        self.assertIn("chunk_hash", first_metadata)
        self.assertIn("chunk_id", first_metadata)


if __name__ == "__main__":
    unittest.main()
