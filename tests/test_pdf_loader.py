import unittest

import fitz

from src.pdf_loader import extract_pdf_pages


class PdfLoaderTests(unittest.TestCase):
    def test_extract_pdf_pages_returns_page_text_and_metadata(self):
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Lecture note text for retrieval testing.")
        pdf_bytes = pdf.tobytes()
        pdf.close()

        pages = extract_pdf_pages(pdf_bytes, "lecture.pdf")

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["filename"], "lecture.pdf")
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Lecture note text", pages[0]["text"])


if __name__ == "__main__":
    unittest.main()
