import unittest
from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"


class FrontendSafetyTests(unittest.TestCase):
    def test_model_text_is_not_inserted_with_inner_html(self):
        html = INDEX_HTML.read_text()

        self.assertNotIn("lastUserMsg.innerHTML = `${msg.transcription}", html)
        self.assertIn("document.createTextNode(text)", html)


if __name__ == "__main__":
    unittest.main()
