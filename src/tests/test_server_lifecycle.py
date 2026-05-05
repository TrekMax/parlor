import unittest
from pathlib import Path


SERVER_PY = Path(__file__).resolve().parent.parent / "server.py"


class ServerLifecycleTests(unittest.TestCase):
    def test_lifespan_releases_engine_on_shutdown(self):
        source = SERVER_PY.read_text()

        self.assertIn("def unload_models():", source)
        self.assertIn("unload_models()", source)
        self.assertIn("engine.__exit__(None, None, None)", source)


if __name__ == "__main__":
    unittest.main()
