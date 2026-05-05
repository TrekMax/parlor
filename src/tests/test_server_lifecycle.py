import unittest
from pathlib import Path


SERVER_PY = Path(__file__).resolve().parent.parent / "server.py"


class ServerLifecycleTests(unittest.TestCase):
    def test_lifespan_releases_engine_on_shutdown(self):
        source = SERVER_PY.read_text()

        self.assertIn("def unload_models():", source)
        self.assertIn("unload_models()", source)
        self.assertIn("engine.__exit__(None, None, None)", source)

    def test_runtime_task_prompts_are_chinese(self):
        source = SERVER_PY.read_text()

        self.assertIn("用户刚刚通过语音与你说话", source)
        self.assertNotIn("The user just spoke", source)
        self.assertNotIn("Respond to what they said", source)

    def test_conversation_registers_time_and_weather_tools(self):
        source = SERVER_PY.read_text()

        self.assertIn("assistant_tools.get_current_time", source)
        self.assertIn("assistant_tools.get_weather", source)
        self.assertIn("仍然必须用 respond_to_user 回复用户", source)


if __name__ == "__main__":
    unittest.main()
