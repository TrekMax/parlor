import unittest
from pathlib import Path


SERVER_PY = Path(__file__).resolve().parent.parent / "server.py"


class ServerLifecycleTests(unittest.TestCase):
    def test_lifespan_releases_engine_on_shutdown(self):
        source = SERVER_PY.read_text()

        self.assertIn("def unload_models():", source)
        self.assertIn("unload_models()", source)
        self.assertIn("engine.__exit__(None, None, None)", source)

    def test_runtime_messages_do_not_append_instruction_prompts(self):
        source = SERVER_PY.read_text()

        self.assertIn("用户正在通过麦克风与你对话", source)
        self.assertNotIn("TASK_PROMPT_", source)
        self.assertNotIn("content.append({\"type\": \"text\", \"text\": TASK_PROMPT_", source)
        self.assertNotIn("用户刚刚通过语音与你说话", source)
        self.assertNotIn("The user just spoke", source)
        self.assertNotIn("Respond to what they said", source)

    def test_conversation_keeps_single_final_response_tool(self):
        source = SERVER_PY.read_text()

        self.assertIn("tools=[respond_to_user]", source)
        self.assertIn("answer_lookup_request(transcription)", source)
        self.assertNotIn("tools=[respond_to_user, assistant_tools", source)


if __name__ == "__main__":
    unittest.main()
