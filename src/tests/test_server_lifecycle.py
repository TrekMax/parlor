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
        self.assertIn("lookup_request_context(transcription)", source)
        self.assertIn("用自然、温暖、口语化的方式回复用户", source)
        self.assertNotIn("tools=[respond_to_user, assistant_tools", source)

    def test_websocket_rejects_second_active_session_before_creating_conversation(self):
        source = SERVER_PY.read_text()

        self.assertIn("active_session_lock = asyncio.Lock()", source)
        self.assertIn("if active_session_lock.locked():", source)
        self.assertIn("await ws.close(code=1013)", source)
        self.assertIn("await active_session_lock.acquire()", source)
        self.assertIn("active_session_lock.release()", source)

    def test_server_uses_tts_worker_pipeline(self):
        source = SERVER_PY.read_text()

        self.assertIn("tts_worker = tts_stream.TTSWorker(tts_backend)", source)
        self.assertIn("await tts_worker.submit(", source)
        self.assertIn("tts_stream.TTSJob(", source)
        self.assertIn("tts_worker.interrupt()", source)
        self.assertIn("await tts_worker.close()", source)


if __name__ == "__main__":
    unittest.main()
