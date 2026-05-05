import unittest
from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"


class FrontendSafetyTests(unittest.TestCase):
    def test_model_text_is_not_inserted_with_inner_html(self):
        html = INDEX_HTML.read_text()

        self.assertNotIn("lastUserMsg.innerHTML = `${msg.transcription}", html)
        self.assertIn("document.createTextNode(text)", html)

    def test_tts_playback_waits_for_audio_end_before_listening(self):
        html = INDEX_HTML.read_text()

        self.assertIn("let ttsStreamActive = false;", html)
        self.assertIn("ttsStreamActive = true;", html)
        self.assertIn("ttsStreamActive = false;", html)
        self.assertIn("!ttsStreamActive && streamSources.length === 0", html)

    def test_vad_listening_threshold_is_shared_between_init_and_state_reset(self):
        html = INDEX_HTML.read_text()

        self.assertIn("const LISTENING_VAD_THRESHOLD = 0.35;", html)
        self.assertIn("positiveSpeechThreshold: LISTENING_VAD_THRESHOLD", html)
        self.assertIn("newState === 'speaking' ? SPEAKING_VAD_THRESHOLD : LISTENING_VAD_THRESHOLD", html)


if __name__ == "__main__":
    unittest.main()
