import unittest
from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


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

    def test_kokoro_fallback_dependency_is_available_on_darwin(self):
        pyproject = PYPROJECT.read_text()

        self.assertIn('"kokoro-onnx>=0.5.0"', pyproject)

    def test_tts_playback_has_audio_context_unlock_path(self):
        html = INDEX_HTML.read_text()

        self.assertIn('id="audioToggle"', html)
        self.assertIn("async function resumeAudioContext()", html)
        self.assertIn("resumeAudioContext();", html)
        self.assertIn("document.addEventListener('pointerdown', initAudio", html)
        self.assertIn("document.addEventListener('touchstart', initAudio", html)

    def test_tts_playback_filters_stale_audio_jobs(self):
        html = INDEX_HTML.read_text()

        self.assertIn("let currentAudioJobId = null;", html)
        self.assertIn("currentAudioJobId = msg.job_id || null;", html)
        self.assertIn("isStaleAudioMessage(msg)", html)


if __name__ == "__main__":
    unittest.main()
