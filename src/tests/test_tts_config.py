import tempfile
import unittest
from pathlib import Path

import numpy as np

import tts


class TTSConfigTests(unittest.TestCase):
    def test_default_qwen_tts_model_is_voice_design(self):
        self.assertEqual(tts.DEFAULT_QWEN_TTS_REPO, "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
        self.assertEqual(tts.DEFAULT_QWEN_TTS_DIRNAME, "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")

    def test_tts_model_path_env_override_is_used_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "custom-tts"
            model_dir.mkdir()

            resolved = tts.resolve_mlx_tts_model_path(
                model_path=str(model_dir),
                models_dir=Path(tmp) / "models",
                downloader=lambda **_: self.fail("downloader should not be called"),
            )

            self.assertEqual(resolved, str(model_dir))

    def test_project_qwen_tts_directory_is_preferred_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            qwen_dir = models_dir / tts.DEFAULT_QWEN_TTS_DIRNAME
            qwen_dir.mkdir(parents=True)

            resolved = tts.resolve_mlx_tts_model_path(
                model_path="",
                models_dir=models_dir,
                downloader=lambda **_: self.fail("downloader should not be called"),
            )

            self.assertEqual(resolved, str(qwen_dir))

    def test_qwen_tts_downloads_into_project_models_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            qwen_dir = models_dir / tts.DEFAULT_QWEN_TTS_DIRNAME
            calls = []

            def downloader(**kwargs):
                calls.append(kwargs)
                return str(qwen_dir)

            resolved = tts.resolve_mlx_tts_model_path(
                model_path="",
                models_dir=models_dir,
                downloader=downloader,
            )

            self.assertEqual(resolved, str(qwen_dir))
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": tts.DEFAULT_QWEN_TTS_REPO,
                        "local_dir": str(qwen_dir),
                    }
                ],
            )


class QwenMLXBackendTests(unittest.TestCase):
    def test_reference_audio_preprocess_trims_and_normalizes_quiet_audio(self):
        speech = np.full(24000, 0.05, dtype=np.float32)
        audio = np.concatenate(
            [
                np.zeros(4800, dtype=np.float32),
                speech,
                np.zeros(4800, dtype=np.float32),
            ]
        )

        processed = tts._preprocess_qwen_reference_audio(
            audio,
            sample_rate=24000,
            trim_threshold=0.01,
            target_peak=0.9,
            min_peak=0.2,
            trim_margin_seconds=0.05,
        )

        self.assertLess(processed.shape[0], audio.shape[0])
        self.assertAlmostEqual(float(np.max(np.abs(processed))), 0.9, places=4)

    def test_reference_audio_requires_reference_text(self):
        class FakeModel:
            sample_rate = 24000

        with self.assertRaises(ValueError):
            tts.QwenMLXBackend(model=FakeModel(), ref_audio="voice.wav", ref_text="")

    def test_invalid_reference_audio_path_raises_clear_error_when_loading_real_model(self):
        missing = Path(tempfile.gettempdir()) / "missing-parlor-reference.wav"
        if missing.exists():
            missing.unlink()

        with self.assertRaises(FileNotFoundError):
            tts.QwenMLXBackend(
                model_path="/tmp/not-loaded-in-this-test",
                ref_audio=str(missing),
                ref_text="你好",
            )

    def test_generate_uses_chinese_language_by_default(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self):
                self.audio = [0.0, 0.1]

        class FakeModel:
            sample_rate = 24000

            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult()

        model = FakeModel()
        backend = tts.QwenMLXBackend(model=model)

        pcm = backend.generate("你好")

        self.assertEqual(model.calls[0]["text"], "你好")
        self.assertEqual(model.calls[0]["lang_code"], "chinese")
        self.assertEqual(model.calls[0]["voice"], None)
        self.assertEqual(len(pcm), 2)

    def test_generate_uses_voice_instruct_path_when_no_reference_audio(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self):
                self.audio = [0.0, 0.1]

        class FakeModel:
            sample_rate = 24000

            def __init__(self):
                self.calls = []

            def _generate_with_instruct(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult()

        model = FakeModel()
        backend = tts.QwenMLXBackend(model=model, voice_instruct="年轻女性", temperature=0.0)

        backend.generate("你好")

        self.assertEqual(model.calls[0]["text"], "你好")
        self.assertEqual(model.calls[0]["language"], "chinese")
        self.assertEqual(model.calls[0]["instruct"], "年轻女性")
        self.assertEqual(model.calls[0]["temperature"], 0.0)

    def test_voice_design_model_uses_public_generate_with_instruct(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self):
                self.audio = [0.0, 0.1]

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()

            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult()

        model = FakeModel()
        backend = tts.QwenMLXBackend(model=model, voice_instruct="年轻女性", temperature=0.0)

        backend.generate("你好")

        self.assertEqual(model.calls[0]["text"], "你好")
        self.assertEqual(model.calls[0]["lang_code"], "chinese")
        self.assertEqual(model.calls[0]["instruct"], "年轻女性")
        self.assertEqual(model.calls[0]["temperature"], 0.0)

    def test_voice_design_model_uses_locked_reference_when_supported(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self, audio):
                self.audio = audio

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

            def __init__(self):
                self.generate_calls = []
                self.icl_calls = []

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                yield FakeResult([0.3, 0.4])

            def _generate_icl(self, **kwargs):
                self.icl_calls.append(kwargs)
                yield FakeResult(np.full(12000, 0.1, dtype=np.float32))

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            voice_instruct="年轻女性",
            voice_lock=True,
            voice_lock_text="你好，我是固定音色。",
            temperature=0.0,
        )

        backend.generate("你好")

        self.assertEqual(model.generate_calls[0]["text"], "你好，我是固定音色。")
        self.assertEqual(model.generate_calls[0]["instruct"], "年轻女性")
        self.assertEqual(model.icl_calls[0]["text"], "你好")
        self.assertEqual(model.icl_calls[0]["ref_text"], "你好，我是固定音色。")
        self.assertEqual(model.icl_calls[0]["temperature"], 0.0)

        backend.generate("第二句")

        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.icl_calls[1]["ref_text"], "你好，我是固定音色。")

    def test_voice_design_model_uses_icl_for_explicit_reference_audio(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self):
                self.audio = np.full(12000, 0.1, dtype=np.float32)

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

            def __init__(self):
                self.generate_calls = []
                self.icl_calls = []

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                yield FakeResult()

            def _generate_icl(self, **kwargs):
                self.icl_calls.append(kwargs)
                yield FakeResult()

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            ref_audio=[0.3, 0.4],
            ref_text="你好，我是固定音色。",
            voice_lock=True,
            temperature=0.0,
        )

        backend.generate("你好")

        self.assertEqual(model.generate_calls, [])
        self.assertEqual(model.icl_calls[0]["text"], "你好")
        self.assertEqual(model.icl_calls[0]["ref_audio"], [0.3, 0.4])
        self.assertEqual(model.icl_calls[0]["ref_text"], "你好，我是固定音色。")

    def test_stream_generate_enables_streaming_for_explicit_reference_audio(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self, audio):
                self.audio = audio

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

            def __init__(self):
                self.icl_calls = []

            def _generate_icl(self, **kwargs):
                self.icl_calls.append(kwargs)
                yield FakeResult(np.full(6000, 0.1, dtype=np.float32))
                yield FakeResult(np.full(6000, 0.1, dtype=np.float32))

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            ref_audio=[0.3, 0.4],
            ref_text="你好，我是固定音色。",
            streaming_interval=0.25,
            temperature=0.0,
        )

        chunks = list(backend.stream_generate("你好"))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(model.icl_calls[0]["stream"], True)
        self.assertEqual(model.icl_calls[0]["streaming_interval"], 0.25)
        self.assertEqual(model.icl_calls[0]["temperature"], 0.0)

    def test_qwen_mode_description_reports_explicit_reference(self):
        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

        backend = tts.QwenMLXBackend(
            model=FakeModel(),
            ref_audio="voice.wav",
            ref_text="你好，我是固定音色。",
        )

        self.assertIn("reference audio", backend.mode_description())

    def test_generate_uses_reference_audio_when_configured(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self):
                self.audio = [0.0, 0.1]

        class FakeModel:
            sample_rate = 24000

            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult()

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            ref_audio="voice.wav",
            ref_text="你好，我是固定音色。",
            voice_instruct="年轻女性",
            temperature=0.0,
        )

        backend.generate("你好")

        self.assertEqual(model.calls[0]["ref_audio"], "voice.wav")
        self.assertEqual(model.calls[0]["ref_text"], "你好，我是固定音色。")
        self.assertEqual(model.calls[0]["temperature"], 0.0)

    def test_reference_audio_empty_generation_falls_back_to_voice_instruct(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self):
                self.audio = [0.2, 0.3]

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

            def __init__(self):
                self.calls = []
                self.icl_calls = []

            def _generate_icl(self, **kwargs):
                self.icl_calls.append(kwargs)
                return
                yield

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult()

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            ref_audio=[0.3, 0.4],
            ref_text="你好，我是固定音色。",
            voice_instruct="年轻女性",
            temperature=0.0,
        )

        pcm = backend.generate("你好")

        self.assertEqual(len(pcm), 2)
        self.assertEqual(model.icl_calls[0]["text"], "你好")
        self.assertEqual(model.calls[0]["instruct"], "年轻女性")

    def test_reference_audio_low_energy_generation_falls_back_to_voice_instruct(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self, audio):
                self.audio = audio

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

            def __init__(self):
                self.calls = []
                self.icl_calls = []

            def _generate_icl(self, **kwargs):
                self.icl_calls.append(kwargs)
                yield FakeResult([0.001, -0.001])

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult([0.2, 0.3])

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            ref_audio=[0.3, 0.4],
            ref_text="你好，我是固定音色。",
            voice_instruct="年轻女性",
            temperature=0.0,
        )

        pcm = backend.generate("你好啊。")

        self.assertEqual(len(pcm), 2)
        self.assertEqual(model.icl_calls[0]["text"], "你好啊。")
        self.assertEqual(model.calls[0]["instruct"], "年轻女性")

    def test_reference_audio_empty_stream_falls_back_to_voice_instruct(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self):
                self.audio = [0.2, 0.3]

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

            def __init__(self):
                self.calls = []
                self.icl_calls = []

            def _generate_icl(self, **kwargs):
                self.icl_calls.append(kwargs)
                return
                yield

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult()

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            ref_audio=[0.3, 0.4],
            ref_text="你好，我是固定音色。",
            voice_instruct="年轻女性",
            temperature=0.0,
        )

        chunks = list(backend.stream_generate("你好"))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(model.icl_calls[0]["stream"], True)
        self.assertEqual(model.calls[0]["instruct"], "年轻女性")

    def test_reference_audio_low_energy_stream_falls_back_to_voice_instruct(self):
        class FakeResult:
            sample_rate = 24000

            def __init__(self, audio):
                self.audio = audio

        class FakeConfig:
            tts_model_type = "voice_design"

        class FakeSpeechTokenizer:
            has_encoder = True

        class FakeModel:
            sample_rate = 24000
            config = FakeConfig()
            speech_tokenizer = FakeSpeechTokenizer()

            def __init__(self):
                self.calls = []
                self.icl_calls = []

            def _generate_icl(self, **kwargs):
                self.icl_calls.append(kwargs)
                yield FakeResult([0.001, -0.001])

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult([0.2, 0.3])

        model = FakeModel()
        backend = tts.QwenMLXBackend(
            model=model,
            ref_audio=[0.3, 0.4],
            ref_text="你好，我是固定音色。",
            voice_instruct="年轻女性",
            temperature=0.0,
        )

        chunks = list(backend.stream_generate("你好啊。"))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(model.icl_calls[0]["stream"], True)
        self.assertEqual(model.calls[0]["instruct"], "年轻女性")


if __name__ == "__main__":
    unittest.main()
