"""Platform-aware TTS: Qwen3 via mlx-audio on Apple Silicon, kokoro-onnx elsewhere."""

import os
import platform
import sys
from pathlib import Path

import numpy as np

import model_config

DEFAULT_QWEN_TTS_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
DEFAULT_QWEN_TTS_DIRNAME = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
DEFAULT_QWEN_VOICE_INSTRUCT = "年轻女性，普通话标准，声音清晰自然，音色稳定，语速适中。"
DEFAULT_QWEN_VOICE_LOCK_TEXT = "你好，我是你的语音助手，声音清晰自然，音色稳定。"
DEFAULT_QWEN_VOICE_LOCK_MAX_TOKENS = 768
DEFAULT_QWEN_TEMPERATURE = 0.0
DEFAULT_QWEN_STREAMING_INTERVAL = 0.5


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str | None = None, speed: float = 1.0) -> np.ndarray:
        raise NotImplementedError

    def stream_generate(self, text: str, voice: str | None = None, speed: float = 1.0):
        yield self.generate(text, voice=voice, speed=speed)


def resolve_mlx_tts_model_path(
    model_path: str | None = None,
    models_dir: Path = model_config.DEFAULT_MODELS_DIR,
    downloader=None,
) -> str:
    """Resolve the local Qwen3 TTS model directory, downloading into models/ if needed."""
    configured = model_path if model_path is not None else os.environ.get("TTS_MODEL_PATH", "")
    if configured:
        return str(Path(configured).expanduser())

    local_dir = models_dir.expanduser() / DEFAULT_QWEN_TTS_DIRNAME
    if local_dir.is_dir():
        return str(local_dir)

    should_log_download = downloader is None
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    if should_log_download:
        print(f"Downloading {DEFAULT_QWEN_TTS_REPO} into {local_dir} (first run only)...")
    return downloader(repo_id=DEFAULT_QWEN_TTS_REPO, local_dir=str(local_dir))


class QwenMLXBackend(TTSBackend):
    """Qwen3 TTS backend through mlx-audio."""

    def __init__(
        self,
        model=None,
        model_path: str | None = None,
        lang_code: str = "chinese",
        voice_instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        temperature: float | None = None,
        voice_lock: bool | None = None,
        voice_lock_text: str | None = None,
        voice_lock_max_tokens: int | None = None,
        streaming_interval: float | None = None,
    ):
        self.lang_code = lang_code
        self.voice_instruct = (
            voice_instruct
            if voice_instruct is not None
            else os.environ.get("TTS_VOICE_INSTRUCT", DEFAULT_QWEN_VOICE_INSTRUCT)
        )
        self.ref_audio = ref_audio if ref_audio is not None else os.environ.get("TTS_REF_AUDIO")
        self.ref_text = ref_text if ref_text is not None else os.environ.get("TTS_REF_TEXT")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.environ.get("TTS_TEMPERATURE", DEFAULT_QWEN_TEMPERATURE))
        )
        self.voice_lock = (
            voice_lock
            if voice_lock is not None
            else os.environ.get("TTS_VOICE_LOCK", "1").lower() not in {"0", "false", "no"}
        )
        self.voice_lock_text = (
            voice_lock_text
            if voice_lock_text is not None
            else os.environ.get("TTS_VOICE_LOCK_TEXT", DEFAULT_QWEN_VOICE_LOCK_TEXT)
        )
        self.voice_lock_max_tokens = (
            voice_lock_max_tokens
            if voice_lock_max_tokens is not None
            else int(os.environ.get("TTS_VOICE_LOCK_MAX_TOKENS", DEFAULT_QWEN_VOICE_LOCK_MAX_TOKENS))
        )
        self.streaming_interval = (
            streaming_interval
            if streaming_interval is not None
            else float(os.environ.get("TTS_STREAMING_INTERVAL", DEFAULT_QWEN_STREAMING_INTERVAL))
        )
        self._locked_ref_audio = None
        self._validate_reference_config(validate_paths=model is None)
        if model is not None:
            self._model = model
            self.sample_rate = self._model.sample_rate
            return
        from mlx_audio.tts.generate import load_model

        resolved_model_path = resolve_mlx_tts_model_path(model_path)
        self._model = load_model(Path(resolved_model_path))
        self.sample_rate = self._model.sample_rate
        if self._can_use_voice_lock():
            self._ensure_voice_lock()
        else:
            list(self._generate_results("你好", max_tokens=256))

    def generate(self, text: str, voice: str | None = None, speed: float = 1.0) -> np.ndarray:
        results = list(self._generate_results(text=text, voice=voice, speed=speed))
        if not results and self.ref_audio and self.ref_text:
            print("TTS: reference audio generated no audio, falling back to voice instruction")
            results = list(self._generate_without_reference(text=text, voice=voice, speed=speed, stream=False))
        if not results:
            raise RuntimeError("Qwen3 TTS generated no audio")
        return np.concatenate([np.array(r.audio) for r in results])

    def stream_generate(self, text: str, voice: str | None = None, speed: float = 1.0):
        yielded = False
        for result in self._generate_results(text=text, voice=voice, speed=speed, stream=True):
            yielded = True
            yield np.array(result.audio)
        if not yielded and self.ref_audio and self.ref_text:
            print("TTS: reference audio stream generated no audio, falling back to voice instruction")
            for result in self._generate_without_reference(text=text, voice=voice, speed=speed, stream=False):
                yield np.array(result.audio)

    def _generate_results(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        max_tokens: int = 4096,
        stream: bool = False,
    ):
        if self.ref_audio and self.ref_text:
            if self._can_use_voice_design_icl():
                return self._model._generate_icl(
                    text=text,
                    ref_audio=self._load_reference_audio(self.ref_audio),
                    ref_text=self.ref_text,
                    language=self.lang_code,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.5,
                    verbose=False,
                    stream=stream,
                    streaming_interval=self.streaming_interval,
                )

            return self._model.generate(
                text=text,
                voice=voice,
                speed=speed,
                lang_code=self.lang_code,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                temperature=self.temperature,
                verbose=False,
                max_tokens=max_tokens,
                stream=stream,
                streaming_interval=self.streaming_interval,
            )

        if self._can_use_voice_lock():
            self._ensure_voice_lock()
            return self._model._generate_icl(
                text=text,
                ref_audio=self._locked_ref_audio,
                ref_text=self.voice_lock_text,
                language=self.lang_code,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.5,
                verbose=False,
                stream=stream,
                streaming_interval=self.streaming_interval,
            )

        if self.voice_instruct and self._model_tts_type() == "voice_design":
            return self._generate_voice_design_instruct(text, voice, speed, max_tokens, stream)

        if self.voice_instruct and hasattr(self._model, "_generate_with_instruct"):
            return self._generate_with_instruct(text, voice, max_tokens, stream)

        return self._model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=self.lang_code,
            temperature=self.temperature,
            verbose=False,
            max_tokens=max_tokens,
            stream=stream,
            streaming_interval=self.streaming_interval,
        )

    def _generate_without_reference(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        max_tokens: int = 4096,
        stream: bool = False,
    ):
        if self.voice_instruct and self._model_tts_type() == "voice_design":
            return self._generate_voice_design_instruct(text, voice, speed, max_tokens, stream)

        if self.voice_instruct and hasattr(self._model, "_generate_with_instruct"):
            return self._generate_with_instruct(text, voice, max_tokens, stream)

        return self._model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=self.lang_code,
            temperature=self.temperature,
            verbose=False,
            max_tokens=max_tokens,
            stream=stream,
            streaming_interval=self.streaming_interval,
        )

    def _generate_voice_design_instruct(
        self,
        text: str,
        voice: str | None,
        speed: float,
        max_tokens: int,
        stream: bool,
    ):
        return self._model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=self.lang_code,
            instruct=self.voice_instruct,
            temperature=self.temperature,
            verbose=False,
            max_tokens=max_tokens,
            stream=stream,
            streaming_interval=self.streaming_interval,
        )

    def _generate_with_instruct(self, text: str, voice: str | None, max_tokens: int, stream: bool):
        return self._model._generate_with_instruct(
            text=text,
            speaker=voice,
            language=self.lang_code,
            instruct=self.voice_instruct,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.05,
            verbose=False,
            stream=stream,
            streaming_interval=self.streaming_interval,
        )

    def _model_tts_type(self) -> str | None:
        config = getattr(self._model, "config", None)
        if isinstance(config, dict):
            return config.get("tts_model_type")
        return getattr(config, "tts_model_type", None)

    def mode_description(self) -> str:
        if self.ref_audio and self.ref_text:
            return f"reference audio ({self.ref_audio})"
        if self.ref_audio and not self.ref_text:
            return "reference audio ignored: TTS_REF_TEXT is missing"
        if self._can_use_voice_lock():
            return f"voice lock ({self.voice_lock_text})"
        if self.voice_instruct and self._model_tts_type() == "voice_design":
            return f"voice design ({self.voice_instruct})"
        return "default generation"

    def _can_use_voice_lock(self) -> bool:
        speech_tokenizer = getattr(self._model, "speech_tokenizer", None)
        return (
            self.voice_lock
            and not (self.ref_audio and self.ref_text)
            and bool(self.voice_instruct)
            and self._model_tts_type() == "voice_design"
            and self._can_use_voice_design_icl()
        )

    def _can_use_voice_design_icl(self) -> bool:
        speech_tokenizer = getattr(self._model, "speech_tokenizer", None)
        return (
            self._model_tts_type() == "voice_design"
            and hasattr(self._model, "_generate_icl")
            and bool(getattr(speech_tokenizer, "has_encoder", False))
        )

    def _load_reference_audio(self, ref_audio):
        if isinstance(ref_audio, (str, Path)):
            from mlx_audio.utils import load_audio

            return load_audio(ref_audio, sample_rate=self.sample_rate)
        return ref_audio

    def _validate_reference_config(self, validate_paths: bool):
        if bool(self.ref_audio) != bool(self.ref_text):
            raise ValueError("TTS_REF_AUDIO and TTS_REF_TEXT must be configured together")

        if validate_paths and isinstance(self.ref_audio, (str, Path)):
            ref_path = Path(self.ref_audio).expanduser()
            if not ref_path.is_file():
                raise FileNotFoundError(f"TTS_REF_AUDIO does not exist: {self.ref_audio}")
            self.ref_audio = str(ref_path)

    def _ensure_voice_lock(self):
        if self._locked_ref_audio is not None:
            return

        results = list(
            self._model.generate(
                text=self.voice_lock_text,
                lang_code=self.lang_code,
                instruct=self.voice_instruct,
                temperature=self.temperature,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.05,
                verbose=False,
                max_tokens=self.voice_lock_max_tokens,
            )
        )
        if not results:
            raise RuntimeError("Qwen3 VoiceDesign voice lock generated no reference audio")

        if len(results) == 1:
            self._locked_ref_audio = results[0].audio
            return

        import mlx.core as mx

        self._locked_ref_audio = mx.concatenate([r.audio for r in results])


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend (ONNX Runtime, CPU)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str | None = "af_heart", speed: float = 1.1) -> np.ndarray:
        pcm, _sr = self._model.create(text, voice=voice, speed=speed)
        return pcm


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform."""
    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = QwenMLXBackend()
            print(
                "TTS: Qwen3 mlx-audio "
                f"(Apple GPU, sample_rate={backend.sample_rate}, mode={backend.mode_description()})"
            )
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to kokoro-onnx")

    backend = ONNXBackend()
    print(f"TTS: kokoro-onnx (CPU, sample_rate={backend.sample_rate})")
    return backend
