# Parlor

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine.

Parlor uses Gemma 4 via LiteRT-LM for understanding speech and vision, and Qwen3 TTS through MLX on Apple Silicon for text-to-speech. You talk, show your camera, and it talks back, all locally.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

## Acknowledgment

This project is based on and modified from [fikrikarim/parlor](https://github.com/fikrikarim/parlor). Thanks to the original author for the clear and useful local multimodal voice AI prototype.

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
FastAPI server
    ├── Gemma 4 via LiteRT-LM (GPU)  →  understands speech + vision
    └── Qwen3 TTS (MLX on Mac, ONNX on Linux fallback)  →  speaks back
    │
    │  WebSocket (streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Voice Activity Detection** in the browser ([Silero VAD](https://github.com/ricky0123/vad)). Hands-free, no push-to-talk.
- **Echo suppression.** Ignores assistant playback during and shortly after TTS to avoid repeated self-triggering.
- **Streaming Qwen TTS.** Qwen audio chunks are sent as soon as they are generated.
- **Project-local models.** Model files are resolved from `models/` by default and are ignored by git.

## Requirements

- Python 3.12+
- macOS with Apple Silicon, or Linux with a supported GPU
- Enough local storage for Gemma LiteRT-LM and Qwen3 TTS model files

## Quick start

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

cd src
uv sync
uv run server.py
```

Open [http://localhost:8000](http://localhost:8000), grant camera and microphone access, and start talking.

If present, the app uses a `.litertlm` file from the project `models/` directory. On Apple Silicon, TTS defaults to `models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`; otherwise that model is downloaded automatically into `models/` on first run.

To use an existing local Qwen3 TTS checkout without another download, place or symlink it at:

```bash
models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
```

Or point to it explicitly:

```bash
TTS_MODEL_PATH=/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 uv run server.py
```

For a fixed young female voice, use the VoiceDesign instruction. Voice lock is enabled by default: on startup the app generates one short reference clip from `TTS_VOICE_LOCK_TEXT`, then clones that reference for later TTS calls.

```bash
TTS_VOICE_INSTRUCT=年轻女性，普通话标准，声音清晰自然，音色稳定，语速适中。 \
TTS_VOICE_LOCK=1 \
TTS_TEMPERATURE=0 \
uv run server.py
```

For the most stable voice, provide a short real reference clip and its exact transcript:

```bash
TTS_REF_AUDIO=../models/voices/young-female.wav \
TTS_REF_TEXT=你好，我是一个声音清晰自然的年轻女性。 \
uv run server.py
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | project `models/`, then auto-download E2B into `models/` | Path to a `.litertlm` file or directory |
| `TTS_MODEL_PATH` | project `models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` on Apple Silicon | Path to a local mlx-audio TTS model directory |
| `TTS_VOICE_INSTRUCT` | `年轻女性，普通话标准，声音清晰自然，音色稳定，语速适中。` | VoiceDesign voice description |
| `TTS_VOICE_LOCK` | `1` | Generate one VoiceDesign reference clip and reuse it through ICL |
| `TTS_VOICE_LOCK_TEXT` | `你好，我是你的语音助手，声音清晰自然，音色稳定。` | Text used for the generated voice-lock reference |
| `TTS_TEMPERATURE` | `0` | Greedy decoding for more stable TTS |
| `TTS_STREAMING_INTERVAL` | `0.5` | Qwen streaming chunk interval in seconds |
| `TTS_REF_AUDIO` / `TTS_REF_TEXT` | unset | Optional real reference clip and exact transcript |
| `PORT` | `8000` | Server port |

## Project structure

```
src/
├── server.py              # FastAPI WebSocket server + Gemma 4 inference
├── tts.py                 # Platform-aware TTS (MLX on Mac, ONNX on Linux)
├── tts_stream.py          # WebSocket TTS streaming helpers
├── response_utils.py      # Response normalization helpers
├── index.html             # Frontend UI (VAD, camera, audio playback)
├── pyproject.toml         # Dependencies
└── benchmarks/
    ├── bench.py           # End-to-end WebSocket benchmark
    └── benchmark_tts.py   # TTS backend comparison
```

## Acknowledgments

- [fikrikarim/parlor](https://github.com/fikrikarim/parlor), the original project this work is based on
- [Gemma](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Qwen3 TTS](https://huggingface.co/Qwen) and mlx-audio for local TTS
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection

## License

[Apache 2.0](LICENSE)
