# Parlor

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine.

Parlor uses Gemma 4 via LiteRT-LM for understanding speech and vision, and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech. You talk, show your camera, and it talks back, all locally.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

# Why?

I'm [self-hosting a totally free voice AI](https://www.fikrikarim.com/bule-ai-initial-release/) on my home server to help people learn speaking English. It has hundreds of monthly active users, and I've been thinking about how to keep it free while making it sustainable.

The obvious answer: run everything on-device, eliminating any server cost. Six months ago I needed an RTX 5090 to run just the voice models in real-time.

Google just released a super capable small model that I can run on my M3 Pro in real-time, with vision too! Sure you can't do agentic coding with this, but it is a game-changer for people learning a new language. Imagine a few years from now that people can run this locally on their phones. They can point their camera at objects and talk about them. And this model is multi-lingual, so people can always fallback to their native language if they want. This is essentially what OpenAI demoed a few years ago.

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
- **Sentence-level TTS streaming.** Audio starts playing before the full response is generated.

## Requirements

- Python 3.12+
- macOS with Apple Silicon, or Linux with a supported GPU
- ~3 GB free RAM for the model

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

| Variable     | Default                                                           | Description                                      |
| ------------ | ----------------------------------------------------------------- | ------------------------------------------------ |
| `MODEL_PATH`     | project `models/`, then auto-download E2B into `models/`           | Path to a `.litertlm` file or directory          |
| `TTS_MODEL_PATH` | project `models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` on Apple Silicon | Path to a local mlx-audio TTS model directory |
| `TTS_VOICE_INSTRUCT` | `年轻女性，普通话标准，声音清晰自然，音色稳定，语速适中。` | VoiceDesign voice description |
| `TTS_VOICE_LOCK` | `1` | Generate one VoiceDesign reference clip and reuse it through ICL |
| `TTS_VOICE_LOCK_TEXT` | `你好，我是你的语音助手，声音清晰自然，音色稳定。` | Text used for the generated voice-lock reference |
| `TTS_TEMPERATURE` | `0` | Greedy decoding for more stable TTS |
| `TTS_STREAMING_INTERVAL` | `0.5` | Qwen streaming chunk interval in seconds |
| `TTS_REF_AUDIO` / `TTS_REF_TEXT` | unset | Optional real reference clip and exact transcript |
| `PORT`           | `8000`                                                            | Server port                                      |

## Performance (Apple M3 Pro)

| Stage                            | Time          |
| -------------------------------- | ------------- |
| Speech + vision understanding    | ~1.8-2.2s     |
| Response generation (~25 tokens) | ~0.3s         |
| Text-to-speech (1-3 sentences)   | ~0.3-0.7s     |
| **Total end-to-end**             | **~2.5-3.0s** |

Decode speed: ~83 tokens/sec on GPU (Apple M3 Pro).

## Project structure

```
src/
├── server.py              # FastAPI WebSocket server + Gemma 4 inference
├── tts.py                 # Platform-aware TTS (MLX on Mac, ONNX on Linux)
├── index.html             # Frontend UI (VAD, camera, audio playback)
├── pyproject.toml         # Dependencies
└── benchmarks/
    ├── bench.py           # End-to-end WebSocket benchmark
    └── benchmark_tts.py   # TTS backend comparison
```

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection

## License

[Apache 2.0](LICENSE)
