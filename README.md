# Parlor

本项目是一个本地运行的实时多模态语音助手：浏览器采集麦克风和摄像头，后端使用 Gemma LiteRT-LM 理解语音和画面，再通过本地 TTS 合成语音返回。整个对话链路可以在本机运行。

> 致谢：本项目基于 [fikrikarim/parlor](https://github.com/fikrikarim/parlor) 修改而来。感谢原作者提供了清晰、实用的本地多模态语音 AI 原型。

[English README](README.en.md)

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **实验项目。** 当前仍是研究和本地实验性质，可能存在边界问题、性能波动和兼容性问题。

## 工作方式

```
浏览器（麦克风 + 摄像头）
    │
    │  WebSocket（WAV 音频 + JPEG 画面）
    ▼
FastAPI 后端
    ├── Gemma via LiteRT-LM（GPU） → 理解语音和画面
    └── Qwen3 TTS（Mac 上使用 MLX） → 合成语音
    │
    │  WebSocket（流式音频块）
    ▼
浏览器（播放语音 + 显示转写/回复）
```

主要能力：

- 浏览器端使用 [Silero VAD](https://github.com/ricky0123/vad) 做语音活动检测，不需要按键说话。
- 支持摄像头画面输入，模型可以结合语音和画面回答。
- 支持回声抑制，AI 播放语音时会抑制 VAD 误触发。
- 支持 Qwen3 TTS 流式输出，第一块音频生成后立即推送到浏览器播放。
- 模型默认放在项目 `models/` 目录下，模型文件和参考音频不会提交到 git。

## 环境要求

- Python 3.12+
- macOS Apple Silicon，或 Linux + 可用 GPU
- 本地有足够磁盘空间存放 Gemma LiteRT-LM 和 Qwen3 TTS 模型
- 推荐使用 `uv` 管理依赖

## 快速开始

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

# 如果还没有安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd src
uv sync
uv run server.py
```

打开 [http://localhost:8000](http://localhost:8000)，允许浏览器访问摄像头和麦克风后即可开始对话。

## 模型目录

默认情况下，后端会优先从项目根目录的 `models/` 中查找 `.litertlm` 模型文件。Apple Silicon 上的 TTS 默认使用：

```bash
models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
```

如果这个目录不存在，首次运行时会尝试下载到 `models/`。

已有本地 Qwen3 TTS 模型时，可以把模型放到默认目录，或创建软链接：

```bash
mkdir -p models
ln -s /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
```

也可以启动时显式指定：

```bash
cd src
TTS_MODEL_PATH=/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 uv run server.py
```

## 固定音色

VoiceDesign 模型默认启用 `TTS_VOICE_LOCK=1`。启动时会先根据 `TTS_VOICE_INSTRUCT` 和 `TTS_VOICE_LOCK_TEXT` 生成一段参考音频，后续语音会复用这段参考音频，减少音色漂移。

```bash
cd src
TTS_VOICE_INSTRUCT=年轻女性，普通话标准，声音清晰自然，音色稳定，语速适中。 \
TTS_VOICE_LOCK=1 \
TTS_TEMPERATURE=0 \
uv run server.py
```

如果你有真实的参考音频，稳定性通常更好。参考音频需要同时提供对应文本：

```bash
cd src
TTS_REF_AUDIO=../models/voices/young-female.wav \
TTS_REF_TEXT=你好，我是一个声音清晰自然的年轻女性。 \
TTS_TEMPERATURE=0 \
uv run server.py
```

注意：如果使用多行环境变量，必须使用反斜杠连接到 `uv run server.py`，或先 `export`。单独写一行 `TTS_REF_AUDIO=...` 不会传入子进程。

## 流式 TTS

Qwen TTS 支持真正的音频块流式输出。后端会优先使用 `stream_generate()`，第一块 PCM 生成后立即发送给浏览器。

```bash
TTS_STREAMING_INTERVAL=0.5 uv run server.py
```

`TTS_STREAMING_INTERVAL` 越小，首包可能更快，但块更多；越大，块更少但首包等待更长。默认值是 `0.5` 秒。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | 项目 `models/`，否则自动下载 E2B 到 `models/` | `.litertlm` 文件或目录路径 |
| `TTS_MODEL_PATH` | Apple Silicon 上为 `models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | 本地 mlx-audio TTS 模型目录 |
| `TTS_VOICE_INSTRUCT` | `年轻女性，普通话标准，声音清晰自然，音色稳定，语速适中。` | VoiceDesign 音色描述 |
| `TTS_VOICE_LOCK` | `1` | 生成一段参考音频，并通过 ICL 复用固定音色 |
| `TTS_VOICE_LOCK_TEXT` | `你好，我是你的语音助手，声音清晰自然，音色稳定。` | 生成 voice lock 参考音频时使用的文本 |
| `TTS_VOICE_LOCK_MAX_TOKENS` | `768` | 生成 voice lock 参考音频的最大 token 数 |
| `TTS_TEMPERATURE` | `0` | TTS 使用 greedy 解码，降低音色和语气随机性 |
| `TTS_STREAMING_INTERVAL` | `0.5` | Qwen TTS 流式音频块间隔，单位秒 |
| `TTS_REF_AUDIO` / `TTS_REF_TEXT` | 未设置 | 可选的真实参考音频和对应文本 |
| `PORT` | `8000` | 服务端口 |

## VAD 配置

当前浏览器端 VAD 使用 `@ricky0123/vad-web@0.0.29`：

```js
positiveSpeechThreshold: 0.35,
negativeSpeechThreshold: 0.20,
redemptionMs: 800,
minSpeechMs: 300,
preSpeechPadMs: 300,
```

AI 说话时会临时提高触发阈值；回到 listening 时恢复为 `0.35`：

```js
positiveSpeechThreshold: 0.92
```

并在 TTS 播放结束后保留 `1200ms` 冷却时间，降低回声误触发。

## 项目结构

```
src/
├── server.py              # FastAPI WebSocket 服务 + Gemma 推理
├── tts.py                 # TTS 后端选择和 Qwen/Kokoro 适配
├── tts_stream.py          # WebSocket TTS 流式发送
├── response_utils.py      # 模型回复清理、句子切分、fallback 判断
├── index.html             # 前端 UI、VAD、摄像头、音频播放
├── pyproject.toml         # Python 依赖
└── benchmarks/
    ├── bench.py           # 端到端 WebSocket benchmark
    └── benchmark_tts.py   # TTS backend benchmark
```

## 致谢

- [fikrikarim/parlor](https://github.com/fikrikarim/parlor)：本项目的原始实现
- [Gemma](https://ai.google.dev/gemma)：Google DeepMind 的本地模型
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)：Google AI Edge 的本地推理运行时
- [Qwen3 TTS](https://huggingface.co/Qwen)：本地中文 TTS 模型
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)：Apple Silicon 上的 MLX 音频推理支持
- [Silero VAD](https://github.com/snakers4/silero-vad)：浏览器端语音活动检测

## 许可证

[Apache 2.0](LICENSE)
