"""WebSocket TTS streaming helpers."""

import asyncio
import base64
import json
import time

import numpy as np


_END_OF_STREAM = object()


def _next_chunk_or_end(chunks):
    try:
        return next(chunks)
    except StopIteration:
        return _END_OF_STREAM


def _encode_pcm_chunk(pcm) -> str:
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    return base64.b64encode(pcm_int16.tobytes()).decode()


def _stream_chunks(tts_backend, sentence: str):
    if hasattr(tts_backend, "stream_generate"):
        return tts_backend.stream_generate(sentence)

    def generate_once():
        yield tts_backend.generate(sentence)

    return generate_once()


async def _stream_tts_sentences_unlocked(ws, tts_backend, sentences: list[str], interrupted: asyncio.Event) -> float:
    """Generate and stream TTS audio, announcing playback only after audio exists."""
    tts_start = time.time()
    audio_started = False
    loop = asyncio.get_event_loop()

    for i, sentence in enumerate(sentences):
        if interrupted.is_set():
            print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
            break

        chunks = _stream_chunks(tts_backend, sentence)
        chunk_index = 0
        while not interrupted.is_set():
            pcm = await loop.run_in_executor(None, lambda: _next_chunk_or_end(chunks))
            if pcm is _END_OF_STREAM:
                break

            if interrupted.is_set():
                break

            if not audio_started:
                await ws.send_text(json.dumps({
                    "type": "audio_start",
                    "sample_rate": tts_backend.sample_rate,
                    "sentence_count": len(sentences),
                }))
                audio_started = True

            await ws.send_text(json.dumps({
                "type": "audio_chunk",
                "audio": _encode_pcm_chunk(pcm),
                "index": i,
                "chunk_index": chunk_index,
            }))
            chunk_index += 1

    tts_time = time.time() - tts_start
    print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

    if not interrupted.is_set():
        await ws.send_text(json.dumps({
            "type": "audio_end",
            "tts_time": round(tts_time, 2),
            "skipped": not audio_started,
        }))

    return tts_time


async def stream_tts_sentences(
    ws,
    tts_backend,
    sentences: list[str],
    interrupted: asyncio.Event,
    generation_lock: asyncio.Lock | None = None,
) -> float:
    """Generate and stream TTS audio, optionally serializing backend generation."""
    if generation_lock is None:
        return await _stream_tts_sentences_unlocked(ws, tts_backend, sentences, interrupted)

    async with generation_lock:
        if interrupted.is_set():
            return 0.0
        return await _stream_tts_sentences_unlocked(ws, tts_backend, sentences, interrupted)
