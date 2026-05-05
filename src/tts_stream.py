"""WebSocket TTS streaming helpers."""

import asyncio
import base64
from dataclasses import dataclass
import json
import time

import numpy as np


_END_OF_STREAM = object()
_INTERRUPTED = object()


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


def _generate_once_or_end(tts_backend, sentence: str):
    pcm = tts_backend.generate(sentence)
    if np.asarray(pcm).size == 0:
        return _END_OF_STREAM
    return pcm


async def _next_chunk_or_interrupt(loop, chunks, interrupted: asyncio.Event, poll_interval: float):
    future = loop.run_in_executor(None, lambda: _next_chunk_or_end(chunks))
    while True:
        done, _pending = await asyncio.wait({future}, timeout=poll_interval)
        if done:
            return future.result(), None
        if interrupted.is_set():
            return _INTERRUPTED, future


async def _stream_tts_sentences_unlocked(
    ws,
    tts_backend,
    sentences: list[str],
    interrupted: asyncio.Event,
    poll_interval: float,
    job_id: str | None = None,
) -> tuple[float, asyncio.Future | None]:
    """Generate and stream TTS audio, announcing playback only after audio exists."""
    tts_start = time.time()
    audio_started = False
    chunk_count = 0
    sample_count = 0
    peak_amplitude = 0.0
    loop = asyncio.get_event_loop()

    for i, sentence in enumerate(sentences):
        if interrupted.is_set():
            print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
            break

        chunks = _stream_chunks(tts_backend, sentence)
        chunk_index = 0
        while not interrupted.is_set():
            pcm, pending_future = await _next_chunk_or_interrupt(loop, chunks, interrupted, poll_interval)
            if pcm is _INTERRUPTED:
                tts_time = time.time() - tts_start
                print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                return tts_time, pending_future
            if pcm is _END_OF_STREAM:
                if chunk_index > 0 or not hasattr(tts_backend, "generate"):
                    break
                pcm = await loop.run_in_executor(None, lambda: _generate_once_or_end(tts_backend, sentence))
                if pcm is _END_OF_STREAM:
                    break

            if interrupted.is_set():
                break

            pcm_array = np.asarray(pcm).reshape(-1)
            chunk_count += 1
            sample_count += pcm_array.size
            if pcm_array.size:
                peak_amplitude = max(peak_amplitude, float(np.max(np.abs(pcm_array))))

            if not audio_started:
                message = {
                    "type": "audio_start",
                    "sample_rate": tts_backend.sample_rate,
                    "sentence_count": len(sentences),
                }
                if job_id:
                    message["job_id"] = job_id
                await ws.send_text(json.dumps(message))
                audio_started = True

            message = {
                "type": "audio_chunk",
                "audio": _encode_pcm_chunk(pcm),
                "index": i,
                "chunk_index": chunk_index,
            }
            if job_id:
                message["job_id"] = job_id
            await ws.send_text(json.dumps(message))
            chunk_index += 1

    tts_time = time.time() - tts_start
    print(
        f"TTS ({tts_time:.2f}s): {len(sentences)} sentences, "
        f"chunks={chunk_count}, samples={sample_count}, peak={peak_amplitude:.4f}"
    )

    if not interrupted.is_set():
        message = {
            "type": "audio_end",
            "tts_time": round(tts_time, 2),
            "skipped": not audio_started,
        }
        if job_id:
            message["job_id"] = job_id
        await ws.send_text(json.dumps(message))

    return tts_time, None


async def stream_tts_sentences(
    ws,
    tts_backend,
    sentences: list[str],
    interrupted: asyncio.Event,
    generation_lock: asyncio.Lock | None = None,
    poll_interval: float = 0.05,
    job_id: str | None = None,
) -> float:
    """Generate and stream TTS audio, optionally serializing backend generation."""
    if generation_lock is None:
        tts_time, _pending_future = await _stream_tts_sentences_unlocked(
            ws, tts_backend, sentences, interrupted, poll_interval, job_id=job_id
        )
        return tts_time

    await generation_lock.acquire()
    release_now = True
    try:
        if interrupted.is_set():
            return 0.0
        tts_time, pending_future = await _stream_tts_sentences_unlocked(
            ws, tts_backend, sentences, interrupted, poll_interval, job_id=job_id
        )
        if pending_future is not None:
            release_now = False
            pending_future.add_done_callback(lambda _future: generation_lock.release())
        return tts_time
    finally:
        if release_now:
            generation_lock.release()


@dataclass
class TTSJob:
    ws: object
    sentences: list[str]
    interrupted: asyncio.Event
    job_id: str


class TTSWorker:
    """Single-worker TTS pipeline that keeps only the latest pending job."""

    def __init__(self, tts_backend, poll_interval: float = 0.05):
        self.tts_backend = tts_backend
        self.poll_interval = poll_interval
        self.queue = asyncio.Queue(maxsize=1)
        self.current_job: TTSJob | None = None
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._task = asyncio.create_task(self._run())

    async def submit(self, job: TTSJob):
        if self.current_job is not None:
            self.current_job.interrupted.set()

        while not self.queue.empty():
            stale = self.queue.get_nowait()
            stale.interrupted.set()
            self.queue.task_done()

        self._idle.clear()
        await self.queue.put(job)

    def interrupt(self):
        if self.current_job is not None:
            self.current_job.interrupted.set()
        while not self.queue.empty():
            stale = self.queue.get_nowait()
            stale.interrupted.set()
            self.queue.task_done()
        if self.current_job is None:
            self._idle.set()

    async def wait_until_idle(self):
        await self._idle.wait()

    async def close(self):
        self._closed = True
        self.interrupt()
        await self.queue.put(None)
        await self._task

    async def _run(self):
        while True:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return

            # Give a burst of submissions a chance to collapse to the latest job.
            await asyncio.sleep(0)
            while not self.queue.empty():
                newer = self.queue.get_nowait()
                if newer is None:
                    self.queue.task_done()
                    self.queue.task_done()
                    return
                job.interrupted.set()
                self.queue.task_done()
                job = newer

            self.current_job = job
            try:
                if not job.interrupted.is_set():
                    await stream_tts_sentences(
                        job.ws,
                        self.tts_backend,
                        job.sentences,
                        job.interrupted,
                        poll_interval=self.poll_interval,
                        job_id=job.job_id,
                    )
            finally:
                self.current_job = None
                self.queue.task_done()
                if self.queue.empty():
                    self._idle.set()
