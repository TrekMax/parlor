import asyncio
import json
import time
import unittest

import numpy as np

import tts_stream


class FakeWebSocket:
    def __init__(self, events=None):
        self.messages = []
        self.events = events

    async def send_text(self, text):
        message = json.loads(text)
        self.messages.append(message)
        if self.events is not None:
            self.events.append(f"send:{message['type']}")


class FakeTTSBackend:
    sample_rate = 24000

    def __init__(self, events):
        self.events = events

    def generate(self, text):
        self.events.append(f"generate:{text}")
        return np.array([0.0, 0.1], dtype=np.float32)


class FakeStreamingTTSBackend:
    sample_rate = 24000

    def __init__(self, events):
        self.events = events

    def stream_generate(self, text):
        self.events.append(f"stream:{text}:first")
        yield np.array([0.0], dtype=np.float32)
        self.events.append(f"stream:{text}:second")
        yield np.array([0.1], dtype=np.float32)


class FakeSlowStreamingTTSBackend:
    sample_rate = 24000

    def __init__(self, events):
        self.events = events

    def stream_generate(self, text):
        self.events.append(f"start:{text}")
        time.sleep(0.01)
        yield np.array([0.0], dtype=np.float32)
        self.events.append(f"end:{text}")


class FakeBlockingStreamingTTSBackend:
    sample_rate = 24000

    def __init__(self, events):
        self.events = events

    def stream_generate(self, text):
        self.events.append(f"start:{text}")
        time.sleep(0.3)
        yield np.array([0.0], dtype=np.float32)


class TTSStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_start_is_sent_after_first_audio_is_generated(self):
        events = []
        ws = FakeWebSocket(events)
        backend = FakeTTSBackend(events)

        await tts_stream.stream_tts_sentences(ws, backend, ["你好"], asyncio.Event())

        self.assertEqual(events[:2], ["generate:你好", "send:audio_start"])
        self.assertEqual([m["type"] for m in ws.messages], ["audio_start", "audio_chunk", "audio_end"])

    async def test_streaming_backend_sends_first_chunk_before_stream_finishes(self):
        events = []
        ws = FakeWebSocket(events)
        backend = FakeStreamingTTSBackend(events)

        await tts_stream.stream_tts_sentences(ws, backend, ["你好"], asyncio.Event())

        self.assertEqual(
            events[:4],
            ["stream:你好:first", "send:audio_start", "send:audio_chunk", "stream:你好:second"],
        )
        self.assertEqual([m["type"] for m in ws.messages], ["audio_start", "audio_chunk", "audio_chunk", "audio_end"])

    async def test_generation_lock_serializes_concurrent_streams(self):
        events = []
        backend = FakeSlowStreamingTTSBackend(events)
        lock = asyncio.Lock()

        await asyncio.gather(
            tts_stream.stream_tts_sentences(FakeWebSocket(), backend, ["一"], asyncio.Event(), generation_lock=lock),
            tts_stream.stream_tts_sentences(FakeWebSocket(), backend, ["二"], asyncio.Event(), generation_lock=lock),
        )

        self.assertIn(
            events,
            [
                ["start:一", "end:一", "start:二", "end:二"],
                ["start:二", "end:二", "start:一", "end:一"],
            ],
        )

    async def test_interrupt_returns_without_waiting_for_blocked_chunk(self):
        events = []
        backend = FakeBlockingStreamingTTSBackend(events)
        interrupted = asyncio.Event()
        lock = asyncio.Lock()
        task = asyncio.create_task(
            tts_stream.stream_tts_sentences(FakeWebSocket(), backend, ["一"], interrupted, generation_lock=lock)
        )

        while not events:
            await asyncio.sleep(0.001)

        started = time.time()
        interrupted.set()
        await task

        self.assertLess(time.time() - started, 0.1)
        self.assertTrue(lock.locked())
        await asyncio.sleep(0.35)
        self.assertFalse(lock.locked())

    async def test_interrupted_before_audio_sends_no_start(self):
        ws = FakeWebSocket()
        interrupted = asyncio.Event()
        interrupted.set()

        await tts_stream.stream_tts_sentences(ws, FakeTTSBackend([]), ["你好"], interrupted)

        self.assertEqual(ws.messages, [])

    async def test_stream_messages_include_job_id_when_provided(self):
        ws = FakeWebSocket()

        await tts_stream.stream_tts_sentences(ws, FakeTTSBackend([]), ["你好"], asyncio.Event(), job_id="job-1")

        self.assertEqual([m["job_id"] for m in ws.messages], ["job-1", "job-1", "job-1"])

    async def test_worker_processes_latest_job_and_skips_stale_job(self):
        events = []
        worker = tts_stream.TTSWorker(FakeTTSBackend(events))
        first = tts_stream.TTSJob(FakeWebSocket(events), ["旧回复"], asyncio.Event(), "job-1")
        second = tts_stream.TTSJob(FakeWebSocket(events), ["新回复"], asyncio.Event(), "job-2")

        await worker.submit(first)
        await worker.submit(second)
        await worker.wait_until_idle()
        await worker.close()

        self.assertNotIn("generate:旧回复", events)
        self.assertIn("generate:新回复", events)

    async def test_worker_interrupt_clears_pending_jobs(self):
        events = []
        worker = tts_stream.TTSWorker(FakeTTSBackend(events))
        job = tts_stream.TTSJob(FakeWebSocket(events), ["你好"], asyncio.Event(), "job-1")

        await worker.submit(job)
        worker.interrupt()
        await worker.wait_until_idle()
        await worker.close()

        self.assertTrue(job.interrupted.is_set())
        self.assertEqual(events, [])

    async def test_worker_submit_interrupts_current_job(self):
        events = []
        worker = tts_stream.TTSWorker(FakeBlockingStreamingTTSBackend(events), poll_interval=0.01)
        first = tts_stream.TTSJob(FakeWebSocket(events), ["旧回复"], asyncio.Event(), "job-1")
        second = tts_stream.TTSJob(FakeWebSocket(events), ["新回复"], asyncio.Event(), "job-2")

        await worker.submit(first)
        while not events:
            await asyncio.sleep(0.001)

        await worker.submit(second)

        self.assertTrue(first.interrupted.is_set())
        await worker.close()


if __name__ == "__main__":
    unittest.main()
