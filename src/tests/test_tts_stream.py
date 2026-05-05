import asyncio
import json
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

    async def test_interrupted_before_audio_sends_no_start(self):
        ws = FakeWebSocket()
        interrupted = asyncio.Event()
        interrupted.set()

        await tts_stream.stream_tts_sentences(ws, FakeTTSBackend([]), ["你好"], interrupted)

        self.assertEqual(ws.messages, [])


if __name__ == "__main__":
    unittest.main()
