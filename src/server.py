"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import litert_lm
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import model_config
import response_utils
import tts
import tts_stream

MODEL_PATH = model_config.resolve_model_path()
SYSTEM_PROMPT = (
    "你是一个友好、善于交谈的AI助手。用户正在通过麦克风与你对话（用户可能会多种语言穿插对话），并且正在用摄像头给你展示画面。\n\n"
    "你**必须始终使用 respond_to_user 工具**来回复用户。\n\n"
    "请按以下两步执行：\n"
    "1. 首先，逐字转述用户说的话\n"
    "2. 然后，写出你的回答\n\n"
)

engine = None
tts_backend = None
tts_generation_lock = asyncio.Lock()


def load_models():
    global engine, tts_backend
    print(f"Loading Gemma model from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU,
        vision_backend=litert_lm.Backend.GPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")

    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Per-connection tool state captured via closure
    tool_result = {}

    def respond_to_user(transcription: str, response: str) -> str:
        """Respond to the user's voice message.

        Args:
            transcription: Exact transcription of what the user said in the audio.
            response: Your conversational response to the user. Keep it to 1-4 short sentences.
        """
        tool_result["transcription"] = transcription
        tool_result["response"] = response
        return "OK"

    def create_conversation():
        return engine.create_conversation(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}],
            tools=[respond_to_user],
        )

    conversation = create_conversation()
    conversation.__enter__()

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()

            content = []
            if msg.get("audio"):
                content.append({"type": "audio", "blob": msg["audio"]})
            if msg.get("image"):
                content.append({"type": "image", "blob": msg["image"]})

            if msg.get("audio") and msg.get("image"):
                content.append({"type": "text", "text": "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant."})
            elif msg.get("audio"):
                content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
            elif msg.get("image"):
                content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
            else:
                content.append({"type": "text", "text": msg.get("text", "Hello!")})

            # LLM inference
            t0 = time.time()
            tool_result.clear()
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: conversation.send_message({"role": "user", "content": content})
                )
            except RuntimeError as e:
                llm_time = time.time() - t0
                if not response_utils.is_litert_tool_parse_error(e):
                    raise
                print(f"LLM ({llm_time:.2f}s) tool parse error; resetting conversation: {e}")
                conversation.__exit__(None, None, None)
                conversation = create_conversation()
                conversation.__enter__()
                await ws.send_text(json.dumps({
                    "type": "text",
                    "text": response_utils.FALLBACK_RESPONSE,
                    "llm_time": round(llm_time, 2),
                }))
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": 0,
                    "skipped": True,
                }))
                continue
            llm_time = time.time() - t0

            # Extract response from tool call or fallback to raw text
            used_tool = bool(tool_result)
            if tool_result:
                transcription = (
                    (tool_result.get("transcription", "") or "")
                    .replace(response_utils.MODEL_DELIMITER, "")
                    .strip()
                )
                text_response = response_utils.normalize_response_text(tool_result.get("response", ""))
                print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response}")
            else:
                transcription = None
                text_response = response_utils.extract_raw_response_text(response)
                print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
            if transcription:
                reply["transcription"] = transcription
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                print("Interrupted before TTS, skipping audio")
                continue

            if not response_utils.should_stream_tts(used_tool=used_tool, text=text_response):
                print("Skipping TTS for non-tool or empty response")
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": 0,
                    "skipped": True,
                }))
                continue

            # Streaming TTS: split into sentences and send chunks progressively
            sentences = response_utils.sentences_for_tts(text_response)

            await tts_stream.stream_tts_sentences(
                ws,
                tts_backend,
                sentences,
                interrupted,
                generation_lock=tts_generation_lock,
            )

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        conversation.__exit__(None, None, None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
