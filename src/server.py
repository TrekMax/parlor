"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import litert_lm
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import assistant_tools
import model_config
import response_utils
import tts
import tts_stream

MODEL_PATH = model_config.resolve_model_path()
SYSTEM_PROMPT = (
    "你是一个友好、善于交谈的AI助手。用户正在通过麦克风与你对话（用户可能会多种语言穿插对话），并且正在用摄像头给你展示画面。\n\n"
    "你**必须始终只使用 respond_to_user 工具**来回复用户。\n\n"
    "当用户消息包含音频时，请根据音频内容转述用户说的话并回复；当消息同时包含图像时，只有在和用户问题相关时才结合画面。\n\n"
    "请按以下两步执行：\n"
    "1. 首先，逐字转述用户说的话\n"
    "2. 然后，写出你的回答\n\n"
)

engine = None
tts_backend = None
active_session_lock = asyncio.Lock()


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


def unload_models():
    global engine, tts_backend
    if engine is not None:
        engine.__exit__(None, None, None)
        engine = None
    tts_backend = None


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    try:
        yield
    finally:
        await asyncio.get_event_loop().run_in_executor(None, unload_models)


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if active_session_lock.locked():
        await ws.send_text(json.dumps({
            "type": "text",
            "text": "当前已有一个语音会话在运行，请关闭其它页面或等待当前会话结束后再试。",
            "llm_time": 0,
        }))
        await ws.send_text(json.dumps({
            "type": "audio_end",
            "tts_time": 0,
            "skipped": True,
        }))
        await ws.close(code=1013)
        return

    await active_session_lock.acquire()

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
    tts_worker = tts_stream.TTSWorker(tts_backend)

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    tts_worker.interrupt()
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

            if not msg.get("audio") and not msg.get("image"):
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
                lookup_context = assistant_tools.lookup_request_context(transcription)
                if lookup_context:
                    tool_result.clear()
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: conversation.send_message({
                            "role": "user",
                            "content": (
                                "请基于以下实时查询结果，用自然、温暖、口语化的方式回复用户。"
                                "不要机械罗列全部字段，只保留用户最关心的信息，并给一句贴心建议。"
                                f"\n用户原话：{transcription}"
                                f"\n实时查询结果：{lookup_context}"
                                f"\n你刚才的草稿回复：{text_response}"
                            ),
                        }),
                    )
                    llm_time = time.time() - t0
                    if tool_result:
                        text_response = response_utils.normalize_response_text(tool_result.get("response", ""))
                        used_tool = True
                    else:
                        text_response = response_utils.extract_raw_response_text(response)
                        used_tool = False
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
            tts_interrupted = asyncio.Event()
            job_id = uuid.uuid4().hex

            await tts_worker.submit(
                tts_stream.TTSJob(
                    ws=ws,
                    sentences=sentences,
                    interrupted=tts_interrupted,
                    job_id=job_id,
                )
            )

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        tts_worker.interrupt()
        await tts_worker.close()
        recv_task.cancel()
        conversation.__exit__(None, None, None)
        active_session_lock.release()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
