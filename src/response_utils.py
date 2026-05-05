"""Response normalization helpers for text and TTS output."""

import re

SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？])\s*|(?<=[.!?])\s+')
SHORT_TTS_SEGMENT_CHARS = 18
MAX_TTS_SEGMENT_CHARS = 80
MODEL_DELIMITER = '<|"|>'
FALLBACK_RESPONSE = "I didn't catch that. Could you say it again?"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def normalize_response_text(text: str | None) -> str:
    """Return speakable assistant text, even when the model returns an empty response."""
    cleaned = (text or "").replace(MODEL_DELIMITER, "").strip()
    return cleaned or FALLBACK_RESPONSE


def extract_raw_response_text(response: dict) -> str:
    """Extract fallback text from a non-tool LLM response."""
    content = response.get("content") or []
    if not content:
        return FALLBACK_RESPONSE
    return normalize_response_text(content[0].get("text") if isinstance(content[0], dict) else None)


def sentences_for_tts(text: str) -> list[str]:
    """Return non-empty sentences that are safe to send to the TTS backend."""
    sentences = split_sentences(text)
    if not any(_contains_cjk(sentence) for sentence in sentences):
        return sentences
    return _merge_short_tts_segments(sentences)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _merge_short_tts_segments(
    sentences: list[str],
    min_chars: int = SHORT_TTS_SEGMENT_CHARS,
    max_chars: int = MAX_TTS_SEGMENT_CHARS,
) -> list[str]:
    """Merge short CJK TTS segments so Qwen3 ICL does not emit near-silence."""
    merged = []
    current = ""

    for sentence in sentences:
        candidate = current + sentence
        if not current:
            current = sentence
            continue

        if len(current) < min_chars or len(candidate) <= max_chars:
            current = candidate
            continue

        merged.append(current)
        current = sentence

    if current:
        if merged and len(current) < min_chars:
            merged[-1] += current
        else:
            merged.append(current)

    return merged


def should_stream_tts(used_tool: bool, text: str) -> bool:
    """Only synthesize speech for valid assistant responses produced through the tool."""
    return used_tool and bool(sentences_for_tts(text))


def is_litert_tool_parse_error(error: Exception) -> bool:
    """Return true when LiteRT failed to parse a malformed model tool call."""
    message = str(error)
    return (
        "Failed to parse tool calls from response" in message
        or "Failed to parse FC tool calls" in message
    )
