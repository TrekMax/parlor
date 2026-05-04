"""Response normalization helpers for text and TTS output."""

import re

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
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
    return split_sentences(text)


def should_stream_tts(used_tool: bool, text: str) -> bool:
    """Only synthesize speech for valid assistant responses produced through the tool."""
    return used_tool and bool(sentences_for_tts(text))
