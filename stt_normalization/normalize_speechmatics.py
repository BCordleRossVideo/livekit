"""Normalize Speechmatics STT responses into unified SpeechEvent types.

Speechmatics produces two kinds of transcript messages over its WebSocket:

1. **AddPartialTranscript** – interim results while the user is still
   speaking.
2. **AddTranscript** – final, immutable results once a segment is confirmed.

Both share this payload shape::

    {
        "metadata": {"start_time": 1.23, "end_time": 2.45},
        "results": [
            {"type": "word", "content": "hello",
             "start_time": 1.23, "end_time": 1.50},
            ...
        ]
    }

Transcript lifecycle
~~~~~~~~~~~~~~~~~~~~
The :class:`SpeechmaticsNormalizer` uses a :class:`StabilityDetector` to
promote partials to stable after 3 consecutive identical texts:

- ``AddPartialTranscript`` → **PARTIAL** (or **STABLE** after 3 repeats)
- ``AddTranscript``        → **FINAL**

Reference:
    https://docs.speechmatics.com/introduction/rt-api-ref
"""

from __future__ import annotations

from typing import Any

from .types import (
    RecognitionUsage,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    StabilityDetector,
    TimedWord,
    TranscriptType,
)


# ---------------------------------------------------------------------------
# Stateless helpers — one-shot conversion (caller manages transcript type)
# ---------------------------------------------------------------------------


def normalize_transcript(
    data: dict[str, Any],
    *,
    transcript_type: TranscriptType,
    language: str = "en",
    start_time_offset: float = 0.0,
) -> SpeechEvent:
    """Convert a Speechmatics transcript message to a :class:`SpeechEvent`.

    Args:
        data: Raw JSON dict from the Speechmatics WebSocket.
        transcript_type: Whether this is PARTIAL, STABLE, or FINAL.
        language: Fallback language code if not present in the payload.
        start_time_offset: Offset (seconds) added to all timestamps.
    """
    metadata = data.get("metadata", {})
    results: list[dict[str, Any]] = data.get("results", [])

    text_parts: list[str] = []
    words: list[TimedWord] = []
    for r in results:
        if r.get("type") != "word":
            continue
        word_text = r.get("content", "")
        text_parts.append(word_text)
        words.append(
            TimedWord(
                text=word_text,
                start_time=r.get("start_time", 0.0) + start_time_offset,
                end_time=r.get("end_time", 0.0) + start_time_offset,
            )
        )

    text = " ".join(text_parts)

    # Speechmatics does not return per-segment confidence; use 1.0 for
    # final results and 0.0 for partials.
    confidence = 1.0 if transcript_type == TranscriptType.FINAL else 0.0

    speech_data = SpeechData(
        language=data.get("language", language),
        text=text,
        transcript_type=transcript_type,
        start_time=metadata.get("start_time", 0.0) + start_time_offset,
        end_time=metadata.get("end_time", 0.0) + start_time_offset,
        confidence=confidence,
        speaker_id=data.get("speaker", data.get("speaker_id", None)),
        result=data,
        words=words if words else None,
    )

    return SpeechEvent(
        type=SpeechEventType.TRANSCRIPT,
        alternatives=[speech_data],
    )


def normalize_recognition_usage(audio_duration: float) -> SpeechEvent:
    """Create a ``RECOGNITION_USAGE`` event for billing/metrics tracking."""
    return SpeechEvent(
        type=SpeechEventType.RECOGNITION_USAGE,
        recognition_usage=RecognitionUsage(audio_duration=audio_duration),
    )


# ---------------------------------------------------------------------------
# Stateful normalizer — tracks stability across partials
# ---------------------------------------------------------------------------


class SpeechmaticsNormalizer:
    """Stateful normalizer with automatic PARTIAL → STABLE promotion.

    Usage::

        norm = SpeechmaticsNormalizer(language="en")

        for msg_type, data in websocket_messages:
            if msg_type == "AddPartialTranscript":
                event = norm.on_partial(data)
            elif msg_type == "AddTranscript":
                event = norm.on_final(data)

            print(event.alternatives[0].transcript_type)
            # → PARTIAL, PARTIAL, STABLE, ..., FINAL
    """

    def __init__(
        self,
        *,
        language: str = "en",
        start_time_offset: float = 0.0,
        stable_threshold: int = 3,
    ) -> None:
        self._language = language
        self._start_time_offset = start_time_offset
        self._detector = StabilityDetector(threshold=stable_threshold)

    def on_partial(self, data: dict[str, Any]) -> SpeechEvent:
        """Handle ``AddPartialTranscript``.

        Returns a PARTIAL event, or STABLE if the same text has appeared
        ``stable_threshold`` times in a row.
        """
        text = _extract_text(data)
        transcript_type = self._detector.on_text(text, provider_is_final=False)

        return normalize_transcript(
            data,
            transcript_type=transcript_type,
            language=self._language,
            start_time_offset=self._start_time_offset,
        )

    def on_final(self, data: dict[str, Any]) -> SpeechEvent:
        """Handle ``AddTranscript``. Always returns a FINAL event."""
        text = _extract_text(data)
        transcript_type = self._detector.on_text(text, provider_is_final=True)

        return normalize_transcript(
            data,
            transcript_type=transcript_type,
            language=self._language,
            start_time_offset=self._start_time_offset,
        )


def _extract_text(data: dict[str, Any]) -> str:
    """Build transcript text from the ``results`` word list."""
    results: list[dict[str, Any]] = data.get("results", [])
    return " ".join(
        r.get("content", "") for r in results if r.get("type") == "word"
    )
