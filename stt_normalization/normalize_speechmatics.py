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
track word-level stability across consecutive partials:

- ``AddPartialTranscript`` → **PARTIAL** (or **STABLE** when every word
  has survived 3+ consecutive partials in its position)
- ``AddTranscript``        → **FINAL**

Individual words become stable independently, so ``stable_text`` may
contain a prefix like ``"hello"`` while the full ``text`` is
``"hello world"`` (where ``"world"`` is still partial).

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
    stable_flags: list[bool] | None = None,
    language: str = "en",
    start_time_offset: float = 0.0,
) -> SpeechEvent:
    """Convert a Speechmatics transcript message to a :class:`SpeechEvent`.

    Args:
        data: Raw JSON dict from the Speechmatics WebSocket.
        transcript_type: Whether this is PARTIAL, STABLE, or FINAL.
        stable_flags: Per-word stability flags from
            :meth:`StabilityDetector.on_words`.  If ``None``, all words
            inherit stability from *transcript_type* (stable if STABLE or
            FINAL, not stable if PARTIAL).
        language: Fallback language code if not present in the payload.
        start_time_offset: Offset (seconds) added to all timestamps.
    """
    metadata = data.get("metadata", {})
    results: list[dict[str, Any]] = data.get("results", [])

    word_texts: list[str] = []
    words: list[TimedWord] = []
    word_idx = 0
    for r in results:
        if r.get("type") != "word":
            continue
        word_text = r.get("content", "")
        word_texts.append(word_text)

        if stable_flags is not None and word_idx < len(stable_flags):
            is_stable = stable_flags[word_idx]
        else:
            is_stable = transcript_type in (TranscriptType.STABLE, TranscriptType.FINAL)

        words.append(
            TimedWord(
                text=word_text,
                start_time=r.get("start_time", 0.0) + start_time_offset,
                end_time=r.get("end_time", 0.0) + start_time_offset,
                is_stable=is_stable,
            )
        )
        word_idx += 1

    text = " ".join(word_texts)

    # Build stable_text from the leading run of stable words.
    stable_parts: list[str] = []
    for w in words:
        if w.is_stable:
            stable_parts.append(w.text)
        else:
            break
    stable_text = " ".join(stable_parts)

    # Speechmatics does not return per-segment confidence; use 1.0 for
    # final results and 0.0 for partials.
    confidence = 1.0 if transcript_type == TranscriptType.FINAL else 0.0

    speech_data = SpeechData(
        language=data.get("language", language),
        text=text,
        transcript_type=transcript_type,
        stable_text=stable_text,
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
# Stateful normalizer — tracks word-level stability across partials
# ---------------------------------------------------------------------------


class SpeechmaticsNormalizer:
    """Stateful normalizer with word-level PARTIAL → STABLE promotion.

    Usage::

        norm = SpeechmaticsNormalizer(language="en")

        for msg_type, data in websocket_messages:
            if msg_type == "AddPartialTranscript":
                event = norm.on_partial(data)
            elif msg_type == "AddTranscript":
                event = norm.on_final(data)

            alt = event.alternatives[0]
            print(f"{alt.transcript_type}: {alt.stable_text!r} | {alt.text!r}")
            # PARTIAL: '' | 'hell'
            # PARTIAL: '' | 'hello'
            # PARTIAL: '' | 'hello wo'
            # PARTIAL: 'hello' | 'hello world'
            # PARTIAL: 'hello' | 'hello world'
            # STABLE:  'hello world' | 'hello world'
            # FINAL:   'hello world' | 'hello world'
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

        Returns a PARTIAL event (or STABLE once every word has stabilized).
        Individual words may be stable even while the overall transcript
        is still PARTIAL — check ``stable_text`` or per-word ``is_stable``.
        """
        word_texts = _extract_words(data)
        transcript_type, stable_flags = self._detector.on_words(word_texts)

        return normalize_transcript(
            data,
            transcript_type=transcript_type,
            stable_flags=stable_flags,
            language=self._language,
            start_time_offset=self._start_time_offset,
        )

    def on_final(self, data: dict[str, Any]) -> SpeechEvent:
        """Handle ``AddTranscript``. Always returns a FINAL event."""
        word_texts = _extract_words(data)
        transcript_type, stable_flags = self._detector.on_words(
            word_texts, final=True
        )

        return normalize_transcript(
            data,
            transcript_type=transcript_type,
            stable_flags=stable_flags,
            language=self._language,
            start_time_offset=self._start_time_offset,
        )


def _extract_words(data: dict[str, Any]) -> list[str]:
    """Extract the list of word strings from a Speechmatics payload."""
    results: list[dict[str, Any]] = data.get("results", [])
    return [r.get("content", "") for r in results if r.get("type") == "word"]
