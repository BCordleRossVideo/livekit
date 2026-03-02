"""Normalized transcript types for STT provider responses.

Every STT provider (Speechmatics, Deepgram, etc.) normalizes its responses
into these types so downstream consumers have a single interface.

Transcript lifecycle
--------------------
Each transcript passes through three states:

1. **PARTIAL** — ephemeral interim result.  May be replaced by the next
   partial.  Display-only; do not act on it.
2. **STABLE** — three consecutive partials produced the same text,
   so the transcript is unlikely to change.  Safe to accumulate or
   display with confidence, but the user may still be mid-sentence.
3. **FINAL** — the provider has explicitly confirmed end-of-utterance.
   This is the trigger for downstream actions (e.g. send to an LLM).

The ``result`` field always carries the raw provider response for
debugging or provider-specific data (word timings, per-word confidence,
etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any


def _now_iso8601() -> str:
    """UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@unique
class TranscriptType(str, Enum):
    """The three lifecycle states of a transcript."""

    PARTIAL = "partial"
    STABLE = "stable"
    FINAL = "final"


@unique
class SpeechEventType(str, Enum):
    """Type of speech event produced by an STT provider."""

    TRANSCRIPT = "transcript"
    START_OF_SPEECH = "start_of_speech"
    END_OF_SPEECH = "end_of_speech"
    RECOGNITION_USAGE = "recognition_usage"


@dataclass
class TimedWord:
    """A single word with start/end timestamps (seconds from stream start)."""

    text: str
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class SpeechData:
    """A single transcription alternative.

    ``transcript_type`` tells you whether this is a partial, stable, or
    final result — see the module docstring for the full lifecycle.
    """

    language: str | None
    text: str
    transcript_type: TranscriptType = TranscriptType.PARTIAL
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    speaker_id: str | None = None
    timestamp: str = field(default_factory=_now_iso8601)
    result: Any = None
    words: list[TimedWord] | None = None


@dataclass
class RecognitionUsage:
    """Tracks how much audio was processed (seconds)."""

    audio_duration: float = 0.0


@dataclass
class SpeechEvent:
    """Top-level normalized speech event.

    For ``TRANSCRIPT`` events, ``alternatives`` is ordered by confidence
    (highest first) and each alternative carries a ``transcript_type``
    of ``PARTIAL``, ``STABLE``, or ``FINAL``.
    """

    type: SpeechEventType
    request_id: str = ""
    alternatives: list[SpeechData] = field(default_factory=list)
    recognition_usage: RecognitionUsage | None = None


# ---------------------------------------------------------------------------
# Stability detector — shared by all provider normalizers
# ---------------------------------------------------------------------------

STABLE_THRESHOLD = 3  # consecutive identical partials required


class StabilityDetector:
    """Promotes a PARTIAL to STABLE after N consecutive identical texts.

    Usage::

        detector = StabilityDetector()

        # On each provider message:
        transcript_type = detector.on_text(text, provider_is_final=False)
        # returns PARTIAL or STABLE

        # When the provider flags a result as final:
        transcript_type = detector.on_text(text, provider_is_final=True)
        # returns FINAL and resets the counter
    """

    def __init__(self, threshold: int = STABLE_THRESHOLD) -> None:
        self._threshold = threshold
        self._last_text: str | None = None
        self._repeat_count: int = 0

    def on_text(self, text: str, *, provider_is_final: bool) -> TranscriptType:
        """Determine the transcript type for *text*.

        Args:
            text: The transcribed text from the provider.
            provider_is_final: ``True`` when the provider has flagged this
                result as final / end-of-utterance.

        Returns:
            ``FINAL`` if the provider says so, ``STABLE`` if the same text
            has appeared ``threshold`` times in a row, otherwise ``PARTIAL``.
        """
        if provider_is_final:
            self._reset()
            return TranscriptType.FINAL

        if text == self._last_text:
            self._repeat_count += 1
        else:
            self._last_text = text
            self._repeat_count = 1

        if self._repeat_count >= self._threshold:
            self._reset()
            return TranscriptType.STABLE

        return TranscriptType.PARTIAL

    def _reset(self) -> None:
        self._last_text = None
        self._repeat_count = 0
