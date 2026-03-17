"""Normalized transcript types for STT provider responses.

Every STT provider (Speechmatics, Deepgram, etc.) normalizes its responses
into these types so downstream consumers have a single interface.

Transcript lifecycle
--------------------
Each transcript passes through three states:

1. **PARTIAL** — ephemeral interim result.  At least one word has not yet
   stabilized.  Display-only; do not act on it.
2. **STABLE** — every word in the transcript has appeared in the same
   position for 3 consecutive partials.  Safe to accumulate, but the
   user may still be mid-sentence.
3. **FINAL** — the provider has explicitly confirmed end-of-utterance.
   This is the trigger for downstream actions (e.g. send to an LLM).

Stability is tracked **per-word**: as the transcript grows, earlier words
can become stable while the trailing edge is still partial.
``SpeechData.stable_text`` contains only the stable prefix, and each
``TimedWord`` carries its own ``is_stable`` flag.

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
    """A single word with start/end timestamps (seconds from stream start).

    ``is_stable`` indicates whether this word has survived enough
    consecutive partials to be considered locked in.
    """

    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    is_stable: bool = False


@dataclass
class SpeechData:
    """A single transcription alternative.

    ``transcript_type`` tells you whether this is a partial, stable, or
    final result — see the module docstring for the full lifecycle.
    """

    language: str | None
    text: str
    transcript_type: TranscriptType = TranscriptType.PARTIAL
    stable_text: str = ""
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

STABLE_THRESHOLD = 3  # consecutive partials a word must survive


@dataclass
class WordStability:
    """Per-word tracking state (internal to StabilityDetector)."""

    text: str
    count: int = 1
    is_stable: bool = False


class StabilityDetector:
    """Word-level stability tracker.

    Tracks each word by position across consecutive partials.  A word
    becomes **stable** once it has appeared in the same position for
    ``threshold`` consecutive partials.  Once stable, a word stays
    stable until the detector is reset (on FINAL).

    The overall transcript is STABLE when *every* word is stable,
    PARTIAL when at least one word is not yet stable, and FINAL when
    the provider explicitly says so.

    Example::

        detector = StabilityDetector(threshold=3)

        detector.on_words(["hell"])                   # → PARTIAL  stable=[]
        detector.on_words(["hello"])                   # → PARTIAL  stable=[]
        detector.on_words(["hello", "wo"])             # → PARTIAL  stable=[]
        detector.on_words(["hello", "world"])          # → PARTIAL  stable=["hello"]
        detector.on_words(["hello", "world"])          # → PARTIAL  stable=["hello"]
        detector.on_words(["hello", "world"])          # → STABLE   stable=["hello", "world"]
        detector.on_words(["hello", "world", "how"])   # → PARTIAL  stable=["hello", "world"]
        detector.on_words(["hello", "world"], final=True)  # → FINAL
    """

    def __init__(self, threshold: int = STABLE_THRESHOLD) -> None:
        self._threshold = threshold
        self._words: list[WordStability] = []

    def on_words(
        self,
        words: list[str],
        *,
        final: bool = False,
    ) -> tuple[TranscriptType, list[bool]]:
        """Update stability state and return per-word flags.

        Args:
            words: The list of word strings from the current partial.
            final: ``True`` when the provider flagged this as final.

        Returns:
            A tuple of ``(transcript_type, stable_flags)`` where
            ``stable_flags[i]`` is ``True`` when word *i* is stable.
        """
        if final:
            stable_flags = [True] * len(words)
            self._reset()
            return TranscriptType.FINAL, stable_flags

        new_state: list[WordStability] = []

        for i, word in enumerate(words):
            if i < len(self._words) and self._words[i].text == word:
                # Same word in same position — increment or keep stable.
                prev = self._words[i]
                new_count = prev.count + 1
                is_stable = prev.is_stable or new_count >= self._threshold
                new_state.append(
                    WordStability(text=word, count=new_count, is_stable=is_stable)
                )
            elif i < len(self._words) and self._words[i].is_stable:
                # Position had a *different* stable word — word changed,
                # so this is a correction; reset this position.
                new_state.append(WordStability(text=word, count=1))
            else:
                # New position or word changed before becoming stable.
                new_state.append(WordStability(text=word, count=1))

        self._words = new_state

        stable_flags = [w.is_stable for w in self._words]
        all_stable = bool(stable_flags) and all(stable_flags)

        return (
            TranscriptType.STABLE if all_stable else TranscriptType.PARTIAL,
            stable_flags,
        )

    @property
    def stable_prefix(self) -> list[str]:
        """Return the longest leading run of stable words."""
        result: list[str] = []
        for w in self._words:
            if w.is_stable:
                result.append(w.text)
            else:
                break
        return result

    def _reset(self) -> None:
        self._words = []
