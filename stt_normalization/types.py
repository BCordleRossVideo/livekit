"""Normalized transcript types for STT provider responses.

These types combine the best of both normalization approaches:

- **LiveKit agents SDK** (``livekit/agents/stt/stt.py``): Multiple
  alternatives with per-alternative confidence and word-level timing.
- **Pipecat** (``pipecat/frames/frames.py``): Two-phase finalization
  protocol, ISO 8601 wall-clock timestamps, and raw result preservation.

All STT providers (Speechmatics, Deepgram, etc.) should normalize their
responses into these types so downstream consumers have a single interface.

Consumer contract
-----------------
1. **Interim** (``is_final=False``): ephemeral, display-only. May be
   replaced by later events.
2. **Final, not finalized** (``is_final=True, finalized=False``): text is
   stable but the utterance may not be complete.  Safe to accumulate.
3. **Final + finalized** (``is_final=True, finalized=True``): the user is
   done talking.  This is the trigger for downstream actions (e.g. send to
   an LLM).
4. ``result`` always carries the raw provider response for debugging or
   provider-specific data (word timings, confidence, etc.).
5. ``language`` may be ``None`` when the provider did not detect one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any


def _now_iso8601() -> str:
    """UTC timestamp with millisecond precision, matching Pipecat's ``time_now_iso8601()``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@unique
class SpeechEventType(str, Enum):
    """Type of speech event produced by an STT provider."""

    START_OF_SPEECH = "start_of_speech"
    INTERIM_TRANSCRIPT = "interim_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
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

    Merges LiveKit's rich per-alternative metadata (confidence, word timing)
    with Pipecat's raw result preservation and ISO timestamp.

    Fields mapped to LiveKit protocol ``TranscriptionSegment``:
      - ``language``  -> ``TranscriptionSegment.language``
      - ``text``      -> ``TranscriptionSegment.text``
      - ``start_time`` / ``end_time`` -> ``TranscriptionSegment.start_time`` / ``end_time``
      - ``is_final``  -> ``TranscriptionSegment.final``

    Fields from Pipecat:
      - ``timestamp`` -> ISO 8601 wall-clock time of normalization
      - ``result``    -> raw, unmodified provider response object
      - ``finalized`` -> two-phase finalization flag (see module docstring)
    """

    language: str | None
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    speaker_id: str | None = None
    is_final: bool = False
    finalized: bool = False
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

    Every STT normalizer should produce instances of this type.
    ``alternatives`` is ordered by confidence (highest first).
    """

    type: SpeechEventType
    request_id: str = ""
    alternatives: list[SpeechData] = field(default_factory=list)
    recognition_usage: RecognitionUsage | None = None
