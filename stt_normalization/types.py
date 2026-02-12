"""Normalized transcript types for STT provider responses.

These types mirror the LiveKit agents SDK's speech event model
(see https://github.com/livekit/agents -> livekit/agents/stt/stt.py)
and the LiveKit protocol's Transcription/TranscriptionSegment protobuf messages
(see livekit_models.proto).

All STT providers (Speechmatics, Deepgram, etc.) should normalize their
responses into these types so downstream consumers have a single interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique


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
    """A single word with start/end timestamps (seconds)."""

    text: str
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class SpeechData:
    """A single transcription alternative.

    Fields align with the LiveKit protocol's ``TranscriptionSegment``:
      - language  -> TranscriptionSegment.language
      - text      -> TranscriptionSegment.text
      - start_time / end_time -> TranscriptionSegment.start_time / end_time
      - confidence -> (no direct proto field; carried in-band)
      - is_final  -> TranscriptionSegment.final
    """

    language: str
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    speaker_id: str | None = None
    is_final: bool = False
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
