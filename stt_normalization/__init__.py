"""STT normalization — convert provider-specific responses to a unified format.

Usage::

    from stt_normalization import (
        SpeechEvent,
        SpeechData,
        SpeechEventType,
        normalize_speechmatics,
        normalize_deepgram,
    )

    # Speechmatics streaming partial
    event = normalize_speechmatics.normalize_partial_transcript(raw_msg)

    # Deepgram live result
    event = normalize_deepgram.normalize_live_transcript(raw_msg)

    # Both produce the same SpeechEvent type
    for alt in event.alternatives:
        print(alt.text, alt.confidence)
"""

from .types import (
    RecognitionUsage,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    TimedWord,
)

from . import normalize_deepgram, normalize_speechmatics

__all__ = [
    "RecognitionUsage",
    "SpeechData",
    "SpeechEvent",
    "SpeechEventType",
    "TimedWord",
    "normalize_deepgram",
    "normalize_speechmatics",
]
