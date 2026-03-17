"""STT normalization — convert provider-specific responses to a unified format.

Transcript lifecycle::

    PARTIAL  →  STABLE  →  FINAL
    (interim)   (3 word repeats)  (provider confirmed)

Stability is tracked per-word.  As the transcript grows, earlier words
can become stable while the trailing edge is still partial.  Use
``alt.stable_text`` for the stable prefix and ``alt.text`` for the full
transcript including partial words.

Usage::

    from stt_normalization import (
        SpeechEvent,
        SpeechData,
        TranscriptType,
        normalize_speechmatics,
        normalize_deepgram,
    )

    # Stateful (recommended) — handles PARTIAL → STABLE promotion
    sm_norm = normalize_speechmatics.SpeechmaticsNormalizer(language="en")
    event = sm_norm.on_partial(raw_msg)   # PARTIAL or STABLE
    event = sm_norm.on_final(raw_msg)     # FINAL

    dg_norm = normalize_deepgram.DeepgramNormalizer(language="en")
    event = dg_norm.on_live_result(raw_msg)  # PARTIAL, STABLE, or FINAL

    # Both produce the same SpeechEvent type
    for alt in event.alternatives:
        print(alt.text, alt.transcript_type)
"""

from .types import (
    RecognitionUsage,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    StabilityDetector,
    TimedWord,
    TranscriptType,
)

from . import normalize_deepgram, normalize_speechmatics

__all__ = [
    "RecognitionUsage",
    "SpeechData",
    "SpeechEvent",
    "SpeechEventType",
    "StabilityDetector",
    "TimedWord",
    "TranscriptType",
    "normalize_deepgram",
    "normalize_speechmatics",
]
