"""Normalize Speechmatics STT responses into unified SpeechEvent types.

Speechmatics produces two kinds of transcript messages over its WebSocket:

1. **AddPartialTranscript** – interim/partial results while the user is still
   speaking.  The payload looks like::

       {
           "metadata": {"start_time": 1.23, "end_time": 2.45},
           "results": [
               {"type": "word", "content": "hello", "start_time": 1.23, "end_time": 1.50},
               ...
           ]
       }

2. **AddTranscript** – final results once a segment is confirmed.  Same shape
   as above but considered immutable.

Speechmatics also emits **speaker diarization** data via ``AudioAdded`` and
turn-level events.  When diarization is enabled the messages include a
``speaker`` field (e.g. ``"S1"``).

This module converts both message shapes into the normalized
:class:`SpeechEvent` / :class:`SpeechData` types defined in ``types.py``.

Reference:
    https://docs.speechmatics.com/features/transcription
    https://github.com/livekit/agents (livekit-plugins-speechmatics)
"""

from __future__ import annotations

from typing import Any

from .types import (
    RecognitionUsage,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    TimedWord,
)


def normalize_partial_transcript(
    data: dict[str, Any],
    *,
    language: str = "en",
    start_time_offset: float = 0.0,
) -> SpeechEvent:
    """Convert a Speechmatics ``AddPartialTranscript`` message.

    Args:
        data: Raw JSON dict from the Speechmatics WebSocket.
        language: Fallback language code if not present in the payload.
        start_time_offset: Offset (seconds) to add to all timestamps so that
            they are relative to the beginning of the session rather than the
            beginning of the audio chunk.

    Returns:
        A :class:`SpeechEvent` with type ``INTERIM_TRANSCRIPT``.
    """
    return _to_speech_event(
        data,
        is_final=False,
        language=language,
        start_time_offset=start_time_offset,
    )


def normalize_final_transcript(
    data: dict[str, Any],
    *,
    language: str = "en",
    start_time_offset: float = 0.0,
) -> SpeechEvent:
    """Convert a Speechmatics ``AddTranscript`` message.

    Args:
        data: Raw JSON dict from the Speechmatics WebSocket.
        language: Fallback language code if not present in the payload.
        start_time_offset: Offset (seconds) to add to all timestamps.

    Returns:
        A :class:`SpeechEvent` with type ``FINAL_TRANSCRIPT``.
    """
    return _to_speech_event(
        data,
        is_final=True,
        language=language,
        start_time_offset=start_time_offset,
    )


def normalize_recognition_usage(audio_duration: float) -> SpeechEvent:
    """Create a ``RECOGNITION_USAGE`` event for billing/metrics tracking.

    Call this at the end of a turn or session with the total audio duration
    that was processed.

    Args:
        audio_duration: Total audio processed in seconds.
    """
    return SpeechEvent(
        type=SpeechEventType.RECOGNITION_USAGE,
        recognition_usage=RecognitionUsage(audio_duration=audio_duration),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_speech_event(
    data: dict[str, Any],
    *,
    is_final: bool,
    language: str,
    start_time_offset: float,
) -> SpeechEvent:
    event_type = (
        SpeechEventType.FINAL_TRANSCRIPT
        if is_final
        else SpeechEventType.INTERIM_TRANSCRIPT
    )

    metadata = data.get("metadata", {})
    results: list[dict[str, Any]] = data.get("results", [])

    # Build full transcript text from word results.
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

    # Speechmatics does not return a per-segment confidence; use 1.0 for
    # final results and 0.0 for partials (following the LiveKit plugin
    # convention).
    confidence = 1.0 if is_final else 0.0

    speech_data = SpeechData(
        language=data.get("language", language),
        text=text,
        start_time=metadata.get("start_time", 0.0) + start_time_offset,
        end_time=metadata.get("end_time", 0.0) + start_time_offset,
        confidence=confidence,
        speaker_id=data.get("speaker", None),
        is_final=is_final,
        words=words if words else None,
    )

    return SpeechEvent(
        type=event_type,
        alternatives=[speech_data],
    )
