"""Normalize Deepgram STT responses into unified SpeechEvent types.

Deepgram produces two response shapes depending on the API used:

1. **Live/streaming** (WebSocket) – ``Results`` messages with this structure::

       {
           "channel": {
               "alternatives": [
                   {
                       "transcript": "hello world",
                       "confidence": 0.98,
                       "words": [
                           {"word": "hello", "start": 0.5, "end": 0.8,
                            "confidence": 0.99, "speaker": 0},
                           ...
                       ]
                   }
               ]
           },
           "is_final": true,
           "speech_final": true,
           "start": 0.5,
           "duration": 1.2,
           "metadata": {"request_id": "..."}
       }

2. **Pre-recorded** (REST) – the response wraps channels inside a
   ``results`` key::

       {
           "results": {
               "channels": [
                   {
                       "alternatives": [
                           {
                               "transcript": "hello world",
                               "confidence": 0.98,
                               "words": [...]
                           }
                       ],
                       "detected_language": "en"
                   }
               ]
           },
           "metadata": {"request_id": "..."}
       }

This module converts both shapes into the normalized :class:`SpeechEvent` /
:class:`SpeechData` types defined in ``types.py``.

Reference:
    https://developers.deepgram.com/docs/results
    https://github.com/livekit/agents (livekit-plugins-deepgram)
"""

from __future__ import annotations

from typing import Any

from .types import (
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    TimedWord,
)


# ---------------------------------------------------------------------------
# Live / streaming results
# ---------------------------------------------------------------------------


def normalize_live_transcript(
    data: dict[str, Any],
    *,
    language: str = "en",
    start_time_offset: float = 0.0,
) -> SpeechEvent:
    """Convert a Deepgram live (streaming) ``Results`` message.

    Args:
        data: Raw JSON dict received from the Deepgram WebSocket.
        language: Fallback language code when not detected by Deepgram.
        start_time_offset: Offset (seconds) added to all timestamps so they
            are relative to the session start.

    Returns:
        A :class:`SpeechEvent` of type ``INTERIM_TRANSCRIPT`` or
        ``FINAL_TRANSCRIPT`` depending on ``data["is_final"]``.
    """
    is_final: bool = data.get("is_final", False)
    event_type = (
        SpeechEventType.FINAL_TRANSCRIPT
        if is_final
        else SpeechEventType.INTERIM_TRANSCRIPT
    )

    channel: dict[str, Any] = data.get("channel", {})
    alternatives: list[dict[str, Any]] = channel.get("alternatives", [])

    # Deepgram may report a detected language at the channel level.
    detected_lang = channel.get("detected_language") or language

    speech_alternatives: list[SpeechData] = []
    for alt in alternatives:
        words_raw: list[dict[str, Any]] = alt.get("words", [])
        words = [
            TimedWord(
                text=w.get("word", w.get("punctuated_word", "")),
                start_time=w.get("start", 0.0) + start_time_offset,
                end_time=w.get("end", 0.0) + start_time_offset,
            )
            for w in words_raw
        ]

        # Speaker ID is only reliable on final results.
        speaker_id: str | None = None
        if is_final and words_raw:
            spk = words_raw[0].get("speaker")
            if spk is not None:
                speaker_id = f"S{spk}"

        transcript_text: str = alt.get("transcript", "")

        # Compute segment start/end from the word-level times when available,
        # falling back to the top-level ``start`` / ``duration``.
        if words:
            seg_start = words[0].start_time
            seg_end = words[-1].end_time
        else:
            seg_start = data.get("start", 0.0) + start_time_offset
            seg_end = seg_start + data.get("duration", 0.0)

        speech_alternatives.append(
            SpeechData(
                language=detected_lang,
                text=transcript_text,
                start_time=seg_start,
                end_time=seg_end,
                confidence=alt.get("confidence", 0.0),
                speaker_id=speaker_id,
                is_final=is_final,
                words=words if words else None,
            )
        )

    request_id = data.get("metadata", {}).get("request_id", "")

    return SpeechEvent(
        type=event_type,
        request_id=request_id,
        alternatives=speech_alternatives,
    )


# ---------------------------------------------------------------------------
# Pre-recorded (REST) results
# ---------------------------------------------------------------------------


def normalize_prerecorded_transcript(
    data: dict[str, Any],
    *,
    language: str | None = None,
) -> SpeechEvent:
    """Convert a Deepgram pre-recorded (batch) transcription response.

    Args:
        data: Full JSON response from the Deepgram
              ``/v1/listen`` REST endpoint.
        language: Fallback language if auto-detection is not used.

    Returns:
        A :class:`SpeechEvent` of type ``FINAL_TRANSCRIPT``.
    """
    results: dict[str, Any] = data.get("results", {})
    channels: list[dict[str, Any]] = results.get("channels", [])
    request_id = data.get("metadata", {}).get("request_id", "")

    speech_alternatives: list[SpeechData] = []

    if channels:
        channel = channels[0]
        detected_lang = channel.get("detected_language") or language or "en"

        for alt in channel.get("alternatives", []):
            words_raw: list[dict[str, Any]] = alt.get("words", [])
            words = [
                TimedWord(
                    text=w.get("word", w.get("punctuated_word", "")),
                    start_time=w.get("start", 0.0),
                    end_time=w.get("end", 0.0),
                )
                for w in words_raw
            ]

            speaker_id: str | None = None
            if words_raw:
                spk = words_raw[0].get("speaker")
                if spk is not None:
                    speaker_id = f"S{spk}"

            seg_start = words[0].start_time if words else 0.0
            seg_end = words[-1].end_time if words else 0.0

            speech_alternatives.append(
                SpeechData(
                    language=detected_lang,
                    text=alt.get("transcript", ""),
                    start_time=seg_start,
                    end_time=seg_end,
                    confidence=alt.get("confidence", 0.0),
                    speaker_id=speaker_id,
                    is_final=True,
                    words=words if words else None,
                )
            )

    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        request_id=request_id,
        alternatives=speech_alternatives,
    )
