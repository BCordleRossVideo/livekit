"""Normalize Deepgram STT responses into unified SpeechEvent types.

Deepgram produces two response shapes:

1. **Live/streaming** (WebSocket) – ``Results`` messages::

       {
           "channel": {
               "alternatives": [
                   {"transcript": "hello world", "confidence": 0.98,
                    "words": [{"word": "hello", "start": 0.5, "end": 0.8, ...}]}
               ]
           },
           "is_final": true,
           "start": 0.5,
           "duration": 1.2,
           "metadata": {"request_id": "..."}
       }

2. **Pre-recorded** (REST) – channels inside a ``results`` key::

       {
           "results": {
               "channels": [
                   {"alternatives": [...], "detected_language": "en"}
               ]
           },
           "metadata": {"request_id": "..."}
       }

Transcript lifecycle
~~~~~~~~~~~~~~~~~~~~
The :class:`DeepgramNormalizer` uses a :class:`StabilityDetector` to
promote partials to stable after 3 consecutive identical texts:

- ``is_final=false`` → **PARTIAL** (or **STABLE** after 3 repeats)
- ``is_final=true``  → **FINAL**

Pre-recorded results are always **FINAL**.

Reference:
    https://developers.deepgram.com/docs/results
"""

from __future__ import annotations

from typing import Any

from .types import (
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


def normalize_live_transcript(
    data: dict[str, Any],
    *,
    transcript_type: TranscriptType,
    language: str = "en",
    start_time_offset: float = 0.0,
) -> SpeechEvent:
    """Convert a Deepgram live (streaming) ``Results`` message.

    Args:
        data: Raw JSON dict received from the Deepgram WebSocket.
        transcript_type: Whether this is PARTIAL, STABLE, or FINAL.
        language: Fallback language code when not detected by Deepgram.
        start_time_offset: Offset (seconds) added to all timestamps.
    """
    channel: dict[str, Any] = data.get("channel", {})
    alternatives: list[dict[str, Any]] = channel.get("alternatives", [])

    detected_lang: str | None = channel.get("detected_language")
    if not detected_lang and alternatives:
        langs = alternatives[0].get("languages", [])
        if langs:
            detected_lang = langs[0]
    detected_lang = detected_lang or language

    speech_alternatives = _build_alternatives(
        alternatives,
        detected_lang=detected_lang,
        transcript_type=transcript_type,
        start_time_offset=start_time_offset,
        raw_result=data,
        fallback_start=data.get("start", 0.0),
        fallback_duration=data.get("duration", 0.0),
    )

    request_id = data.get("metadata", {}).get("request_id", "")

    return SpeechEvent(
        type=SpeechEventType.TRANSCRIPT,
        request_id=request_id,
        alternatives=speech_alternatives,
    )


def normalize_prerecorded_transcript(
    data: dict[str, Any],
    *,
    language: str | None = None,
) -> SpeechEvent:
    """Convert a Deepgram pre-recorded (batch) transcription response.

    Pre-recorded results are always FINAL.

    Args:
        data: Full JSON response from Deepgram ``/v1/listen``.
        language: Fallback language if auto-detection is not used.
    """
    results: dict[str, Any] = data.get("results", {})
    channels: list[dict[str, Any]] = results.get("channels", [])
    request_id = data.get("metadata", {}).get("request_id", "")

    speech_alternatives: list[SpeechData] = []

    if channels:
        channel = channels[0]
        detected_lang = channel.get("detected_language") or language or "en"

        speech_alternatives = _build_alternatives(
            channel.get("alternatives", []),
            detected_lang=detected_lang,
            transcript_type=TranscriptType.FINAL,
            start_time_offset=0.0,
            raw_result=data,
            fallback_start=0.0,
            fallback_duration=0.0,
        )

    return SpeechEvent(
        type=SpeechEventType.TRANSCRIPT,
        request_id=request_id,
        alternatives=speech_alternatives,
    )


# ---------------------------------------------------------------------------
# Stateful normalizer — tracks stability across partials
# ---------------------------------------------------------------------------


class DeepgramNormalizer:
    """Stateful normalizer with automatic PARTIAL → STABLE promotion.

    Usage::

        norm = DeepgramNormalizer(language="en")

        for msg in websocket_messages:
            event = norm.on_live_result(msg)
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

    def on_live_result(self, data: dict[str, Any]) -> SpeechEvent:
        """Handle a live ``Results`` message from the Deepgram WebSocket.

        Automatically promotes PARTIAL → STABLE after ``stable_threshold``
        consecutive identical texts, and returns FINAL when Deepgram's
        ``is_final`` flag is set.
        """
        provider_is_final: bool = data.get("is_final", False)
        text = _extract_text(data)
        transcript_type = self._detector.on_text(
            text, provider_is_final=provider_is_final
        )

        return normalize_live_transcript(
            data,
            transcript_type=transcript_type,
            language=self._language,
            start_time_offset=self._start_time_offset,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the top transcript string from a Deepgram live result."""
    channel: dict[str, Any] = data.get("channel", {})
    alts: list[dict[str, Any]] = channel.get("alternatives", [])
    if alts:
        return alts[0].get("transcript", "")
    return ""


def _build_alternatives(
    alternatives: list[dict[str, Any]],
    *,
    detected_lang: str,
    transcript_type: TranscriptType,
    start_time_offset: float,
    raw_result: Any,
    fallback_start: float,
    fallback_duration: float,
) -> list[SpeechData]:
    """Build a list of :class:`SpeechData` from Deepgram alternatives."""
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

        speaker_id: str | None = None
        if transcript_type == TranscriptType.FINAL and words_raw:
            spk = words_raw[0].get("speaker")
            if spk is not None:
                speaker_id = f"S{spk}"

        if words:
            seg_start = words[0].start_time
            seg_end = words[-1].end_time
        else:
            seg_start = fallback_start + start_time_offset
            seg_end = seg_start + fallback_duration

        speech_alternatives.append(
            SpeechData(
                language=detected_lang,
                text=alt.get("transcript", ""),
                transcript_type=transcript_type,
                start_time=seg_start,
                end_time=seg_end,
                confidence=alt.get("confidence", 0.0),
                speaker_id=speaker_id,
                result=raw_result,
                words=words if words else None,
            )
        )

    return speech_alternatives
