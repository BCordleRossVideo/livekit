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
track word-level stability across consecutive partials:

- ``is_final=false`` → **PARTIAL** (or **STABLE** when every word has
  survived 3+ consecutive partials in its position)
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
    stable_flags: list[bool] | None = None,
    language: str = "en",
    start_time_offset: float = 0.0,
) -> SpeechEvent:
    """Convert a Deepgram live (streaming) ``Results`` message.

    Args:
        data: Raw JSON dict received from the Deepgram WebSocket.
        transcript_type: Whether this is PARTIAL, STABLE, or FINAL.
        stable_flags: Per-word stability flags from the detector.
            Applied to the first alternative's words.  If ``None``,
            all words inherit from *transcript_type*.
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
        stable_flags=stable_flags,
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

    Pre-recorded results are always FINAL (all words stable).

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
            stable_flags=None,
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
# Stateful normalizer — tracks word-level stability across partials
# ---------------------------------------------------------------------------


class DeepgramNormalizer:
    """Stateful normalizer with word-level PARTIAL → STABLE promotion.

    Usage::

        norm = DeepgramNormalizer(language="en")

        for msg in websocket_messages:
            event = norm.on_live_result(msg)
            alt = event.alternatives[0]
            print(f"{alt.transcript_type}: {alt.stable_text!r} | {alt.text!r}")
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

        Tracks word-level stability and returns FINAL when Deepgram's
        ``is_final`` flag is set.
        """
        provider_is_final: bool = data.get("is_final", False)
        word_texts = _extract_words(data)
        transcript_type, stable_flags = self._detector.on_words(
            word_texts, final=provider_is_final
        )

        return normalize_live_transcript(
            data,
            transcript_type=transcript_type,
            stable_flags=stable_flags,
            language=self._language,
            start_time_offset=self._start_time_offset,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_words(data: dict[str, Any]) -> list[str]:
    """Pull word strings from the first alternative of a Deepgram result."""
    channel: dict[str, Any] = data.get("channel", {})
    alts: list[dict[str, Any]] = channel.get("alternatives", [])
    if not alts:
        return []
    return [
        w.get("word", w.get("punctuated_word", ""))
        for w in alts[0].get("words", [])
    ]


def _build_alternatives(
    alternatives: list[dict[str, Any]],
    *,
    detected_lang: str,
    transcript_type: TranscriptType,
    stable_flags: list[bool] | None,
    start_time_offset: float,
    raw_result: Any,
    fallback_start: float,
    fallback_duration: float,
) -> list[SpeechData]:
    """Build a list of :class:`SpeechData` from Deepgram alternatives."""
    speech_alternatives: list[SpeechData] = []

    for alt_idx, alt in enumerate(alternatives):
        words_raw: list[dict[str, Any]] = alt.get("words", [])
        words: list[TimedWord] = []
        for i, w in enumerate(words_raw):
            word_text = w.get("word", w.get("punctuated_word", ""))

            # Apply per-word stability (only for the first alternative,
            # since the detector tracks a single word sequence).
            if stable_flags is not None and alt_idx == 0 and i < len(stable_flags):
                is_stable = stable_flags[i]
            else:
                is_stable = transcript_type in (
                    TranscriptType.STABLE,
                    TranscriptType.FINAL,
                )

            words.append(
                TimedWord(
                    text=word_text,
                    start_time=w.get("start", 0.0) + start_time_offset,
                    end_time=w.get("end", 0.0) + start_time_offset,
                    is_stable=is_stable,
                )
            )

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

        # Build stable_text from the leading run of stable words.
        stable_parts: list[str] = []
        for w in words:
            if w.is_stable:
                stable_parts.append(w.text)
            else:
                break
        stable_text = " ".join(stable_parts)

        speech_alternatives.append(
            SpeechData(
                language=detected_lang,
                text=alt.get("transcript", ""),
                transcript_type=transcript_type,
                stable_text=stable_text,
                start_time=seg_start,
                end_time=seg_end,
                confidence=alt.get("confidence", 0.0),
                speaker_id=speaker_id,
                result=raw_result,
                words=words if words else None,
            )
        )

    return speech_alternatives
