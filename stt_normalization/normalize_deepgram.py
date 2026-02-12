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
           "from_finalize": false,
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

Finalization
~~~~~~~~~~~~
Following the Pipecat two-phase protocol, Deepgram's ``from_finalize``
field in the live response confirms that the server acknowledged a
``connection.finalize()`` call.  When a live result has both
``is_final=true`` and ``from_finalize=true``, the normalizer confirms
finalization and the resulting event will carry ``finalized=True``.

Reference:
    https://developers.deepgram.com/docs/results
    https://developers.deepgram.com/docs/finalize
    https://github.com/pipecat-ai/pipecat  (services/deepgram/stt.py)
    https://github.com/livekit/agents       (livekit-plugins-deepgram)
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
# Stateless helpers — one-shot conversion of a single message
# ---------------------------------------------------------------------------


def normalize_live_transcript(
    data: dict[str, Any],
    *,
    language: str = "en",
    start_time_offset: float = 0.0,
    finalized: bool = False,
) -> SpeechEvent:
    """Convert a Deepgram live (streaming) ``Results`` message.

    Args:
        data: Raw JSON dict received from the Deepgram WebSocket.
        language: Fallback language code when not detected by Deepgram.
        start_time_offset: Offset (seconds) added to all timestamps so they
            are relative to the session start.
        finalized: Pass ``True`` when the two-phase finalization protocol
            has been confirmed (see :class:`DeepgramNormalizer`).

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

    # Deepgram may report a detected language at the channel level or in
    # the first alternative's ``languages`` list.
    detected_lang: str | None = channel.get("detected_language")
    if not detected_lang and alternatives:
        langs = alternatives[0].get("languages", [])
        if langs:
            detected_lang = langs[0]
    detected_lang = detected_lang or language

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
                finalized=finalized and is_final,
                result=data,
                words=words if words else None,
            )
        )

    request_id = data.get("metadata", {}).get("request_id", "")

    return SpeechEvent(
        type=event_type,
        request_id=request_id,
        alternatives=speech_alternatives,
    )


def normalize_prerecorded_transcript(
    data: dict[str, Any],
    *,
    language: str | None = None,
) -> SpeechEvent:
    """Convert a Deepgram pre-recorded (batch) transcription response.

    Pre-recorded results are always final and finalized (the entire audio
    has been processed).

    Args:
        data: Full JSON response from the Deepgram ``/v1/listen`` REST
              endpoint.
        language: Fallback language if auto-detection is not used.

    Returns:
        A :class:`SpeechEvent` of type ``FINAL_TRANSCRIPT`` with
        ``finalized=True``.
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
                    finalized=True,
                    result=data,
                    words=words if words else None,
                )
            )

    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        request_id=request_id,
        alternatives=speech_alternatives,
    )


# ---------------------------------------------------------------------------
# Stateful normalizer — manages the two-phase finalization protocol
# ---------------------------------------------------------------------------


class DeepgramNormalizer:
    """Stateful normalizer that tracks the Pipecat two-phase finalization.

    Usage::

        norm = DeepgramNormalizer(language="en")

        # On each WebSocket message:
        event = norm.on_live_result(data)

        # When VAD detects user stopped speaking:
        norm.request_finalize()
        await connection.finalize()   # tell Deepgram to flush

    The ``finalized`` flag on the resulting :class:`SpeechData` will be set
    automatically when the sequence ``request_finalize()`` →
    ``from_finalize=True`` response is observed.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        start_time_offset: float = 0.0,
    ) -> None:
        self._language = language
        self._start_time_offset = start_time_offset
        self._finalize_requested = False
        self._finalize_pending = False

    # -- Finalization state machine (mirrors Pipecat STTService) -----------

    def request_finalize(self) -> None:
        """Phase 1: caller asks to finalize (e.g. VAD user-stopped-speaking).

        After calling this, send ``connection.finalize()`` to the Deepgram
        WebSocket so the server flushes its buffer.
        """
        self._finalize_requested = True

    def confirm_finalize(self) -> None:
        """Phase 2: provider confirmed the finalize request."""
        if self._finalize_requested:
            self._finalize_pending = True
            self._finalize_requested = False

    # -- Message handler ---------------------------------------------------

    def on_live_result(self, data: dict[str, Any]) -> SpeechEvent:
        """Handle a live ``Results`` message from the Deepgram WebSocket.

        Automatically detects the ``from_finalize`` flag to confirm
        finalization.
        """
        is_final: bool = data.get("is_final", False)
        from_finalize: bool = data.get("from_finalize", False)

        if is_final and from_finalize:
            self.confirm_finalize()

        finalized = self._finalize_pending
        event = normalize_live_transcript(
            data,
            language=self._language,
            start_time_offset=self._start_time_offset,
            finalized=finalized,
        )

        if finalized:
            self._finalize_pending = False

        return event
