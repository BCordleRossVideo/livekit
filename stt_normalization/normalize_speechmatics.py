"""Normalize Speechmatics STT responses into unified SpeechEvent types.

Speechmatics produces two kinds of transcript messages over its WebSocket:

1. **AddPartialTranscript** (``ADD_PARTIAL_SEGMENT``) – interim/partial
   results while the user is still speaking.  The payload looks like::

       {
           "metadata": {"start_time": 1.23, "end_time": 2.45},
           "results": [
               {"type": "word", "content": "hello", "start_time": 1.23, "end_time": 1.50},
               ...
           ]
       }

2. **AddTranscript** (``ADD_SEGMENT``) – final results once a segment is
   confirmed.  Same shape as above but considered immutable.

Speechmatics also emits turn-level events (``START_OF_TURN`` /
``END_OF_TURN``) and speaker diarization data.  When diarization is enabled
the segment dicts include a ``speaker_id`` field (e.g. ``"S1"``).

Finalization
~~~~~~~~~~~~
Following the Pipecat two-phase protocol, a Speechmatics segment with
``is_eou: true`` (end-of-utterance) confirms finalization — but only
when finalization was previously *requested* by the caller (typically in
response to a VAD user-stopped-speaking event).  See
``SpeechmaticsNormalizer.confirm_finalize_if_eou()`` and the state
helpers ``request_finalize()`` / ``confirm_finalize()``.

Reference:
    https://docs.speechmatics.com/introduction/rt-api-ref
    https://github.com/pipecat-ai/pipecat  (services/speechmatics/stt.py)
    https://github.com/livekit/agents       (livekit-plugins-speechmatics)
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


# ---------------------------------------------------------------------------
# Stateless helpers — one-shot conversion of a single message
# ---------------------------------------------------------------------------


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
        finalized=False,
        language=language,
        start_time_offset=start_time_offset,
    )


def normalize_final_transcript(
    data: dict[str, Any],
    *,
    language: str = "en",
    start_time_offset: float = 0.0,
    finalized: bool = False,
) -> SpeechEvent:
    """Convert a Speechmatics ``AddTranscript`` message.

    Args:
        data: Raw JSON dict from the Speechmatics WebSocket.
        language: Fallback language code if not present in the payload.
        start_time_offset: Offset (seconds) to add to all timestamps.
        finalized: Pass ``True`` when the two-phase finalization protocol
            has been confirmed (see :class:`SpeechmaticsNormalizer`).

    Returns:
        A :class:`SpeechEvent` with type ``FINAL_TRANSCRIPT``.
    """
    return _to_speech_event(
        data,
        is_final=True,
        finalized=finalized,
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
# Stateful normalizer — manages the two-phase finalization protocol
# ---------------------------------------------------------------------------


class SpeechmaticsNormalizer:
    """Stateful normalizer that tracks the Pipecat two-phase finalization.

    Usage::

        norm = SpeechmaticsNormalizer(language="en")

        # On each WebSocket message:
        if msg_type == "AddPartialTranscript":
            event = norm.on_partial(data)
        elif msg_type == "AddTranscript":
            event = norm.on_final(data)

        # When VAD detects user stopped speaking:
        norm.request_finalize()

    The ``finalized`` flag on the resulting :class:`SpeechData` will be set
    automatically when the sequence ``request_finalize()`` →
    ``is_eou: true`` segment → next final frame is observed.
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
        """Phase 1: caller asks to finalize (e.g. VAD user-stopped-speaking)."""
        self._finalize_requested = True

    def confirm_finalize(self) -> None:
        """Phase 2: provider confirmed end-of-utterance."""
        if self._finalize_requested:
            self._finalize_pending = True
            self._finalize_requested = False

    def confirm_finalize_if_eou(self, data: dict[str, Any]) -> None:
        """Convenience: call ``confirm_finalize()`` when any result has ``is_eou``."""
        results: list[dict[str, Any]] = data.get("results", [])
        if any(r.get("is_eou", False) for r in results):
            self.confirm_finalize()

    # -- Message handlers --------------------------------------------------

    def on_partial(self, data: dict[str, Any]) -> SpeechEvent:
        """Handle ``AddPartialTranscript``."""
        return normalize_partial_transcript(
            data,
            language=self._language,
            start_time_offset=self._start_time_offset,
        )

    def on_final(self, data: dict[str, Any]) -> SpeechEvent:
        """Handle ``AddTranscript``.

        Checks for ``is_eou`` to confirm finalization, then marks the
        resulting event accordingly.
        """
        self.confirm_finalize_if_eou(data)

        finalized = self._finalize_pending
        event = normalize_final_transcript(
            data,
            language=self._language,
            start_time_offset=self._start_time_offset,
            finalized=finalized,
        )

        if finalized:
            self._finalize_pending = False

        return event


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_speech_event(
    data: dict[str, Any],
    *,
    is_final: bool,
    finalized: bool,
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
        speaker_id=data.get("speaker", data.get("speaker_id", None)),
        is_final=is_final,
        finalized=finalized,
        result=data,
        words=words if words else None,
    )

    return SpeechEvent(
        type=event_type,
        alternatives=[speech_data],
    )
