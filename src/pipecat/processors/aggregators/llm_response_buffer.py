#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Read-only access to the generated text of the latest LLM response."""

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class LLMResponseBuffer(FrameProcessor):
    """Retain generated response text for decisions made during playback.

    Place immediately after the LLM and before TTS. ``text`` includes generated
    tokens even while their audio is queued. It survives generation ending and
    resets on the next response, interruption, or session end. It represents
    the latest generation, rather than a playback position or unheard suffix.
    Direct ``TTSSpeakFrame`` messages replace the current response.
    """

    def __init__(self, **kwargs):
        """Initialize the buffer.

        Args:
            **kwargs: Arguments forwarded to FrameProcessor.
        """
        super().__init__(**kwargs)
        self._text = ""

    @property
    def text(self) -> str:
        """Return the latest response's generated text."""
        return self._text

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Observe response text and forward each frame unchanged."""
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, CancelFrame, EndFrame)):
            self._text = ""
        elif direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._text = ""
            elif isinstance(frame, LLMTextFrame):
                self._text += frame.text
            elif isinstance(frame, TTSSpeakFrame):
                self._text = frame.text
        await self.push_frame(frame, direction)
