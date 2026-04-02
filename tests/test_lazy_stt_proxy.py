#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from typing import AsyncGenerator
from unittest.mock import MagicMock

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.services.lazy_stt_proxy import LazySTTProxy
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import run_test


class MockSTTService(STTService):
    """A minimal STT service that echoes a transcription for each audio frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_count = 0

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        self.audio_count += 1
        yield TranscriptionFrame(text=f"transcript-{self.audio_count}", user_id="", timestamp=0)


class TestLazySTTProxy(unittest.IsolatedAsyncioTestCase):
    async def test_stt_not_created_without_audio(self):
        """STT service should not be created if no audio frames arrive."""
        factory = MagicMock(return_value=MockSTTService())
        proxy = LazySTTProxy(factory=factory)

        await run_test(
            proxy,
            frames_to_send=[],
            expected_down_frames=[],
        )

        factory.assert_not_called()

    async def test_stt_created_on_first_audio(self):
        """STT service should be created when first audio frame arrives."""
        mock_stt = MockSTTService()
        factory = MagicMock(return_value=mock_stt)
        proxy = LazySTTProxy(factory=factory)

        audio = InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)

        down_frames, _ = await run_test(
            proxy,
            frames_to_send=[audio],
            expected_down_frames=[
                STTMetadataFrame,  # metadata from StartFrame processing
                TranscriptionFrame,  # transcription from MockSTTService
                InputAudioRawFrame,  # audio passthrough from STTService
            ],
        )

        factory.assert_called_once()
        assert mock_stt.audio_count == 1

    async def test_factory_called_only_once(self):
        """Factory should be called exactly once, even with multiple audio frames."""
        mock_stt = MockSTTService()
        factory = MagicMock(return_value=mock_stt)
        proxy = LazySTTProxy(factory=factory)

        audio1 = InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
        audio2 = InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)

        await run_test(
            proxy,
            frames_to_send=[audio1, audio2],
            expected_down_frames=[
                STTMetadataFrame,
                TranscriptionFrame,
                InputAudioRawFrame,
                TranscriptionFrame,
                InputAudioRawFrame,
            ],
        )

        factory.assert_called_once()
        assert mock_stt.audio_count == 2

    async def test_non_audio_frames_passthrough_when_idle(self):
        """Non-audio frames should pass through when STT is not yet created."""
        factory = MagicMock()
        proxy = LazySTTProxy(factory=factory)

        # Use a generic frame that is not AudioRawFrame
        frame = TranscriptionFrame(text="hello", user_id="", timestamp=0)

        await run_test(
            proxy,
            frames_to_send=[frame],
            expected_down_frames=[TranscriptionFrame],
        )

        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
