#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Continued final transcripts supersede inference awaiting a turn verdict."""

import asyncio
import copy

import pytest

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import QueuedFrameProcessor, SleepFrame, run_test
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop import (
    LLMTurnCompletionUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.deferred_user_turn_stop_strategy import deferred
from pipecat.turns.user_turn_strategies import UserTurnStrategies


class DelayedVerdictLLM(LLMService):
    """Use the real marker protocol, with a delayed first provider response."""

    def __init__(self, *, continuation_delay=0.01):
        super().__init__(
            settings=LLMSettings(
                model="test",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
            )
        )
        self.contexts = []
        self.cancelled = []
        self.answers = []
        self.continuation_delay = continuation_delay

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        attempt = len(self.contexts)
        self.contexts.append(copy.deepcopy(frame.context.get_messages()))
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            await asyncio.sleep(0.2 if attempt == 0 else self.continuation_delay)
            answer = " | ".join(m["content"] for m in self.contexts[-1] if m["role"] == "user")
            await self._push_llm_text("✓ " + answer)
            self.answers.append(answer)
        except asyncio.CancelledError:
            self.cancelled.append(attempt)
            raise
        finally:
            await self.push_frame(LLMFullResponseEndFrame())


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_timeout,continuation_after", [(0.01, 0.08), (0.1, 0.25)])
@pytest.mark.parametrize(
    "continuation", [None, "　", "では、お願いします。", "違います。3本です。"]
)
async def test_only_latest_unfinalized_user_input_produces_an_answer(
    continuation, endpoint_timeout, continuation_after
):
    context = LLMContext([])
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy()],
                stop=[
                    deferred(
                        SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=endpoint_timeout)
                    ),
                    LLMTurnCompletionUserTurnStopStrategy(),
                ],
            )
        ),
    )
    llm = DelayedVerdictLLM()
    stopped = []

    @pair.user().event_handler("on_user_turn_stopped")
    async def on_stopped(_aggregator, _strategy, message):
        stopped.append(message.content)

    frames = [
        TranscriptionFrame("はい。", "caller", "now"),
        SleepFrame(sleep=continuation_after),
    ]
    if continuation:
        frames.append(TranscriptionFrame(continuation, "caller", "now"))
    frames.append(SleepFrame(sleep=0.4))
    output = asyncio.Queue()
    capture = QueuedFrameProcessor(queue=output, queue_direction=FrameDirection.DOWNSTREAM)
    await run_test(Pipeline([pair.user(), llm, capture, pair.assistant()]), frames_to_send=frames)
    downstream = [output.get_nowait() for _ in range(output.qsize())]

    assert len(llm.answers) == 1
    has_continuation = continuation is not None and bool(continuation.strip())
    assert llm.cancelled == ([0] if has_continuation else [])
    expected_user = ["はい。"] + ([continuation] if has_continuation else [])
    assert [m["content"] for m in llm.contexts[-1] if m["role"] == "user"] == expected_user
    assert stopped == [" ".join(expected_user)]

    assert [f.text for f in downstream if isinstance(f, LLMTextFrame)] == [
        " | ".join(expected_user)
    ]
