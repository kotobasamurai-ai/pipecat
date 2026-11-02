#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observable admission behavior across speech signals and session shutdown."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from pipecat.classifiers.base_classifier import ChoiceResult
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.user_start import ClassifierUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


@asynccontextmanager
async def aggregator(classifier, *, timeout=1):
    context = LLMContext()
    user, _ = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[
                    ClassifierUserTurnStartStrategy(
                        classifier=classifier, context=dict, cooldown=0, timeout=timeout
                    )
                ],
                stop=[ExternalUserTurnStopStrategy()],
            )
        ),
    )
    await user.setup(frame_processor_setup(TaskManager()))
    user.push_frame = AsyncMock()
    user.broadcast_frame = AsyncMock()
    user.broadcast_interruption = AsyncMock()
    try:
        yield user, context
    finally:
        await user.cleanup()


async def send(user, *frames):
    for frame in frames:
        await user.process_frame(frame, FrameDirection.DOWNSTREAM)


def interrupt():
    return {"turn": ChoiceResult(choice="interrupt", confidence=1, probabilities={"interrupt": 1})}


@pytest.mark.asyncio
@pytest.mark.parametrize("native_first", [True, False])
async def test_inference_waits_for_both_native_and_vad_endpoints(native_first):
    classifier = AsyncMock()
    requested, release, started, stopped = (asyncio.Event() for _ in range(4))

    async def decide(*args):
        requested.set()
        await release.wait()
        return interrupt()

    classifier.choice.side_effect = decide
    async with aggregator(classifier) as (user, context):
        user.add_event_handler("on_user_turn_started", lambda *args: started.set())
        user.add_event_handler("on_user_turn_stopped", lambda *args: stopped.set())
        await send(
            user,
            BotStartedSpeakingFrame(),
            ProposedUserStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame("Wait.", "user", "now"),
        )
        await asyncio.wait_for(requested.wait(), 1)
        endpoints = [ProposedUserStoppedSpeakingFrame(), VADUserStoppedSpeakingFrame()]
        if not native_first:
            endpoints.reverse()
        await send(user, endpoints[0])
        release.set()
        await asyncio.wait_for(started.wait(), 1)
        assert context.get_messages() == []
        assert not stopped.is_set()
        await send(user, endpoints[1])
        await asyncio.wait_for(stopped.wait(), 1)
        assert context.get_messages() == [{"role": "user", "content": "Wait."}]
        emitted = [c.args[0] for c in user.push_frame.await_args_list]
        assert sum(isinstance(frame, LLMContextFrame) for frame in emitted) == 1


@pytest.mark.asyncio
async def test_transcript_updates_do_not_extend_original_classifier_timeout():
    classifier = AsyncMock()
    requested, cancelled, started, stopped = (asyncio.Event() for _ in range(4))

    async def decide(*args):
        requested.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    classifier.choice.side_effect = decide
    async with aggregator(classifier, timeout=0.08) as (user, context):
        user.add_event_handler("on_user_turn_started", lambda *args: started.set())
        user.add_event_handler("on_user_turn_stopped", lambda *args: stopped.set())
        await send(
            user,
            BotStartedSpeakingFrame(),
            ProposedUserStartedSpeakingFrame(),
            InterimTranscriptionFrame("Wait", "user", "now"),
        )
        await asyncio.wait_for(requested.wait(), 1)
        updates = []

        async def update_transcript():
            while not started.is_set():
                text = f"Wait, correction {len(updates) + 1}"
                updates.append(text)
                await send(user, InterimTranscriptionFrame(text, "user", "now"))
                await asyncio.sleep(0.01)

        updater = user.create_task(update_transcript(), "test-transcript-updates")
        try:
            await asyncio.wait_for(started.wait(), 0.4)
        finally:
            updater.cancel()
            await asyncio.gather(updater, return_exceptions=True)
        assert len(updates) >= 2
        assert cancelled.is_set()
        classifier.choice.assert_awaited_once()
        assert classifier.choice.call_args.args[0]["user_speech"] == "Wait"
        assert context.get_messages() == []
        await send(
            user,
            TranscriptionFrame(updates[-1], "user", "now"),
            ProposedUserStoppedSpeakingFrame(),
        )
        await asyncio.wait_for(stopped.wait(), 1)
        assert context.get_messages() == [{"role": "user", "content": updates[-1]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [CancelFrame, EndFrame])
async def test_shutdown_discards_pending_aggregation_and_cancels_classification(ending):
    classifier = AsyncMock()
    requested, cancelled = asyncio.Event(), asyncio.Event()
    events = []

    async def decide(*args):
        requested.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return interrupt()
        finally:
            cancelled.set()

    classifier.choice.side_effect = decide
    async with aggregator(classifier) as (user, context):
        user.add_event_handler("on_user_turn_started", lambda *args: events.append("started"))
        user.add_event_handler("on_user_turn_stopped", lambda *args: events.append("stopped"))
        await send(
            user,
            BotStartedSpeakingFrame(),
            ProposedUserStartedSpeakingFrame(),
            TranscriptionFrame("Wait.", "user", "now"),
            ProposedUserStoppedSpeakingFrame(),
        )
        await asyncio.wait_for(requested.wait(), 1)
        await asyncio.wait_for(send(user, ending()), 1)
        await asyncio.wait_for(cancelled.wait(), 1)
        assert context.get_messages() == []
        assert events == []
        user.broadcast_interruption.assert_not_called()
        assert not any(
            isinstance(call.args[0], LLMContextFrame) for call in user.push_frame.await_args_list
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("supersede_with_bot_stop", [True, False])
async def test_cancelled_classifier_rejection_cannot_discard_admitted_speech(
    supersede_with_bot_stop,
):
    classifier = AsyncMock()
    requested, cancelled, stopped = (asyncio.Event() for _ in range(3))
    events = []

    async def decide(*args):
        requested.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            return {
                "turn": ChoiceResult(choice="continue", confidence=1, probabilities={"continue": 1})
            }

    classifier.choice.side_effect = decide
    async with aggregator(classifier, timeout=1 if supersede_with_bot_stop else 0.03) as (
        user,
        context,
    ):
        user.add_event_handler("on_user_turn_started", lambda *args: events.append("started"))

        def record_stop(*args):
            events.append("stopped")
            stopped.set()

        user.add_event_handler("on_user_turn_stopped", record_stop)
        await send(
            user,
            BotStartedSpeakingFrame(),
            ProposedUserStartedSpeakingFrame(),
            TranscriptionFrame("Wait.", "user", "now"),
            ProposedUserStoppedSpeakingFrame(),
        )
        await asyncio.wait_for(requested.wait(), 1)
        if supersede_with_bot_stop:
            await send(user, BotStoppedSpeakingFrame())
        await asyncio.wait_for(cancelled.wait(), 1)
        await asyncio.wait_for(stopped.wait(), 1)
        assert events == ["started", "stopped"]
        assert context.get_messages() == [{"role": "user", "content": "Wait."}]
        classifier.choice.assert_awaited_once()
        assert (
            sum(
                isinstance(call.args[0], LLMContextFrame)
                for call in user.push_frame.await_args_list
            )
            == 1
        )
