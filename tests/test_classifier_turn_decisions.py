#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classification outcomes, provisional acknowledgements, and generated response context."""

from unittest.mock import AsyncMock

import pytest

from pipecat.classifiers.base_classifier import ChoiceQuestion, ChoiceResult
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserBackchannelFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_buffer import LLMResponseBuffer
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_start import ClassifierUserTurnStartStrategy, UserTurnStartDecision
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def verdict(choice):
    return {"turn": ChoiceResult(choice=choice, confidence=1, probabilities={choice: 1})}


def make_pipeline(classifier):
    context = LLMContext()
    start = ClassifierUserTurnStartStrategy(classifier=classifier, context=dict, cooldown=0)
    user, assistant = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[start], stop=[ExternalUserTurnStopStrategy()]
            )
        ),
    )
    return context, user, start, Pipeline([user, assistant])


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["backchannel", "continue", "interrupt"])
async def test_choice_controls_context_and_backchannel_notification(choice):
    classifier = AsyncMock()
    classifier.choice.return_value = verdict(choice)
    context, user, start, pipeline = make_pipeline(classifier)
    messages, classifications = [], []
    user.add_event_handler("on_user_backchannel", lambda _, message: messages.append(message))
    start.add_event_handler(
        "on_classification", lambda _, text, result: classifications.append((text, result.choice))
    )
    down, _ = await run_test(
        pipeline,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            ProposedUserStartedSpeakingFrame(),
            InterimTranscriptionFrame("はい", "caller", "now"),
            SleepFrame(sleep=0.03),
            TranscriptionFrame("はい", "caller", "now"),
            ProposedUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),
            ProposedUserStoppedSpeakingFrame(),
        ],
    )
    assert classifications == [("はい", choice)]
    assert context.get_messages() == (
        [{"role": "user", "content": "はい"}] if choice == "interrupt" else []
    )
    frames = [f for f in down if isinstance(f, UserBackchannelFrame)]
    assert len(frames) == len(messages) == (1 if choice == "backchannel" else 0)
    if frames:
        assert (frames[0].text, frames[0].user_id, frames[0].timestamp) == ("はい", "caller", "now")


@pytest.mark.asyncio
async def test_interim_backchannel_can_become_correction_before_endpoint():
    classifier = AsyncMock()
    classifier.choice.side_effect = [verdict("backchannel"), verdict("interrupt")]
    context, _, _, pipeline = make_pipeline(classifier)
    down, _ = await run_test(
        pipeline,
        frames_to_send=[
            BotStartedSpeakingFrame(),
            ProposedUserStartedSpeakingFrame(),
            InterimTranscriptionFrame("はい", "caller", "now"),
            SleepFrame(sleep=0.03),
            TranscriptionFrame("はい、でも日付が違います", "caller", "now"),
            ProposedUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),
        ],
    )
    assert not any(isinstance(f, UserBackchannelFrame) for f in down)
    assert context.get_messages() == [{"role": "user", "content": "はい、でも日付が違います"}]
    assert [call.args[0]["user_speech"] for call in classifier.choice.call_args_list] == [
        "はい",
        "はい、でも日付が違います",
    ]


@pytest.mark.asyncio
async def test_generated_response_is_available_before_playback_and_cleared_on_interruption():
    buffer = LLMResponseBuffer()
    await buffer.setup(frame_processor_setup(TaskManager()))
    buffer.push_frame = AsyncMock()
    classifier = AsyncMock()
    classifier.choice.return_value = verdict("continue")
    state = {"conversation": [{"role": "user", "content": "いつ届きますか？"}]}
    start = ClassifierUserTurnStartStrategy(
        classifier=classifier,
        context=lambda: {**state, "current_assistant_response": buffer.text},
    )
    try:
        for frame in [
            LLMFullResponseStartFrame(),
            LLMTextFrame("明日の"),
            LLMTextFrame("午後3時に届きます。"),
            LLMFullResponseEndFrame(),
        ]:
            await buffer.process_frame(frame, FrameDirection.DOWNSTREAM)
        assert await start.decide("いつ？") == UserTurnStartDecision.CONTINUE
        sent = classifier.choice.call_args.args[0]
        assert sent["current_assistant_response"] == "明日の午後3時に届きます。"
        assert sent["user_speech"] == "いつ？"
        assert "user_speech" not in state
        await buffer.process_frame(InterruptionFrame(), FrameDirection.UPSTREAM)
        assert buffer.text == ""
        await buffer.process_frame(TTSSpeakFrame("確認いたします。"), FrameDirection.DOWNSTREAM)
        assert buffer.text == "確認いたします。"
        await buffer.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        assert buffer.text == ""
        assert buffer.push_frame.await_count == 7
    finally:
        await buffer.cleanup()


def test_custom_question_preserves_outcome_names():
    with pytest.raises(ValueError, match="backchannel, continue, and interrupt"):
        ClassifierUserTurnStartStrategy(
            classifier=AsyncMock(),
            context=dict,
            question=ChoiceQuestion(instructions="Choose", options={"yes": "Yes"}),
        )
