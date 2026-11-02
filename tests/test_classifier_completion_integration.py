#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Completion attribution through the OpenAI stream and user aggregation paths."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openai.types.chat import ChatCompletionChunk

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.turns.user_start.base_async_user_turn_start_strategy import (
    BaseAsyncUserTurnStartStrategy,
)
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class _AcceptSpeech(BaseAsyncUserTurnStartStrategy):
    async def decide(self, text):
        return True


class _ControlledOpenAI(OpenAILLMService):
    """Replace only the provider stream with explicitly released SDK chunks."""

    def __init__(self):
        super().__init__(api_key="unused")
        self.requests = asyncio.Queue()

    async def get_chat_completions(self, context):
        output = asyncio.get_running_loop().create_future()
        await self.requests.put(output)

        async def stream():
            result = await output
            delta = {"content": result} if isinstance(result, str) else result
            yield ChatCompletionChunk(
                id="controlled",
                created=0,
                model="gpt-4.1",
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": delta}],
            )

        return stream()


class TestClassifierCompletionIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = _ControlledOpenAI()
        self.user, self.assistant = LLMContextAggregatorPair(
            LLMContext(),
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=FilterIncompleteUserTurnStrategies(
                    start=[_AcceptSpeech(cooldown=0)],
                    stop=[ExternalUserTurnStopStrategy()],
                )
            ),
        )
        setup = frame_processor_setup(TaskManager(), pipeline_worker=Mock())
        await self.user.setup(setup)
        await self.service.setup(setup)
        await self.assistant.setup(setup)
        self.user.push_frame = AsyncMock()
        self.user.broadcast_frame = AsyncMock()
        self.user.broadcast_interruption = AsyncMock()
        self.stops = []
        self.completions = []
        self.tasks = []
        self.followups = asyncio.Queue()
        self.texts = []
        self.route_tools = False

        async def stopped(*args):
            self.stops.append(True)

        self.user.add_event_handler("on_user_turn_stopped", stopped)

        async def sink(processor, frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, UserTurnInferenceCompletedFrame):
                if direction == FrameDirection.UPSTREAM:
                    self.completions.append(frame)
                    await self.user.process_frame(frame, direction)

            if processor is self.service and direction == FrameDirection.DOWNSTREAM:
                if isinstance(frame, LLMTextFrame):
                    self.texts.append(frame.text)
                if self.route_tools and isinstance(
                    frame,
                    (
                        FunctionCallsStartedFrame,
                        FunctionCallInProgressFrame,
                        FunctionCallResultFrame,
                    ),
                ):
                    await self.assistant.process_frame(frame, direction)
            elif processor is self.assistant and isinstance(frame, LLMContextFrame):
                await self.followups.put(frame)

        self.patch = patch.object(FrameProcessor, "push_frame", sink)
        self.patch.start()
        await self.service.process_frame(
            LLMUpdateSettingsFrame(delta=self.service.Settings(filter_incomplete_user_turns=True)),
            FrameDirection.DOWNSTREAM,
        )

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.service.cleanup()
        await self.user.cleanup()
        await self.assistant.cleanup()
        await self.service._client.close()
        self.patch.stop()

    async def utterance(self, text):
        for frame in (
            ProposedUserStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text, "user", "now"),
            VADUserStoppedSpeakingFrame(),
            ProposedUserStoppedSpeakingFrame(),
        ):
            await self.user.process_frame(frame, FrameDirection.DOWNSTREAM)
            if isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
                await self.service.process_frame(frame, FrameDirection.DOWNSTREAM)
        task = self.user._user_turn_controller._admission._decision_task
        if task:
            await asyncio.wait_for(asyncio.shield(task), 1)
        contexts = [
            call.args[0]
            for call in self.user.push_frame.await_args_list
            if isinstance(call.args[0], LLMContextFrame)
        ]
        return contexts[-1]

    async def begin_response(self, frame):
        task = self.service.create_task(
            self.service.process_frame(frame, FrameDirection.DOWNSTREAM), "controlled-response"
        )
        self.tasks.append(task)
        output = await asyncio.wait_for(self.service.requests.get(), 1)
        return task, output

    async def test_incomplete_response_then_resumed_speech_completes_current_inference(self):
        first = await self.utterance("Please book")
        task, output = await self.begin_response(first)
        output.set_result("◐")
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.stops, [])

        second = await self.utterance("a table for two.")
        self.assertNotEqual(first.inference_id, second.inference_id)
        task, output = await self.begin_response(second)
        output.set_result("● Certainly.")
        await asyncio.wait_for(task, 1)
        self.assertEqual([f.inference_id for f in self.completions], [second.inference_id])
        self.assertEqual(len(self.stops), 1)

    async def test_delayed_old_completion_does_not_prevent_current_response_completing(self):
        first = await self.utterance("Please book a table.")
        old_task, old_output = await self.begin_response(first)
        second = await self.utterance("Actually, make that tomorrow.")
        self.assertNotEqual(first.inference_id, second.inference_id)

        old_output.set_result("● Certainly.")
        await asyncio.wait_for(old_task, 1)
        self.assertEqual([f.inference_id for f in self.completions], [first.inference_id])
        self.assertEqual(self.stops, [])

        task, output = await self.begin_response(second)
        output.set_result("● Tomorrow it is.")
        await asyncio.wait_for(task, 1)
        self.assertEqual(len(self.stops), 1)
        self.assertEqual(self.completions[-1].inference_id, second.inference_id)

    async def test_resumed_signals_without_new_text_do_not_repeat_inference(self):
        inferences = []
        self.user.add_event_handler(
            "on_user_turn_inference_triggered", lambda *args: inferences.append(True)
        )
        first = await self.utterance("Please book a table.")
        for frame in (ProposedUserStartedSpeakingFrame(), ProposedUserStoppedSpeakingFrame()):
            await self.user.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(len(inferences), 1)
        self.assertEqual(
            sum(
                isinstance(call.args[0], LLMContextFrame)
                for call in self.user.push_frame.await_args_list
            ),
            1,
        )
        await self.user.process_frame(
            UserTurnInferenceCompletedFrame(inference_id=first.inference_id),
            FrameDirection.UPSTREAM,
        )
        self.assertEqual(self.stops, [])
        second = await self.utterance("Tomorrow, please.")
        task, output = await self.begin_response(second)
        output.set_result("● Tomorrow it is.")
        await asyncio.wait_for(task, 1)
        self.assertEqual(len(self.stops), 1)

    async def test_tool_result_followup_keeps_admitted_inference_and_can_speak(self):
        self.route_tools = True

        async def get_booking(params):
            await params.result_callback({"time": "tomorrow"})

        self.service.register_function("get_booking", get_booking)
        context = await self.utterance("When is my booking?")
        task, output = await self.begin_response(context)
        output.set_result(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "booking-call",
                        "type": "function",
                        "function": {"name": "get_booking", "arguments": "{}"},
                    }
                ]
            }
        )
        await asyncio.wait_for(task, 1)
        followup = await asyncio.wait_for(self.followups.get(), 1)
        self.assertEqual(followup.inference_id, context.inference_id)
        self.assertEqual(len(self.stops), 1)
        self.assertEqual(self.completions[0].inference_id, context.inference_id)

        task, output = await self.begin_response(followup)
        output.set_result("● Your booking is tomorrow.")
        await asyncio.wait_for(task, 1)
        self.assertTrue(any("Your booking is tomorrow." in text for text in self.texts))
        self.assertEqual(self.completions[-1].inference_id, context.inference_id)
        self.assertEqual(len(self.stops), 1)
