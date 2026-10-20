#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Admission races exercised with real turn controllers and stop strategies."""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, Mock

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    InterimTranscriptionFrame,
    LLMUpdateSettingsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    TranscriptionFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_start.base_async_user_turn_start_strategy import (
    BaseAsyncUserTurnStartStrategy,
)
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy, SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import (
    FilterIncompleteUserTurnStrategies,
    UserTurnStrategies,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class ControlledStart(BaseAsyncUserTurnStartStrategy):
    """Expose each classification request to the test before it can complete."""

    def __init__(self):
        super().__init__(cooldown=0, timeout=5)
        self.requests = asyncio.Queue()

    async def decide(self, text):
        result = asyncio.get_running_loop().create_future()
        await self.requests.put((text, result))
        return await result


class TestAdmissionRegressions(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.resources = []
        self.tasks = []
        self.events = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for resource in reversed(self.resources):
            await resource.cleanup()

    async def make_controller(self, *, stop=None, incomplete=False, start=None, watchdog=5):
        start = start or ControlledStart()
        config = FilterIncompleteUserTurnStrategies if incomplete else UserTurnStrategies
        controller = UserTurnController(
            user_turn_strategies=config(
                start=[start], stop=[stop or ExternalUserTurnStopStrategy()]
            ),
            user_turn_stop_timeout=watchdog,
        )
        await controller.setup(frame_processor_setup(TaskManager()))
        self.resources.append(controller)
        for name in ("started", "inference_triggered", "stopped"):

            async def record(*args, event=name):
                self.events.append(event)

            controller.add_event_handler("on_user_turn_" + name, record)
        return controller, start

    async def utterance(self, controller, text):
        for frame in (
            ProposedUserStartedSpeakingFrame(),
            TranscriptionFrame(text, "user", "now"),
            ProposedUserStoppedSpeakingFrame(),
        ):
            await controller.process_frame(frame)

    async def request(self, start):
        return await asyncio.wait_for(start.requests.get(), 1)

    async def settle(self, controller):
        task = controller._admission._decision_task
        if task:
            await asyncio.wait_for(asyncio.shield(task), 1)

    def spawn(self, controller, coro):
        task = controller.create_task(coro, "admission-test")
        self.tasks.append(task)
        return task

    async def test_resumed_speech_preserves_unresolved_interrupt(self):
        controller, start = await self.make_controller()
        await self.utterance(controller, "Wait!")
        text, result = await self.request(start)
        self.assertEqual(text, "Wait!")
        await controller.process_frame(ProposedUserStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame("I need to correct that.", "u", "now"))
        result.set_result(True)
        await self.settle(controller)
        self.assertEqual(self.events, ["started"])
        self.assertEqual(controller._admission.candidate.text, "Wait! I need to correct that.")
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertEqual(self.events, ["started", "inference_triggered", "stopped"])

    async def test_stale_rejection_reclassifies_combined_speech(self):
        controller, start = await self.make_controller()
        await self.utterance(controller, "Yes.")
        _, first = await self.request(start)
        await self.utterance(controller, "But that's wrong.")
        first.set_result(False)
        text, second = await self.request(start)
        self.assertEqual(text, "Yes. But that's wrong.")
        self.assertEqual(self.events, [])
        second.set_result(True)
        await self.settle(controller)
        self.assertEqual(self.events, ["started", "inference_triggered", "stopped"])

    async def test_start_callback_and_resumed_speech_are_serialized(self):
        controller, start = await self.make_controller()
        entered, release, attempting = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def hold_start(*args):
            entered.set()
            await release.wait()

        controller.add_event_handler("on_user_turn_started", hold_start)
        await self.utterance(controller, "Wait!")
        _, result = await self.request(start)
        result.set_result(True)
        await asyncio.wait_for(entered.wait(), 1)

        async def resume():
            attempting.set()
            await controller.process_frame(ProposedUserStartedSpeakingFrame())

        task = self.spawn(controller, resume())
        await asyncio.wait_for(attempting.wait(), 1)
        self.assertFalse(task.done())
        release.set()
        await self.settle(controller)
        await asyncio.wait_for(task, 1)
        self.assertTrue(controller._user_speaking)
        self.assertEqual(self.events, ["started", "inference_triggered", "stopped"])
        await controller.process_frame(TranscriptionFrame("Another correction", "u", "now"))
        _, next_result = await self.request(start)
        next_result.set_result(True)
        await self.settle(controller)
        self.assertEqual(self.events.count("inference_triggered"), 1)
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertEqual(self.events.count("inference_triggered"), 2)

    async def test_cancel_while_public_start_callback_waits_cannot_infer(self):
        controller, start = await self.make_controller()
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold_start(*args):
            entered.set()
            await release.wait()

        controller.add_event_handler("on_user_turn_started", hold_start)
        await self.utterance(controller, "Wait!")
        _, result = await self.request(start)
        result.set_result(True)
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(controller.process_frame(CancelFrame()), 1)
        release.set()
        self.assertTrue(controller._admission.closed)
        self.assertEqual(self.events, ["started"])

    async def test_smart_turn_completion_is_not_analyzed_twice_on_admission(self):
        analyzer = Mock(spec=BaseTurnAnalyzer)
        analyzer.analyze_end_of_turn = AsyncMock(
            side_effect=[(EndOfTurnState.COMPLETE, None), (EndOfTurnState.INCOMPLETE, None)]
        )
        analyzer.cleanup = AsyncMock()
        stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)
        controller, start = await self.make_controller(stop=stop)
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame("Wait!", "u", "now", finalized=True))
        _, result = await self.request(start)
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertEqual(self.events, [])
        result.set_result(True)
        await self.settle(controller)
        self.assertEqual(self.events, ["started", "inference_triggered", "stopped"])
        analyzer.analyze_end_of_turn.assert_awaited_once()

    async def test_incomplete_turn_requires_current_inference_completion(self):
        controller, start = await self.make_controller(incomplete=True)
        await self.utterance(controller, "Wait!")
        _, result = await self.request(start)
        result.set_result(True)
        await self.settle(controller)
        first_id = controller.inference_id
        self.assertIsNotNone(first_id)
        await controller.process_frame(UserTurnInferenceCompletedFrame(inference_id=first_id))
        self.assertEqual(self.events.count("stopped"), 1)
        await self.utterance(controller, "Another correction.")
        _, result = await self.request(start)
        result.set_result(True)
        await self.settle(controller)
        second_id = controller.inference_id
        self.assertNotEqual(first_id, second_id)
        await controller.process_frame(UserTurnInferenceCompletedFrame(inference_id=first_id))
        await controller.process_frame(UserTurnInferenceCompletedFrame())
        self.assertEqual(self.events.count("stopped"), 1)
        await controller.process_frame(UserTurnInferenceCompletedFrame(inference_id=second_id))
        self.assertEqual(self.events.count("stopped"), 2)

    async def test_aggregator_keeps_resumed_speech_and_tags_emitted_context(self):
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.processors.frame_processor import FrameDirection

        start = ControlledStart()
        context = LLMContext()
        user, _ = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=FilterIncompleteUserTurnStrategies(
                    start=[start], stop=[ExternalUserTurnStopStrategy()]
                )
            ),
        )
        await user.setup(frame_processor_setup(TaskManager()))
        self.resources.append(user)
        user.push_frame = AsyncMock()
        user.broadcast_frame = AsyncMock()
        user.broadcast_interruption = AsyncMock()
        stopped = []
        stopped_event = asyncio.Event()

        async def record_stop(*args):
            stopped.append(True)
            stopped_event.set()

        user.add_event_handler("on_user_turn_stopped", record_stop)

        async def send(frame):
            await user.process_frame(frame, FrameDirection.DOWNSTREAM)

        for frame in (
            ProposedUserStartedSpeakingFrame(),
            TranscriptionFrame("Wait!", "u", "now"),
            ProposedUserStoppedSpeakingFrame(),
        ):
            await send(frame)
        _, first = await self.request(start)
        for frame in (
            ProposedUserStartedSpeakingFrame(),
            TranscriptionFrame("That's wrong.", "u", "now"),
            ProposedUserStoppedSpeakingFrame(),
        ):
            await send(frame)
        self.assertEqual(context.get_messages(), [])
        first.set_result(True)
        controller = user._user_turn_controller
        await self.settle(controller)
        self.assertEqual(
            context.get_messages(), [{"role": "user", "content": "Wait! That's wrong."}]
        )
        emitted = [
            call.args[0]
            for call in user.push_frame.await_args_list
            if isinstance(call.args[0], LLMContextFrame)
        ]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].inference_id, controller.inference_id)
        self.assertIsNotNone(emitted[0].inference_id)
        await send(UserTurnInferenceCompletedFrame(inference_id="previous-bot-response"))
        self.assertTrue(controller._user_turn)
        self.assertEqual(stopped, [])
        await send(UserTurnInferenceCompletedFrame(inference_id=emitted[0].inference_id))
        await asyncio.wait_for(stopped_event.wait(), 1)
        self.assertEqual(stopped, [True])

    async def test_incomplete_strategy_configures_llm_before_any_admission(self):
        controller, _ = await self.make_controller(incomplete=True)
        pushed = []

        async def record(controller, frame, direction):
            pushed.append(frame)

        controller.add_event_handler("on_push_frame", record)
        await controller.process_frame(StartFrame())
        settings = [frame for frame in pushed if isinstance(frame, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings), 1)
        self.assertTrue(settings[0].delta.filter_incomplete_user_turns)
        self.assertTrue(settings[0].reach_inactive_services)
        self.assertEqual(self.events, [])

    async def test_interim_after_external_inference_does_not_reuse_endpoint(self):
        controller, start = await self.make_controller(incomplete=True)
        await self.utterance(controller, "Wait!")
        _, decision = await self.request(start)
        decision.set_result(True)
        await self.settle(controller)
        original_id = controller.inference_id
        await controller.process_frame(InterimTranscriptionFrame("And", "u", "now"))
        self.assertEqual(self.events, ["started", "inference_triggered"])
        await controller.process_frame(UserTurnInferenceCompletedFrame(inference_id=original_id))
        self.assertEqual(self.events, ["started", "inference_triggered"])
        await controller.process_frame(TranscriptionFrame("And another thing.", "u", "now"))
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertEqual(self.events.count("inference_triggered"), 2)
        self.assertNotEqual(controller.inference_id, original_id)

    async def test_interim_after_speech_timeout_inference_does_not_reuse_endpoint(self):
        controller, start = await self.make_controller(
            incomplete=True, stop=SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)
        )
        inferred = asyncio.Event()

        async def record(*args):
            inferred.set()

        controller.add_event_handler("on_user_turn_inference_triggered", record)
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame("Wait!", "u", "now", finalized=True))
        _, decision = await self.request(start)
        decision.set_result(True)
        await self.settle(controller)
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.wait_for(inferred.wait(), 1)
        original_id = controller.inference_id
        await controller.process_frame(InterimTranscriptionFrame("And", "u", "now"))
        await controller.process_frame(UserTurnInferenceCompletedFrame(inference_id=original_id))
        self.assertEqual(self.events, ["started", "inference_triggered"])

    async def test_interim_activity_rearms_admitted_turn_watchdog(self):
        controller, start = await self.make_controller(incomplete=True, watchdog=0.12)
        await controller.start()
        await self.utterance(controller, "Wait!")
        _, decision = await self.request(start)
        decision.set_result(True)
        await self.settle(controller)
        timed_out = asyncio.Event()

        async def record(*args):
            timed_out.set()

        controller.add_event_handler("on_user_turn_stop_timeout", record)
        for text in ("And", "And another", "And another thing", "And another thing please"):
            await asyncio.sleep(0.04)
            await controller.process_frame(InterimTranscriptionFrame(text, "u", "now"))
            self.assertFalse(timed_out.is_set())
            self.assertTrue(controller._user_turn)
        await asyncio.wait_for(timed_out.wait(), 1)
        self.assertEqual(self.events.count("stopped"), 1)

    async def test_idle_admission_callback_does_not_block_cancel_processing(self):
        from pipecat.turns.user_start import ClassifierUserTurnStartStrategy

        for initially_speaking in (False, True):
            with self.subTest(initially_speaking=initially_speaking):
                classifier = AsyncMock()
                requested = asyncio.Event()
                blocked = asyncio.Event()

                async def decide(*args, requested=requested, blocked=blocked):
                    requested.set()
                    await blocked.wait()

                classifier.choice.side_effect = decide
                strategy = ClassifierUserTurnStartStrategy(classifier=classifier, context=dict)
                controller, _ = await self.make_controller(start=strategy)
                entered, release = asyncio.Event(), asyncio.Event()

                async def hold_start(*args, entered=entered, release=release):
                    entered.set()
                    await release.wait()

                controller.add_event_handler("on_user_turn_started", hold_start)
                if initially_speaking:
                    await controller.process_frame(BotStartedSpeakingFrame())
                await self.utterance(controller, "Wait!")
                if initially_speaking:
                    await asyncio.wait_for(requested.wait(), 1)
                    await asyncio.wait_for(controller.process_frame(BotStoppedSpeakingFrame()), 1)
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(controller.process_frame(CancelFrame()), 1)
                release.set()
                self.assertTrue(controller._admission.closed)
                self.assertNotIn("inference_triggered", self.events)
                self.assertNotIn("stopped", self.events)
                if not initially_speaking:
                    classifier.choice.assert_not_called()

    async def test_cooldown_survives_restarted_decision_task(self):
        start = ControlledStart()
        start.cooldown = 0.2
        controller, _ = await self.make_controller(start=start)
        before = time.monotonic()
        await controller.process_frame(ProposedUserStartedSpeakingFrame())
        await controller.process_frame(InterimTranscriptionFrame("Yes", "u", "now"))
        _, first = await asyncio.wait_for(start.requests.get(), 0.1)
        first_started = time.monotonic()
        self.assertLess(first_started - before, start.cooldown)
        first.set_result(False)
        await self.settle(controller)
        self.assertIsNone(controller._admission._decision_task)
        await controller.process_frame(InterimTranscriptionFrame("Yes but", "u", "now"))
        text, second = await self.request(start)
        self.assertEqual(text, "Yes but")
        self.assertGreaterEqual(time.monotonic() - first_started, start.cooldown - 0.015)
        second.set_result(True)
        await self.settle(controller)
