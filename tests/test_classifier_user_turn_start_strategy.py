#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Asynchronous turn admission across native endpoint boundaries."""

import asyncio
import unittest
from unittest.mock import AsyncMock

from pipecat.classifiers.base_classifier import ChoiceResult
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.turns.user_start import ClassifierUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import (
    FilterIncompleteUserTurnStrategies,
    UserTurnStrategies,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def verdict(choice):
    return {"turn": ChoiceResult(choice=choice, confidence=1, probabilities={choice: 1})}


class TestBackchannelStart(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.classifier = AsyncMock()
        self.release = asyncio.Event()
        self.called = asyncio.Event()
        self.events = []
        self.controllers = []

        async def decide(*args, **kwargs):
            self.called.set()
            await self.release.wait()
            return verdict("interrupt")

        self.classifier.choice.side_effect = decide

    async def asyncTearDown(self):
        for controller in self.controllers:
            await controller.cleanup()

    async def controller(self, *, incomplete=False, timeout=1):
        start = ClassifierUserTurnStartStrategy(
            classifier=self.classifier,
            context=lambda: {
                "recent_conversation": [{"role": "user", "content": "When will my order arrive?"}]
            },
            cooldown=0.01,
            timeout=timeout,
        )
        config = FilterIncompleteUserTurnStrategies if incomplete else UserTurnStrategies
        controller = UserTurnController(
            user_turn_strategies=config(
                start=[start],
                stop=[ExternalUserTurnStopStrategy()],
            )
        )
        await controller.setup(frame_processor_setup(TaskManager()))
        self.controllers.append(controller)
        for name in [
            "on_user_turn_started",
            "on_user_turn_inference_triggered",
            "on_user_turn_stopped",
            "on_reset_aggregation",
        ]:

            async def record(*args, event=name):
                self.events.append(event)

            controller.add_event_handler(name, record)
        await controller.process_frame(BotStartedSpeakingFrame())
        return controller, start

    async def send(self, controller, text="Wait."):
        for frame in [
            ProposedUserStartedSpeakingFrame(),
            TranscriptionFrame(text, "user", "now"),
            ProposedUserStoppedSpeakingFrame(),
        ]:
            await controller.process_frame(frame)

    async def settle(self, start):
        controller = next(c for c in self.controllers if start in c.user_turn_strategies.start)
        task = controller._admission._decision_task
        if task:
            await asyncio.wait_for(asyncio.shield(task), 1)

    async def test_endpoint_before_verdict_starts_and_stops_once(self):
        controller, start = await self.controller()
        await self.send(controller)
        await self.called.wait()
        self.assertEqual(self.events, [])
        self.release.set()
        await self.settle(start)
        self.assertEqual(
            self.events,
            ["on_user_turn_started", "on_user_turn_inference_triggered", "on_user_turn_stopped"],
        )
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertEqual(self.events.count("on_user_turn_stopped"), 1)

    async def test_rejection_clears_aggregation_and_next_utterance_can_start(self):
        self.classifier.choice.side_effect = [verdict("continue"), verdict("interrupt")]
        controller, start = await self.controller()
        await self.send(controller, "Yes.")
        await self.settle(start)
        self.assertEqual(self.events, ["on_reset_aggregation"])
        await self.send(controller)
        await self.settle(start)
        self.assertEqual(self.events.count("on_user_turn_stopped"), 1)

    async def test_timeout_after_endpoint_admits_and_finishes(self):
        controller, start = await self.controller(timeout=0.02)
        await self.send(controller)
        await self.settle(start)
        self.assertEqual(
            self.events,
            ["on_user_turn_started", "on_user_turn_inference_triggered", "on_user_turn_stopped"],
        )

    async def test_error_admits_and_finishes(self):
        self.classifier.choice.side_effect = RuntimeError("unavailable")
        controller, start = await self.controller()
        await self.send(controller)
        await self.settle(start)
        self.assertIn("on_user_turn_stopped", self.events)

    async def test_deferred_stop_waits_for_completion_after_late_start(self):
        controller, start = await self.controller(incomplete=True)
        await self.send(controller)
        self.release.set()
        await self.settle(start)
        self.assertEqual(self.events, ["on_user_turn_started", "on_user_turn_inference_triggered"])
        await controller.process_frame(
            UserTurnInferenceCompletedFrame(inference_id=controller.inference_id)
        )
        self.assertEqual(self.events[-1], "on_user_turn_stopped")

    async def test_updated_transcript_rechecks_without_new_start_signal(self):
        self.classifier.choice.side_effect = [verdict("continue"), verdict("interrupt")]
        controller, start = await self.controller()
        await controller.process_frame(ProposedUserStartedSpeakingFrame())
        await controller.process_frame(InterimTranscriptionFrame("That", "user", "now"))
        await self.settle(start)
        self.assertEqual(self.events, [])
        await controller.process_frame(TranscriptionFrame("That's wrong.", "user", "now"))
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        await self.settle(start)
        self.assertIn("on_user_turn_stopped", self.events)
        self.assertEqual(self.classifier.choice.call_count, 2)

    async def test_verdict_before_endpoint_keeps_turn_open(self):
        controller, start = await self.controller()
        await controller.process_frame(ProposedUserStartedSpeakingFrame())
        await controller.process_frame(InterimTranscriptionFrame("Wait", "user", "now"))
        self.release.set()
        await self.settle(start)
        self.assertEqual(self.events, ["on_user_turn_started"])
        await controller.process_frame(TranscriptionFrame("Wait.", "user", "now"))
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertEqual(self.events[-1], "on_user_turn_stopped")

    async def test_bot_stop_admits_pending_speech_and_cancels_decision(self):
        controller, start = await self.controller()
        await self.send(controller)
        await self.called.wait()
        await controller.process_frame(BotStoppedSpeakingFrame())
        await self.settle(start)
        self.assertEqual(self.events[-1], "on_user_turn_stopped")
        self.assertTrue(
            controller._admission._decision_task is None
            or controller._admission._decision_task.done()
        )

    async def test_new_utterance_preserves_pending_ended_candidate(self):
        controller, start = await self.controller()
        await self.send(controller, "Wait.")
        await self.called.wait()
        await self.send(controller, "Please.")
        self.release.set()
        await self.settle(start)
        self.assertEqual(self.events.count("on_user_turn_started"), 1)
        self.assertNotIn("on_reset_aggregation", self.events)

    async def test_bot_idle_bypasses_classifier(self):
        controller, start = await self.controller()
        await controller.process_frame(BotStoppedSpeakingFrame())
        await self.send(controller)
        await self.settle(start)
        self.classifier.choice.assert_not_called()
        self.assertEqual(self.events[-1], "on_user_turn_stopped")

    async def test_aggregator_records_only_admitted_transcript_once(self):
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.tests.utils import SleepFrame, run_test

        async def decide(state, questions):
            await asyncio.sleep(0.02)
            return verdict("interrupt" if state["user_speech"] == "Wait." else "continue")

        self.classifier.choice.side_effect = decide
        start = ClassifierUserTurnStartStrategy(classifier=self.classifier, context=dict)
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[start],
                    stop=[ExternalUserTurnStopStrategy()],
                ),
            ),
        )
        frames = [BotStartedSpeakingFrame()]
        for text in ["Yes.", "Wait."]:
            frames.extend(
                [
                    ProposedUserStartedSpeakingFrame(),
                    TranscriptionFrame(text, "user", "now"),
                    ProposedUserStoppedSpeakingFrame(),
                    SleepFrame(sleep=0.1),
                ]
            )
        await run_test(Pipeline([user, assistant]), frames_to_send=frames)
        self.assertEqual(
            [m for m in context.get_messages() if m["role"] == "user"],
            [{"role": "user", "content": "Wait."}],
        )

    async def test_vad_endpoint_before_decision_uses_speech_timeout(self):
        from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
        from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

        controller, start = await self.controller()
        await controller.update_strategies(
            UserTurnStrategies(
                start=[start],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)],
            )
        )
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame("Wait.", "user", "now"))
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.release.set()
        await self.settle(start)
        await asyncio.sleep(0.05)
        self.assertEqual(self.events.count("on_user_turn_stopped"), 1)

    async def test_cleanup_cancels_pending_result(self):
        controller, start = await self.controller()
        await self.send(controller)
        await self.called.wait()
        await controller.cleanup()
        self.controllers.remove(controller)
        self.release.set()
        await asyncio.sleep(0)
        self.assertEqual(self.events, [])
        self.classifier.cleanup.assert_awaited_once()

    async def test_final_segments_are_combined_for_reclassification(self):
        self.classifier.choice.side_effect = [verdict("continue"), verdict("interrupt")]
        controller, start = await self.controller()
        await controller.process_frame(ProposedUserStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame("Yes", "user", "now"))
        await self.settle(start)
        await controller.process_frame(TranscriptionFrame("but that's wrong.", "user", "now"))
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        await self.settle(start)
        self.assertEqual(
            self.classifier.choice.call_args.args[0]["user_speech"], "Yes but that's wrong."
        )
        self.assertIn("on_user_turn_stopped", self.events)
