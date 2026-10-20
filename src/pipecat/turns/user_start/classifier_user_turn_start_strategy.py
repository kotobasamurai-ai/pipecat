#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Contextual classification of listening acknowledgements."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceQuestion
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_async_user_turn_start_strategy import (
    BaseAsyncUserTurnStartStrategy,
)

_QUESTION = ChoiceQuestion(
    instructions=(
        "The assistant is currently speaking. Decide whether the user's latest speech "
        "calls for the assistant to yield the floor now. Interpret the speech in the "
        "conversation context; the transcript may be partial."
    ),
    options={
        "continue": "A listening acknowledgement that lets the explanation continue.",
        "interrupt": "A request for attention, a question, correction, or request to stop.",
    },
)


class ClassifierUserTurnStartStrategy(BaseAsyncUserTurnStartStrategy):
    """Let acknowledgements pass while admitting questions, corrections and stops.

    The controller retains speech and endpoint results while the classifier
    runs. Unresolved successive utterances are combined until admission or a
    rejection of the latest text at an endpoint. Confirmed acknowledgements
    are discarded from aggregation. Bot silence admits pending speech.

    Example::

        start = ClassifierUserTurnStartStrategy(
            classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),
            context=lambda: {
                "conversation": context.get_messages(),
                "pending_assistant_text": assistant_aggregator.aggregation_string(),
            },
        )
        strategies = UserTurnStrategies(
            start=[start], stop=[ExternalUserTurnStopStrategy()]
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(user_turn_strategies=strategies),
        )
    """

    def __init__(
        self, *, classifier: BaseClassifier, context: Callable[[], dict[str, Any]], **kwargs
    ):
        """Initialize the classifier.

        Args:
            classifier: Classifier owned and cleaned up by this strategy.
            context: Fast synchronous supplier of conversation context, called
                for each decision. The observed user speech is added separately.
            **kwargs: Async start parameters, including cooldown and timeout.
        """
        super().__init__(**kwargs)
        self._classifier = classifier
        self._context = context
        self._bot_speaking = False

    @property
    def needs_decision(self) -> bool:
        """Classify only while the bot speaks."""
        return self._bot_speaking

    async def setup(self, setup: FrameProcessorSetup):
        """Open the classifier connection."""
        await super().setup(setup)
        await self._classifier.setup(setup.task_manager)

    async def cleanup(self):
        """Close the owned classifier."""
        await self._classifier.cleanup()
        await super().cleanup()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Observe bot playback state."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        return ProcessFrameResult.CONTINUE

    async def decide(self, text: str) -> bool:
        """Ask whether the bot should yield to the observed speech."""
        state = deepcopy(self._context())
        state["user_speech"] = text
        results = await self._classifier.choice(state, {"turn": _QUESTION})
        return results["turn"].choice == "interrupt"
