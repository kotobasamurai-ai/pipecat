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
    UserTurnStartDecision,
)

_QUESTION = ChoiceQuestion(
    instructions=(
        "Classify the user's latest speech during assistant playback. The conversation "
        "contains prior messages; current_assistant_response contains the response generated "
        "so far. The user transcript may be partial, and recognition may arrive late. "
        "Assess what the user contributes and whether a user turn is needed. A question "
        "whose answer is explicitly present in the current response can be handled by "
        "continuing it. Answers to assistant questions, selections, consent, corrections, "
        "and requests to stop, repeat, or clarify require accepting a user turn."
    ),
    options={
        "backchannel": (
            "A listening signal whose purpose is to acknowledge hearing or understanding."
        ),
        "continue": (
            "The user asks a question whose answer is explicitly present in "
            "current_assistant_response, so continuing that response answers the question."
        ),
        "interrupt": (
            "The user supplies requested information, makes a selection, grants or refuses "
            "permission, corrects something, asks for attention or repetition, or asks a "
            "question that needs an additional response."
        ),
    },
)


class ClassifierUserTurnStartStrategy(BaseAsyncUserTurnStartStrategy):
    """Yield when the user's speech requires changing or stopping the current response.

    The controller retains speech and endpoint results while the classifier
    runs. Unresolved successive utterances are combined until admission or a
    rejection of the latest text at an endpoint. Confirmed acknowledgements emit
    ``UserBackchannelFrame``. Acknowledgements and requests addressed by the
    current response stay out of context. Bot silence admits pending speech.

    Event handlers available:

    - on_classification: Called with the classified text and ``ChoiceResult``.
      Reports each model verdict, which later transcript updates may supersede.

    Example::

        from pipecat.processors.aggregators.llm_response_buffer import LLMResponseBuffer

        response_buffer = LLMResponseBuffer()  # Place between the LLM and TTS.
        start = ClassifierUserTurnStartStrategy(
            classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),
            context=lambda: {
                "conversation": context.get_messages(),
                "current_assistant_response": response_buffer.text,
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
        self,
        *,
        classifier: BaseClassifier,
        context: Callable[[], dict[str, Any]],
        question: ChoiceQuestion | None = None,
        cooldown: float = 1.5,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            classifier: Classifier owned and cleaned up by this strategy.
            context: Fast synchronous supplier of conversation context, called
                for each decision. The observed user speech is added separately.
            question: Classification instructions with ``backchannel``, ``continue``,
                and ``interrupt`` choices. Defaults to contextual response matching.
            cooldown: Minimum seconds between request starts. The first is immediate.
            **kwargs: Async start parameters, including timeout.
        """
        super().__init__(cooldown=cooldown, **kwargs)
        self._classifier = classifier
        self._context = context
        self._question = question or _QUESTION
        if set(self._question.options) != {d.value for d in UserTurnStartDecision}:
            raise ValueError("question must offer backchannel, continue, and interrupt")
        self._bot_speaking = False
        self._register_event_handler("on_classification")

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

    async def decide(self, text: str) -> UserTurnStartDecision:
        """Ask whether the bot should yield to the observed speech."""
        state = deepcopy(self._context())
        state["user_speech"] = text
        results = await self._classifier.choice(state, {"turn": self._question})
        result = results["turn"]
        decision = UserTurnStartDecision(result.choice)
        await self._call_event_handler("on_classification", text, result)
        return decision
