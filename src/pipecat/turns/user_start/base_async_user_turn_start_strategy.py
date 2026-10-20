#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Asynchronous admission of observed user speech."""

from abc import abstractmethod

from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class BaseAsyncUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Decide whether observed speech should start a public user turn.

    The controller owns pending speech and detector results. Implementations
    return decisions without emitting start events or resetting aggregation.
    Configure this as the sole start strategy in ``LLMUserAggregatorParams``,
    with compatible stop strategies. The aggregator retains pending transcripts
    and discards rejected speech.
    """

    def __init__(self, *, cooldown: float = 0.3, timeout: float = 1.0, **kwargs):
        """Configure the decision schedule.

        Args:
            cooldown: Minimum seconds between request starts. Initial decisions
                run immediately; updates are coalesced into the latest text.
            timeout: Maximum seconds per decision; expiry admits the speech.
                Text updates do not cancel or extend an in-flight decision.
            **kwargs: Arguments passed to the start strategy base class.
        """
        super().__init__(**kwargs)
        if cooldown < 0 or timeout <= 0:
            raise ValueError("cooldown must be nonnegative and timeout must be positive")
        self.cooldown = cooldown
        self.timeout = timeout

    @property
    def resolves_proposed_turn_start_frames(self) -> bool:
        """Resolve native proposals through transcript admission."""
        return True

    @property
    def needs_decision(self) -> bool:
        """Whether speech requires classification rather than immediate admission."""
        return True

    @abstractmethod
    async def decide(self, text: str) -> bool:
        """Return True to admit speech, or False to keep listening for updates."""
        raise NotImplementedError
