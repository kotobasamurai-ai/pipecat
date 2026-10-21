#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Serialized admission of speech observed before a public user turn starts."""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserBackchannelFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import UserTurnSpeculation
from pipecat.turns.user_start.base_async_user_turn_start_strategy import (
    BaseAsyncUserTurnStartStrategy,
    UserTurnStartDecision,
)
from pipecat.turns.user_start.base_user_turn_start_strategy import UserTurnStartedParams
from pipecat.turns.user_stop.base_user_turn_stop_strategy import (
    BaseUserTurnStopStrategy,
    PendingTurnSupport,
    UserTurnStoppedParams,
)

if TYPE_CHECKING:
    from pipecat.turns.user_turn_controller import UserTurnController


@dataclass
class _Candidate:
    """One unresolved aggregate, including speech resumed before its admission."""

    text: str = ""
    final_text: str = ""
    revision: int = 0
    rejected_revision: int = -1
    decision: UserTurnStartDecision | None = None
    user_id: str = ""
    timestamp: str = ""
    inferred: tuple[int, int] | None = None  # Text revision and speech activity at inference.
    activity: int = 0
    inference: tuple[BaseUserTurnStopStrategy, UserTurnSpeculation | None] | None = None
    stop: tuple[BaseUserTurnStopStrategy, UserTurnStoppedParams] | None = None
    last_request: float = float("-inf")


class UserTurnAdmission:
    """Own pending speech while an asynchronous start strategy decides.

    A pending aggregate absorbs resumed speech. Acceptance is monotonic:
    an earlier request to interrupt still admits the combined speech. Rejection
    applies only to the exact text revision judged and only at a detector's end.
    """

    def __init__(self, controller: "UserTurnController", strategy: BaseAsyncUserTurnStartStrategy):
        """Bind admission to its state owner and decision policy."""
        self.controller = controller
        self.strategy = strategy
        self.candidate: _Candidate | None = None
        self.closed = False
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task | None = None
        self._decision_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._last_inference_id: str | None = None
        self._speaking_sources: set[str] = set()

    @asynccontextmanager
    async def transaction(self):
        """Serialize observations, aggregation, and asynchronous decision application.

        Synchronous event callbacks re-enter from the owning task. Cancellation
        invalidates state before waiting for this transaction, so a returning
        decision cannot re-enter it during shutdown.
        """
        task = asyncio.current_task()
        if self._owner is task:
            yield
            return
        async with self._lock:
            self._owner = task
            try:
                yield
            finally:
                self._owner = None

    def invalidate(self):
        """Invalidate results immediately, even during an awaited event handler."""
        self.closed = True
        self.candidate = None
        # Input tasks also own transactions. Release awaited callbacks on shutdown.
        if self._owner is not None and self._owner is not asyncio.current_task():
            self._owner.cancel()
        for task in self._tasks:
            if task is not asyncio.current_task():
                task.cancel()

    async def cleanup(self):
        """Invalidate pending speech and join outstanding decision tasks."""
        self.invalidate()
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                await self.controller.cancel_task(task)

    def valid(self, candidate: _Candidate) -> bool:
        """Check that a result still belongs to live pending speech."""
        return not self.closed and self.candidate is candidate

    @property
    def inference_id(self) -> str | None:
        """Return the token carried by context and completion frames."""
        return self._last_inference_id

    async def observe(self, frame: Frame):
        """Observe each original frame once, before detector processing."""
        starts = isinstance(frame, (ProposedUserStartedSpeakingFrame, VADUserStartedSpeakingFrame))
        transcript = isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
        if (
            starts
            or transcript
            or isinstance(frame, (ProposedUserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame))
        ):
            self.controller._user_turn_stop_timeout_event.set()
        if self.candidate is None and (starts or (transcript and frame.text.strip())):
            self.candidate = _Candidate()
            for detector in self.controller.user_turn_strategies.stop or []:
                if detector.pending_turn_support == PendingTurnSupport.OBSERVATION:
                    await detector.handle_user_turn_started()
        c = self.candidate
        if isinstance(frame, ProposedUserStartedSpeakingFrame):
            self._speaking_sources.add("native")
        elif isinstance(frame, ProposedUserStoppedSpeakingFrame):
            self._speaking_sources.discard("native")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking_sources.add("vad")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking_sources.discard("vad")
        self.controller._user_speaking = bool(self._speaking_sources)
        if c is None:
            return
        if starts:
            c.activity += 1
            c.inference = None
            c.stop = None
        if transcript and frame.text.strip():
            c.user_id = frame.user_id
            c.timestamp = frame.timestamp
            text = " ".join(part for part in (c.final_text, frame.text) if part)
            if isinstance(frame, TranscriptionFrame):
                c.final_text = text
            if c.text != text:
                c.text = text
                c.revision += 1
                if isinstance(frame, InterimTranscriptionFrame):
                    c.activity += 1
                    c.inference = None
                    c.stop = None

    def accepts_completion(self, frame: UserTurnInferenceCompletedFrame) -> bool:
        """Match completion to the current admitted inference."""
        c = self.candidate
        return bool(
            c
            and self.controller._user_turn
            and c.inferred == (c.revision, c.activity)
            and frame.inference_id is not None
            and frame.inference_id == self._last_inference_id
        )

    async def advance(self):
        """Admit idle-bot speech or schedule classification, then apply held ends."""
        c = self.candidate
        if self.closed or c is None:
            return
        if not self.controller._user_turn and c.text:
            if not self.strategy.needs_decision and self._decision_task is not None:
                if self._decision_task is not asyncio.current_task():
                    self._decision_task.cancel()
                    self._decision_task = None
            if self._decision_task is None and (
                not self.strategy.needs_decision or c.rejected_revision != c.revision
            ):
                task = self.controller.create_task(self._decide(c), f"{self.controller}::admission")
                self._decision_task = task
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        await self.flush(c)

    async def _admit(self, c: _Candidate):
        if not self.valid(c) or self.controller._user_turn:
            return
        await self.controller._trigger_user_turn_start(
            self.strategy,
            UserTurnStartedParams(
                enable_interruptions=self.strategy._enable_interruptions,
                enable_user_speaking_frames=self.strategy._enable_user_speaking_frames,
            ),
        )

    async def _decide(self, c: _Candidate):
        try:
            while self.valid(c) and not self.controller._user_turn:
                if self.strategy.needs_decision:
                    await asyncio.sleep(
                        max(0, c.last_request + self.strategy.cooldown - time.monotonic())
                    )
                async with self.transaction():
                    if not self.valid(c) or self.controller._user_turn:
                        return
                    if not self.strategy.needs_decision:
                        await self._admit(c)
                        await self.flush(c)
                        return
                    if c.rejected_revision == c.revision:
                        return
                    text, revision = c.text, c.revision
                    c.last_request = time.monotonic()
                try:
                    deadline = asyncio.timeout(self.strategy.timeout)
                    async with deadline:
                        decision = await self.strategy.decide(text)
                    if deadline.expired():
                        decision = True
                except Exception as exc:
                    logger.warning(f"{self.strategy}: admitting speech after {type(exc).__name__}")
                    decision = True
                if not self.valid(c) or asyncio.current_task() is not self._decision_task:
                    return
                async with self.transaction():
                    if (
                        not self.valid(c)
                        or asyncio.current_task() is not self._decision_task
                        or self.controller._user_turn
                    ):
                        return
                    if decision is True or decision is UserTurnStartDecision.INTERRUPT:
                        await self._admit(c)
                        await self.flush(c)
                        return
                    c.rejected_revision = revision
                    c.decision = UserTurnStartDecision.CONTINUE if decision is False else decision
                    await self.flush(c)
                    if c.rejected_revision == c.revision:
                        return
        finally:
            if self._decision_task is asyncio.current_task():
                self._decision_task = None

    async def inference(
        self, strategy: BaseUserTurnStopStrategy, speculation: UserTurnSpeculation | None
    ):
        """Retain a detector's inference readiness until admission."""
        c = self.candidate
        activity = c.activity if c else None
        async with self.transaction():
            if c is None or not self.valid(c) or activity != c.activity:
                return
            if self.controller._user_turn and c.inferred and c.inferred[0] == c.revision:
                return
            c.inference = (strategy, speculation)
            await self.flush(c)

    async def stopped(self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams):
        """Retain finalization without re-running its detector."""
        c = self.candidate
        activity = c.activity if c else None
        async with self.transaction():
            if c is None or not self.valid(c) or activity != c.activity:
                return
            c.stop = (strategy, params)
            await self.flush(c)

    async def flush(self, c: _Candidate):
        """Apply held outcomes once the speech and admission states permit it."""
        if not self.valid(c) or self.controller._user_speaking:
            return
        ended = c.inference is not None or c.stop is not None
        if not self.controller._user_turn:
            if ended and c.text and c.rejected_revision == c.revision:
                self.candidate = None
                for detector in self.controller.user_turn_strategies.stop or []:
                    if detector.pending_turn_support == PendingTurnSupport.OBSERVATION:
                        await detector.handle_user_turn_stopped()
                if not self.closed:
                    await self.controller._call_event_handler("on_reset_aggregation", self.strategy)
                    if c.decision is UserTurnStartDecision.BACKCHANNEL:
                        await self.strategy.push_frame(
                            UserBackchannelFrame(
                                text=c.final_text or c.text,
                                user_id=c.user_id,
                                timestamp=c.timestamp,
                            )
                        )
            return
        if c.inference is not None and (c.inferred is None or c.inferred[0] != c.revision):
            c.inferred = (c.revision, c.activity)
            self._last_inference_id = uuid.uuid4().hex
            strategy, speculation = c.inference
            c.inference = None
            await self.controller._trigger_user_turn_inference_triggered(strategy, speculation)
        if self.valid(c) and c.stop is not None:
            strategy, params = c.stop
            await self.controller._trigger_user_turn_stop(strategy, params)

    def finished(self):
        """Release the aggregate after its public turn ends."""
        self.candidate = None
