#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Completion attribution across asynchronous inference lifetimes."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    LLMContextFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionLLMServiceMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_call", [False, True])
async def test_completion_keeps_origin_when_new_context_arrives(tool_call):
    service = UserTurnCompletionLLMServiceMixin()
    received = []
    old_context_received = asyncio.Event()
    new_context_received = asyncio.Event()

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        received.append(frame)

    async def old_response():
        await service.process_frame(
            LLMContextFrame(LLMContext(), inference_id="old"), FrameDirection.DOWNSTREAM
        )
        old_context_received.set()
        await new_context_received.wait()
        if tool_call:
            await service.push_frame(FunctionCallsStartedFrame(function_calls=[]))
        else:
            await service._push_turn_text("● Hello")

    async def new_context():
        await old_context_received.wait()
        await service.process_frame(
            LLMContextFrame(LLMContext(), inference_id="new"), FrameDirection.DOWNSTREAM
        )
        new_context_received.set()

    with patch.object(FrameProcessor, "push_frame", side_effect=capture):
        await asyncio.gather(old_response(), new_context())

    completed = [f for f in received if isinstance(f, UserTurnInferenceCompletedFrame)]
    assert len(completed) == 2
    assert all(f.inference_id == "old" for f in completed)


@pytest.mark.asyncio
async def test_unattributed_context_clears_previous_attribution():
    service = UserTurnCompletionLLMServiceMixin()
    service.push_frame = AsyncMock()
    await service.process_frame(
        LLMContextFrame(LLMContext(), inference_id="previous"), FrameDirection.DOWNSTREAM
    )
    await service.process_frame(LLMContextFrame(LLMContext()), FrameDirection.DOWNSTREAM)
    await service._push_turn_text("● Hello")
    completed = [
        call.args[0]
        for call in service.push_frame.call_args_list
        if isinstance(call.args[0], UserTurnInferenceCompletedFrame)
    ]
    assert len(completed) == 2
    assert all(f.inference_id is None for f in completed)
