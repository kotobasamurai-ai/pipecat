# Classifier-based user turn admission

`ClassifierUserTurnStartStrategy` starts a user turn when speech requires the
bot's attention. While the bot speaks, it classifies the latest transcript in
conversation context. While the bot is silent, it admits speech as soon as text
is available.

Configure it in `LLMUserAggregatorParams` as the sole start strategy:

```python
import os

from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_buffer import LLMResponseBuffer
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import ClassifierUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

context = LLMContext()
response_buffer = LLMResponseBuffer()
strategies = UserTurnStrategies(
    start=[
        ClassifierUserTurnStartStrategy(
            classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),
            context=lambda: {
                "conversation": context.get_messages(),
                "current_assistant_response": response_buffer.text,
            },
            cooldown=1.5,
            timeout=1.0,
        )
    ],
    stop=[ExternalUserTurnStopStrategy()],
)
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(user_turn_strategies=strategies),
)
```

Place `response_buffer` immediately after the LLM and before TTS in the pipeline:

```python
pipeline = Pipeline([
    transport.input(), stt, user_aggregator,
    llm, response_buffer, tts, transport.output(), assistant_aggregator,
])
```

The callback reads the committed conversation and the latest response's generated
text, including text awaiting synthesis or playback. The buffer survives the end
of generation and resets on a new response, interruption, or session end. It does
not predict ungenerated text or indicate what the caller has heard. A new LLM
generation replaces the buffer, including a generation following a tool call.
`TTSSpeakFrame` messages passing through it also replace the buffer. Speech from
other pipeline paths must be supplied by the application if it is to be classified.

The strategy copies the supplied state and adds `user_speech`: accumulated final
text plus the latest interim. Interim updates replace one another and stay out
of conversation history. The callback must be fast and synchronous. Each strategy
owns its classifier connection.

## Classification and conversation history

The default question asks whether a question's answer is explicitly present in
the current response. Answers to the assistant's questions, selections, and
consent admit a user turn. Recognition delay does not imply that the caller heard
an answer. Explicit requests to stop, repeat, or clarify also require yielding.

| Choice | Action at the end of a rejected utterance | LLM context |
| --- | --- | --- |
| `backchannel` | Emit `UserBackchannelFrame` with the acknowledgement | Omit |
| `continue` | Let the current response address the request | Omit |
| `interrupt` | Admit the turn and apply normal end detection | Add final transcripts |

Both non-interrupt choices remain provisional while speech continues. Applications
can record each verdict, including its confidence and probabilities:

```python
start = strategies.start[0]

@start.event_handler("on_classification")
async def on_classification(strategy, text, result):
    logger.info("User speech {!r}: {} ({:.2f})", text, result.choice, result.confidence)
```

Here `start` is the configured `ClassifierUserTurnStartStrategy` instance.
The event reports a model verdict, not a committed turn action; updated speech,
bot silence, or a timeout can change the eventual action. A custom `question`
may change the instructions while preserving the three choice names.

## Start admission and end detection

The controller observes speech before announcing a user turn. Stop detectors
receive the original frames during this period, so their timers and audio
analysis can finish while classification is pending. Admission retains their
results, rather than replaying endpoint frames or running audio analysis again.

| Input and configuration | End detection |
| --- | --- |
| Native STT turn proposals | `ExternalUserTurnStopStrategy` |
| VAD start/stop and transcripts | `SpeechTimeoutUserTurnStopStrategy` |
| VAD, audio and Smart Turn | `TurnAnalyzerUserTurnStopStrategy` |

A VAD stop indicates a pause in audio. The configured stop strategy still
decides when that pause permits inference, including its transcript wait.
Native proposals and VAD observations update physical speaking state separately.
A held end is released only when neither reports ongoing speech.

If the detector finishes before classification, an interrupt decision announces
the start and then releases the held inference/end. If classification finishes
first, the admitted turn waits for its normal stop detector.

`FilterIncompleteUserTurnStrategies(start=[...], stop=[...])` also works:
admission releases the detector's inference readiness, and the LLM's completion
verdict finalizes the turn. An inference identifier prevents a delayed verdict
from finalizing a different inference.

## Decision lifecycle

- The first available transcript starts classification immediately. VAD or native
  start proposals alone supply no text to classify.
- Transcript updates are coalesced. `cooldown` limits request starts; it does not
  postpone the first request or restart an in-flight request.
- Each request has a fixed timeout. Expiry or classifier failure admits speech.
- An interrupt result admits all speech accumulated while the decision was
  pending, including resumed speech. A rejection applies only to the exact text
  revision that was classified.
- A rejected revision is discarded from aggregation only when a detector reports
  its end. Updated text can trigger another decision without another start signal.
- Bot playback ending admits pending text. Session cancellation invalidates
  outstanding decisions and discards unadmitted aggregation.

Confirmed acknowledgements are reported through the aggregator's
`on_user_backchannel` event and RTVI. Requests addressed by the current response
have a `continue` classification event but no backchannel frame. Applications
that need a complete transcript can record incoming STT frames separately.

## Extension boundary

Subclass `BaseAsyncUserTurnStartStrategy` to supply another asynchronous decision
policy. The controller owns candidate state, serialization and end-result
retention; the strategy supplies `decide(text)` and, optionally, `needs_decision`.
Return `UserTurnStartDecision` to distinguish acknowledgements from other
non-interrupting speech. Boolean decisions also work: `True` admits and `False`
keeps listening without reporting a backchannel.

Stop strategies declare their `pending_turn_support` using `PendingTurnSupport`.
`OBSERVATION` detectors are armed at the first speech observation and retain their
state across public admission. `COMPLETION` finalizers are armed at public start.
`NONE` requires synchronous admission. Configuration frames still reach both
supported kinds.

Asynchronous admission requires the user aggregator to own pending transcripts.
Standalone `UserTurnProcessor` rejects this configuration because it forwards
transcripts before a decision. Eager speculative stop strategies and additional
start strategies are also rejected at configuration time.
