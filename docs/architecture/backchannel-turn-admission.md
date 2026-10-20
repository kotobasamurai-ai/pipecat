# Classifier-based user turn admission

`ClassifierUserTurnStartStrategy` starts a user turn when speech requires the
bot's attention. While the bot speaks, it classifies the latest transcript in
conversation context. While the bot is silent, it admits speech as soon as text
is available.

Configure it in `LLMUserAggregatorParams` as the sole start strategy:

```python
import os

from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import ClassifierUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

context = LLMContext()
strategies = UserTurnStrategies(
    start=[
        ClassifierUserTurnStartStrategy(
            classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),
            context=lambda: {
                "conversation": context.get_messages(),
                "pending_assistant_text": assistant_aggregator.aggregation_string(),
            },
            cooldown=0.3,
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

The callback reads the complete committed conversation and the assistant's
uncommitted text for each decision. It runs after the aggregator pair has been
constructed, so `assistant_aggregator` is available when the callback executes.
`aggregation_string()` reads the existing buffer without committing or clearing
it. This text reflects frames received by the assistant aggregator, rather than
an exact audio playback position. The strategy copies the supplied state and
adds the pending user transcript separately. The callback must be fast and
synchronous. Each strategy owns its classifier connection.

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

Confirmed acknowledgements do not enter the LLM conversation context. Applications
that need a complete transcript can record incoming STT frames separately.

## Extension boundary

Subclass `BaseAsyncUserTurnStartStrategy` to supply another asynchronous decision
policy. The controller owns candidate state, serialization and end-result
retention; the strategy supplies `decide(text)` and, optionally, `needs_decision`.

Stop strategies declare their `pending_turn_support` using `PendingTurnSupport`.
`OBSERVATION` detectors are armed at the first speech observation and retain their
state across public admission. `COMPLETION` finalizers are armed at public start.
`NONE` requires synchronous admission. Configuration frames still reach both
supported kinds.

Asynchronous admission requires the user aggregator to own pending transcripts.
Standalone `UserTurnProcessor` rejects this configuration because it forwards
transcripts before a decision. Eager speculative stop strategies and additional
start strategies are also rejected at configuration time.
