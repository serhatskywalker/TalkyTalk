"""
Early / Partial Intent Inference.

"Kullanıcı konuşurken sistem düşünmeye başlasın"

This predictor produces progressive hypothesis updates
as audio streams in, without waiting for utterance completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections import deque

from talkytalk.core.packet import Intent
from talkytalk.predictors.base import Predictor, PredictionContext

if TYPE_CHECKING:
    from talkytalk.core.pipeline import PipelineState


@dataclass
class Hypothesis:
    """A partial intent hypothesis with temporal tracking."""
    intent: Intent
    confidence: float
    first_seen_ms: int
    last_updated_ms: int
    evidence_count: int = 1
    
    @property
    def age_ms(self) -> int:
        """How long this hypothesis has been active."""
        return self.last_updated_ms - self.first_seen_ms
    
    @property
    def stability_score(self) -> float:
        """Higher = more stable hypothesis."""
        age_factor = min(1.0, self.age_ms / 500)
        evidence_factor = min(1.0, self.evidence_count / 10)
        return (age_factor * 0.4 + evidence_factor * 0.3 + self.confidence * 0.3)


@dataclass
class EarlyIntentState:
    """Tracks progressive intent hypotheses."""
    hypotheses: dict[Intent, Hypothesis] = field(default_factory=dict)
    leading_hypothesis: Intent = Intent.UNKNOWN
    leading_confidence: float = 0.0
    interruptibility: float = 0.0
    hypothesis_stable: bool = False
    frames_since_change: int = 0


class EarlyIntentPredictor(Predictor):
    """
    Progressive intent inference during speech.
    
    Key differentiator from standard IntentPredictor:
    - Produces hypotheses *during* speech, not after
    - Tracks hypothesis stability over time
    - Computes interruptibility score
    - Enables "thinking while listening"
    
    Interruptibility Score (0.0 - 1.0):
    - 0.0 = Do not interrupt, hypothesis unstable
    - 0.5 = May prepare response, don't act yet
    - 1.0 = Safe to begin response generation
    
    Usage for downstream LLM:
    - interruptibility > 0.3 → Start prefetching/preparing
    - interruptibility > 0.6 → Begin generating (speculatively)
    - interruptibility > 0.8 + interrupt_safe → Deliver response
    """
    
    def __init__(
        self,
        stability_threshold: int = 5,
        confidence_momentum: float = 0.8,
        hypothesis_timeout_ms: int = 2000,
    ) -> None:
        self._stability_threshold = stability_threshold
        self._confidence_momentum = confidence_momentum
        self._hypothesis_timeout_ms = hypothesis_timeout_ms
        
        self._state = EarlyIntentState()
        self._confidence_history: deque[float] = deque(maxlen=10)
    
    @property
    def name(self) -> str:
        return "early_intent"
    
    @property
    def early_state(self) -> EarlyIntentState:
        """Access to early intent state for external consumers."""
        return self._state
    
    def predict(self, context: PredictionContext, state: PipelineState) -> None:
        timestamp = context.frame.timestamp_ms
        
        self._prune_stale_hypotheses(timestamp)
        
        current_intent = state.current_intent
        current_confidence = state.intent_confidence
        
        self._update_hypothesis(current_intent, current_confidence, timestamp)
        
        self._select_leading_hypothesis()
        
        self._compute_interruptibility(context, state)
        
        self._check_stability()
        
        state.analysis_results["early_intent"] = type(
            "EarlyIntentResult", (), {
                "data": {
                    "leading_intent": self._state.leading_hypothesis.value,
                    "leading_confidence": self._state.leading_confidence,
                    "interruptibility": self._state.interruptibility,
                    "hypothesis_stable": self._state.hypothesis_stable,
                    "active_hypotheses": len(self._state.hypotheses),
                    "frames_since_change": self._state.frames_since_change,
                }
            }
        )()
    
    def _update_hypothesis(
        self,
        intent: Intent,
        confidence: float,
        timestamp: int,
    ) -> None:
        """Update or create hypothesis for given intent."""
        if intent in self._state.hypotheses:
            h = self._state.hypotheses[intent]
            h.confidence = (
                h.confidence * self._confidence_momentum +
                confidence * (1 - self._confidence_momentum)
            )
            h.last_updated_ms = timestamp
            h.evidence_count += 1
        else:
            self._state.hypotheses[intent] = Hypothesis(
                intent=intent,
                confidence=confidence,
                first_seen_ms=timestamp,
                last_updated_ms=timestamp,
            )
    
    def _prune_stale_hypotheses(self, timestamp: int) -> None:
        """Remove hypotheses that haven't been updated recently."""
        stale = [
            intent for intent, h in self._state.hypotheses.items()
            if (timestamp - h.last_updated_ms) > self._hypothesis_timeout_ms
        ]
        for intent in stale:
            del self._state.hypotheses[intent]
    
    def _select_leading_hypothesis(self) -> None:
        """Select the best hypothesis based on stability and confidence."""
        if not self._state.hypotheses:
            new_leader = Intent.UNKNOWN
            new_confidence = 0.0
        else:
            best = max(
                self._state.hypotheses.values(),
                key=lambda h: h.stability_score
            )
            new_leader = best.intent
            new_confidence = best.confidence
        
        if new_leader != self._state.leading_hypothesis:
            self._state.frames_since_change = 0
        else:
            self._state.frames_since_change += 1
        
        self._state.leading_hypothesis = new_leader
        self._state.leading_confidence = new_confidence
    
    def _compute_interruptibility(
        self,
        context: PredictionContext,
        state: PipelineState,
    ) -> float:
        """
        Compute interruptibility score.
        
        Factors:
        - Hypothesis stability
        - Confidence level
        - Timing signals (pause, silence)
        - Speech likelihood
        """
        if not self._state.hypotheses:
            self._state.interruptibility = 0.0
            return 0.0
        
        leading = self._state.hypotheses.get(self._state.leading_hypothesis)
        if not leading:
            self._state.interruptibility = 0.0
            return 0.0
        
        stability_factor = leading.stability_score
        
        confidence_factor = self._state.leading_confidence
        
        timing_factor = 0.0
        if state.timing.user_paused:
            timing_factor = 0.5
        if state.timing.interrupt_safe:
            timing_factor = 1.0
        
        speech_factor = 1.0 - state.timing.speech_likelihood
        
        interruptibility = (
            stability_factor * 0.25 +
            confidence_factor * 0.25 +
            timing_factor * 0.30 +
            speech_factor * 0.20
        )
        
        self._state.interruptibility = min(1.0, max(0.0, interruptibility))
        return self._state.interruptibility
    
    def _check_stability(self) -> None:
        """Check if leading hypothesis is stable."""
        self._state.hypothesis_stable = (
            self._state.frames_since_change >= self._stability_threshold and
            self._state.leading_confidence > 0.5
        )
    
    def reset(self) -> None:
        self._state = EarlyIntentState()
        self._confidence_history.clear()
