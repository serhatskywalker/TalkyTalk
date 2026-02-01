"""
Turn-Taking & Overlap Detection.

"Ne zaman araya girebilirim?"

Advanced turn-taking prediction that goes beyond
simple silence detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from collections import deque

from talkytalk.core.packet import Timing
from talkytalk.predictors.base import Predictor, PredictionContext

if TYPE_CHECKING:
    from talkytalk.core.pipeline import PipelineState


class TurnState(str, Enum):
    """Current turn-taking state."""
    USER_SPEAKING = "user_speaking"
    USER_PAUSING = "user_pausing"
    TURN_YIELDED = "turn_yielded"
    SYSTEM_CAN_SPEAK = "system_can_speak"
    OVERLAP_DETECTED = "overlap_detected"


class InterruptReason(str, Enum):
    """Why interruption is or isn't safe."""
    SPEAKING = "user_still_speaking"
    RISING_INTONATION = "question_forming"
    SHORT_PAUSE = "pause_too_short"
    FALLING_COMPLETE = "falling_intonation_complete"
    LONG_SILENCE = "extended_silence"
    HIGH_CONFIDENCE = "high_intent_confidence"
    FORCED = "forced_by_system"


@dataclass
class TurnTakingSignal:
    """Detailed turn-taking signal."""
    state: TurnState
    can_interrupt: bool
    should_wait: bool
    interrupt_reason: InterruptReason
    confidence: float
    
    suggested_wait_ms: int
    
    overlap_probability: float
    
    turn_duration_ms: int
    silence_in_turn_ms: int
    
    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "can_interrupt": self.can_interrupt,
            "should_wait": self.should_wait,
            "interrupt_reason": self.interrupt_reason.value,
            "confidence": self.confidence,
            "suggested_wait_ms": self.suggested_wait_ms,
            "overlap_probability": self.overlap_probability,
            "turn_duration_ms": self.turn_duration_ms,
            "silence_in_turn_ms": self.silence_in_turn_ms,
        }


class TurnTakingPredictor(Predictor):
    """
    Advanced turn-taking prediction.
    
    Goes beyond TimingPredictor by:
    - Tracking complete turn structure
    - Predicting overlap probability
    - Providing detailed interrupt recommendations
    - Considering prosodic turn-final cues
    
    Turn-Final Cues (linguistic research):
    - Falling pitch at phrase boundary
    - Lengthening of final syllable
    - Decrease in intensity
    - Specific pause patterns
    
    Signals:
    - can_interrupt: Technical possibility
    - should_wait: Strategic recommendation
    - overlap_probability: Risk of talking over user
    """
    
    def __init__(
        self,
        min_turn_gap_ms: int = 200,
        safe_interrupt_gap_ms: int = 500,
        max_wait_ms: int = 2000,
    ) -> None:
        self._min_turn_gap_ms = min_turn_gap_ms
        self._safe_interrupt_gap_ms = safe_interrupt_gap_ms
        self._max_wait_ms = max_wait_ms
        
        self._turn_start_ms: int | None = None
        self._last_speech_ms: int = 0
        self._silence_segments: deque[int] = deque(maxlen=10)
        self._speech_segments: deque[int] = deque(maxlen=10)
        
        self._current_state = TurnState.USER_SPEAKING
        self._state_start_ms: int = 0
    
    @property
    def name(self) -> str:
        return "turn_taking"
    
    def predict(self, context: PredictionContext, state: PipelineState) -> None:
        timestamp = context.frame.timestamp_ms
        vad = context.analysis_results.get("vad")
        prosody = context.analysis_results.get("prosody")
        
        is_speech = vad.data.get("is_speech", False) if vad else False
        
        self._update_segments(timestamp, is_speech)
        
        new_state = self._determine_turn_state(
            timestamp, is_speech, prosody, state
        )
        
        if new_state != self._current_state:
            self._state_start_ms = timestamp
            self._current_state = new_state
        
        signal = self._generate_signal(timestamp, prosody, state)
        
        state.analysis_results["turn_taking"] = type(
            "TurnTakingResult", (), {"data": signal.to_dict()}
        )()
    
    def _update_segments(self, timestamp: int, is_speech: bool) -> None:
        """Track speech and silence segment durations."""
        if is_speech:
            if self._turn_start_ms is None:
                self._turn_start_ms = timestamp
            
            if self._last_speech_ms > 0:
                silence_duration = timestamp - self._last_speech_ms
                if silence_duration > 50:
                    self._silence_segments.append(silence_duration)
            
            self._last_speech_ms = timestamp
        else:
            if self._last_speech_ms > 0:
                speech_duration = self._last_speech_ms - (
                    self._turn_start_ms or self._last_speech_ms
                )
                if speech_duration > 0:
                    self._speech_segments.append(min(speech_duration, 5000))
    
    def _determine_turn_state(
        self,
        timestamp: int,
        is_speech: bool,
        prosody,
        state: PipelineState,
    ) -> TurnState:
        """Determine current turn-taking state."""
        
        if is_speech:
            return TurnState.USER_SPEAKING
        
        silence_duration = timestamp - self._last_speech_ms if self._last_speech_ms > 0 else 0
        
        if silence_duration < self._min_turn_gap_ms:
            return TurnState.USER_SPEAKING
        
        if silence_duration < self._safe_interrupt_gap_ms:
            return TurnState.USER_PAUSING
        
        is_falling = prosody.data.get("is_falling_intonation", False) if prosody else False
        
        if is_falling or silence_duration > self._safe_interrupt_gap_ms:
            return TurnState.TURN_YIELDED
        
        if silence_duration > self._max_wait_ms:
            return TurnState.SYSTEM_CAN_SPEAK
        
        return TurnState.USER_PAUSING
    
    def _generate_signal(
        self,
        timestamp: int,
        prosody,
        state: PipelineState,
    ) -> TurnTakingSignal:
        """Generate detailed turn-taking signal."""
        
        silence_duration = timestamp - self._last_speech_ms if self._last_speech_ms > 0 else 0
        turn_duration = timestamp - self._turn_start_ms if self._turn_start_ms else 0
        
        is_rising = prosody.data.get("is_rising_intonation", False) if prosody else False
        is_falling = prosody.data.get("is_falling_intonation", False) if prosody else False
        
        can_interrupt, reason = self._evaluate_interrupt(
            silence_duration, is_rising, is_falling, state
        )
        
        should_wait = self._should_wait(
            silence_duration, is_rising, state
        )
        
        overlap_prob = self._calculate_overlap_probability(
            silence_duration, is_rising, state
        )
        
        suggested_wait = self._calculate_suggested_wait(
            silence_duration, is_falling, state
        )
        
        confidence = self._calculate_confidence(
            silence_duration, is_falling
        )
        
        silence_in_turn = sum(self._silence_segments)
        
        return TurnTakingSignal(
            state=self._current_state,
            can_interrupt=can_interrupt,
            should_wait=should_wait,
            interrupt_reason=reason,
            confidence=confidence,
            suggested_wait_ms=suggested_wait,
            overlap_probability=overlap_prob,
            turn_duration_ms=turn_duration,
            silence_in_turn_ms=silence_in_turn,
        )
    
    def _evaluate_interrupt(
        self,
        silence_duration: int,
        is_rising: bool,
        is_falling: bool,
        state: PipelineState,
    ) -> tuple[bool, InterruptReason]:
        """Evaluate if interruption is possible and why."""
        
        if silence_duration < self._min_turn_gap_ms:
            return False, InterruptReason.SPEAKING
        
        if is_rising:
            return False, InterruptReason.RISING_INTONATION
        
        if silence_duration < self._safe_interrupt_gap_ms and not is_falling:
            return False, InterruptReason.SHORT_PAUSE
        
        if is_falling and silence_duration >= self._min_turn_gap_ms:
            return True, InterruptReason.FALLING_COMPLETE
        
        if silence_duration >= self._safe_interrupt_gap_ms:
            return True, InterruptReason.LONG_SILENCE
        
        if state.intent_confidence > 0.7:
            return True, InterruptReason.HIGH_CONFIDENCE
        
        return False, InterruptReason.SHORT_PAUSE
    
    def _should_wait(
        self,
        silence_duration: int,
        is_rising: bool,
        state: PipelineState,
    ) -> bool:
        """Strategic recommendation to wait."""
        
        if is_rising:
            return True
        
        if state.timing.speech_likelihood > 0.6:
            return True
        
        if silence_duration < self._safe_interrupt_gap_ms:
            return True
        
        return False
    
    def _calculate_overlap_probability(
        self,
        silence_duration: int,
        is_rising: bool,
        state: PipelineState,
    ) -> float:
        """Estimate probability of overlap if system speaks now."""
        
        if silence_duration > self._max_wait_ms:
            return 0.0
        
        base_prob = max(0.0, 1.0 - (silence_duration / self._safe_interrupt_gap_ms))
        
        if is_rising:
            base_prob = min(1.0, base_prob + 0.3)
        
        speech_factor = state.timing.speech_likelihood * 0.3
        
        return min(1.0, base_prob + speech_factor)
    
    def _calculate_suggested_wait(
        self,
        silence_duration: int,
        is_falling: bool,
        state: PipelineState,
    ) -> int:
        """Calculate how long to wait before responding."""
        
        if silence_duration >= self._safe_interrupt_gap_ms and is_falling:
            return 0
        
        if silence_duration >= self._max_wait_ms:
            return 0
        
        remaining = self._safe_interrupt_gap_ms - silence_duration
        
        if is_falling:
            remaining = max(0, remaining - 200)
        
        if state.intent_confidence > 0.8:
            remaining = max(0, remaining - 100)
        
        return max(0, remaining)
    
    def _calculate_confidence(
        self,
        silence_duration: int,
        is_falling: bool,
    ) -> float:
        """Confidence in the turn-taking assessment."""
        
        base = min(1.0, silence_duration / self._safe_interrupt_gap_ms)
        
        if is_falling:
            base = min(1.0, base + 0.2)
        
        return base
    
    def reset(self) -> None:
        self._turn_start_ms = None
        self._last_speech_ms = 0
        self._silence_segments.clear()
        self._speech_segments.clear()
        self._current_state = TurnState.USER_SPEAKING
        self._state_start_ms = 0
