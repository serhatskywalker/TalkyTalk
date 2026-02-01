"""
Timing predictor.

Determines when system intervention is appropriate.
Handles pause detection and turn-taking signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from talkytalk.core.packet import Timing
from talkytalk.predictors.base import Predictor, PredictionContext

if TYPE_CHECKING:
    from talkytalk.core.pipeline import PipelineState


class TimingPredictor(Predictor):
    """
    Turn-taking and timing prediction.
    
    Key outputs:
    - user_paused: Is the user in a meaningful pause?
    - interrupt_safe: Can the system safely respond now?
    - speech_likelihood: Probability user will continue speaking
    
    Heuristics:
    
    1. **Pause detection**:
       - Silence > 300ms after speech = pause
       - Silence > 700ms = likely turn end
       - But: hesitation pauses are different from completion pauses
       
    2. **Interrupt safety**:
       - Safe after falling intonation + pause
       - Safe after high-confidence intent + pause
       - Unsafe during rising intonation (question forming)
       - Unsafe during high speech likelihood
       
    3. **Speech likelihood**:
       - Decays during silence
       - Resets on speech detection
       - Affected by intonation patterns
    """
    
    def __init__(
        self,
        pause_threshold_ms: int = 300,
        turn_end_threshold_ms: int = 700,
        interrupt_confidence: float = 0.6,
    ) -> None:
        self._pause_threshold_ms = pause_threshold_ms
        self._turn_end_threshold_ms = turn_end_threshold_ms
        self._interrupt_confidence = interrupt_confidence
        
        self._silence_start_ms: int | None = None
        self._last_speech_ms: int = 0
        self._speech_likelihood: float = 0.0
    
    @property
    def name(self) -> str:
        return "timing"
    
    def predict(self, context: PredictionContext, state: PipelineState) -> None:
        vad = context.analysis_results.get("vad")
        prosody = context.analysis_results.get("prosody")
        
        is_speech = False
        if vad:
            is_speech = vad.data.get("is_speech", False)
        
        timestamp = context.frame.timestamp_ms
        
        if is_speech:
            self._last_speech_ms = timestamp
            self._silence_start_ms = None
            self._speech_likelihood = 1.0
            state.speech_active = True
        else:
            if self._silence_start_ms is None:
                self._silence_start_ms = timestamp
            self._update_speech_likelihood(timestamp, prosody)
        
        silence_duration = 0
        if self._silence_start_ms is not None:
            silence_duration = timestamp - self._silence_start_ms
        
        user_paused = silence_duration >= self._pause_threshold_ms
        
        is_falling = False
        is_rising = False
        if prosody:
            is_falling = prosody.data.get("is_falling_intonation", False)
            is_rising = prosody.data.get("is_rising_intonation", False)
        
        interrupt_safe = self._calculate_interrupt_safety(
            user_paused=user_paused,
            silence_duration=silence_duration,
            is_falling=is_falling,
            is_rising=is_rising,
            intent_confidence=state.intent_confidence,
        )
        
        state.timing = Timing(
            user_paused=user_paused,
            interrupt_safe=interrupt_safe,
            speech_likelihood=self._speech_likelihood,
            silence_duration_ms=silence_duration,
        )
    
    def _update_speech_likelihood(self, timestamp: int, prosody) -> None:
        """Update probability that user will continue speaking."""
        if self._silence_start_ms is None:
            return
        
        silence_duration = timestamp - self._silence_start_ms
        
        if silence_duration < 200:
            decay = 0.95
        elif silence_duration < 500:
            decay = 0.85
        else:
            decay = 0.7
        
        self._speech_likelihood *= decay
        
        if prosody:
            is_rising = prosody.data.get("is_rising_intonation", False)
            if is_rising:
                self._speech_likelihood = min(1.0, self._speech_likelihood + 0.1)
    
    def _calculate_interrupt_safety(
        self,
        user_paused: bool,
        silence_duration: int,
        is_falling: bool,
        is_rising: bool,
        intent_confidence: float,
    ) -> bool:
        """Determine if system can safely interrupt/respond."""
        
        if is_rising:
            return False
        
        if self._speech_likelihood > 0.7:
            return False
        
        if not user_paused:
            return False
        
        if silence_duration >= self._turn_end_threshold_ms:
            return True
        
        if is_falling and silence_duration >= self._pause_threshold_ms:
            return True
        
        if intent_confidence >= self._interrupt_confidence and user_paused:
            return True
        
        return False
    
    def reset(self) -> None:
        self._silence_start_ms = None
        self._last_speech_ms = 0
        self._speech_likelihood = 0.0
