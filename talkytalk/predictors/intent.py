"""
Early intent predictor.

Predicts user intent before sentence completion using
prosodic and behavioral cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from talkytalk.core.packet import Intent
from talkytalk.predictors.base import Predictor, PredictionContext

if TYPE_CHECKING:
    from talkytalk.core.pipeline import PipelineState


@dataclass
class IntentSignal:
    """Internal signal for intent tracking."""
    intent: Intent
    weight: float
    decay: float = 0.95


class IntentPredictor(Predictor):
    """
    Early intent prediction from behavioral cues.
    
    Heuristics (to be refined with data):
    
    1. **Command pattern**: 
       - High arousal + falling intonation
       - Fast speech onset
       - Short pause before speech
       
    2. **Query pattern**:
       - Rising intonation
       - Moderate arousal
       - Hesitation markers (pauses within speech)
       
    3. **Conversation pattern**:
       - Neutral arousal
       - Mixed intonation
       - Longer utterances
       
    4. **Music/Media pattern**:
       - Often starts with specific keywords (requires ASR)
       - For now: placeholder based on tempo detection
       
    5. **Translation pattern**:
       - Language switch detected
       - Specific prosodic markers
    
    All predictions are probabilistic and updated incrementally.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        decay_rate: float = 0.95,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._decay_rate = decay_rate
        
        self._intent_scores: dict[Intent, float] = {
            intent: 0.0 for intent in Intent
        }
        self._frame_count = 0
    
    @property
    def name(self) -> str:
        return "intent"
    
    def predict(self, context: PredictionContext, state: PipelineState) -> None:
        self._frame_count += 1
        
        self._decay_scores()
        
        self._apply_heuristics(context, state)
        
        best_intent, confidence = self._get_best_intent()
        
        state.current_intent = best_intent
        state.intent_confidence = confidence
        
        language_result = context.analysis_results.get("language")
        if language_result:
            lang = language_result.data.get("language", "unknown")
            state.language = lang
        
        emotion_result = context.analysis_results.get("emotion")
        if emotion_result:
            from talkytalk.core.packet import Emotion
            state.emotion = Emotion(
                arousal=emotion_result.data.get("arousal", 0.5),
                valence=emotion_result.data.get("valence", 0.5),
            )
    
    def _decay_scores(self) -> None:
        """Apply temporal decay to all scores."""
        for intent in self._intent_scores:
            self._intent_scores[intent] *= self._decay_rate
    
    def _apply_heuristics(self, context: PredictionContext, state: PipelineState) -> None:
        """Apply rule-based heuristics to update intent scores."""
        
        vad = context.analysis_results.get("vad")
        prosody = context.analysis_results.get("prosody")
        emotion = context.analysis_results.get("emotion")
        
        is_speech = False
        if vad:
            is_speech = vad.data.get("is_speech", False)
        
        if not is_speech:
            return
        
        arousal = 0.5
        if emotion:
            arousal = emotion.data.get("arousal", 0.5)
        
        is_rising = False
        is_falling = False
        tempo = 0.0
        pause_duration = 0
        
        if prosody:
            is_rising = prosody.data.get("is_rising_intonation", False)
            is_falling = prosody.data.get("is_falling_intonation", False)
            tempo = prosody.data.get("tempo", 0.0)
            pause_duration = prosody.data.get("pause_duration_ms", 0)
        
        if arousal > 0.7 and is_falling and tempo > 4.0:
            self._intent_scores[Intent.COMMAND] += 0.15
        
        if is_rising:
            self._intent_scores[Intent.QUERY] += 0.12
        
        if 0.3 < arousal < 0.7 and not is_rising and not is_falling:
            self._intent_scores[Intent.CONVERSATION] += 0.08
        
        if pause_duration > 300 and pause_duration < 1000:
            self._intent_scores[Intent.QUERY] += 0.05
        
        self._intent_scores[Intent.UNKNOWN] += 0.02
    
    def _get_best_intent(self) -> tuple[Intent, float]:
        """Get the intent with highest score."""
        total = sum(self._intent_scores.values())
        
        if total < 0.01:
            return Intent.UNKNOWN, 0.0
        
        best_intent = max(self._intent_scores, key=lambda i: self._intent_scores[i])
        confidence = self._intent_scores[best_intent] / total
        
        raw_confidence = min(1.0, self._intent_scores[best_intent])
        
        if confidence < 0.4:
            return Intent.UNKNOWN, raw_confidence * 0.5
        
        return best_intent, confidence
    
    def reset(self) -> None:
        self._intent_scores = {intent: 0.0 for intent in Intent}
        self._frame_count = 0
