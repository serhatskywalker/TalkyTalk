"""
Emotion analyzer.

Extracts arousal/valence from prosodic and acoustic features.
Based on dimensional emotion model (Russell's circumplex).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from talkytalk.analyzers.base import Analyzer, AnalysisResult
from talkytalk.core.packet import Emotion

if TYPE_CHECKING:
    from talkytalk.core.stream import AudioFrame, FrameBuffer
    from talkytalk.core.pipeline import PipelineState


class EmotionAnalyzer(Analyzer):
    """
    Dimensional emotion estimator.
    
    Maps acoustic features to arousal-valence space:
    
    Arousal correlates:
    - Energy (louder = higher arousal)
    - Speech rate (faster = higher arousal)
    - Pitch variance (more varied = higher arousal)
    
    Valence correlates (weaker, more context-dependent):
    - Pitch height (higher = more positive, generally)
    - Intonation patterns
    - Speech fluency
    
    Note: Valence is notoriously difficult to estimate from
    acoustics alone. This provides a rough signal only.
    """
    
    def __init__(
        self,
        smoothing_alpha: float = 0.3,
    ) -> None:
        self._smoothing_alpha = smoothing_alpha
        self._current_arousal: float = 0.5
        self._current_valence: float = 0.5
        
        self._energy_baseline: float = -40.0
        self._pitch_baseline: float = 150.0
    
    @property
    def name(self) -> str:
        return "emotion"
    
    def analyze(
        self,
        frame: AudioFrame,
        buffer: FrameBuffer,
        state: PipelineState,
    ) -> AnalysisResult:
        vad_result = state.analysis_results.get("vad")
        prosody_result = state.analysis_results.get("prosody")
        
        energy_db = -60.0
        if vad_result:
            energy_db = vad_result.data.get("energy_db", -60.0)
        
        pitch_hz = 0.0
        pitch_variance = 0.0
        tempo = 0.0
        is_rising = False
        
        if prosody_result:
            pitch_hz = prosody_result.data.get("pitch_hz", 0.0)
            pitch_variance = prosody_result.data.get("pitch_variance", 0.0)
            tempo = prosody_result.data.get("tempo", 0.0)
            is_rising = prosody_result.data.get("is_rising_intonation", False)
        
        raw_arousal = self._estimate_arousal(energy_db, pitch_variance, tempo)
        raw_valence = self._estimate_valence(pitch_hz, is_rising)
        
        self._current_arousal = self._smooth(self._current_arousal, raw_arousal)
        self._current_valence = self._smooth(self._current_valence, raw_valence)
        
        emotion = Emotion(
            arousal=self._current_arousal,
            valence=self._current_valence,
        )
        
        return AnalysisResult(
            analyzer_name=self.name,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            data={
                "arousal": self._current_arousal,
                "valence": self._current_valence,
                "quadrant": emotion.quadrant,
                "raw_arousal": raw_arousal,
                "raw_valence": raw_valence,
            },
        )
    
    def _estimate_arousal(
        self,
        energy_db: float,
        pitch_variance: float,
        tempo: float,
    ) -> float:
        """
        Estimate arousal from acoustic features.
        
        Higher energy, faster tempo, more pitch variation → higher arousal
        """
        energy_contrib = (energy_db - self._energy_baseline + 30) / 60
        energy_contrib = max(0.0, min(1.0, energy_contrib))
        
        variance_contrib = min(1.0, pitch_variance / 1000)
        
        tempo_contrib = min(1.0, tempo / 8.0)
        
        arousal = (
            energy_contrib * 0.5 +
            variance_contrib * 0.3 +
            tempo_contrib * 0.2
        )
        
        return max(0.0, min(1.0, arousal))
    
    def _estimate_valence(
        self,
        pitch_hz: float,
        is_rising: bool,
    ) -> float:
        """
        Estimate valence from acoustic features.
        
        Note: This is a weak signal. Valence is better inferred
        from semantic content and context.
        """
        if pitch_hz <= 0:
            return 0.5
        
        pitch_contrib = (pitch_hz - self._pitch_baseline) / 200
        pitch_contrib = max(-0.3, min(0.3, pitch_contrib))
        
        rising_contrib = 0.1 if is_rising else 0.0
        
        valence = 0.5 + pitch_contrib + rising_contrib
        
        return max(0.0, min(1.0, valence))
    
    def _smooth(self, current: float, new: float) -> float:
        """Exponential smoothing."""
        return current * (1 - self._smoothing_alpha) + new * self._smoothing_alpha
    
    def reset(self) -> None:
        self._current_arousal = 0.5
        self._current_valence = 0.5
