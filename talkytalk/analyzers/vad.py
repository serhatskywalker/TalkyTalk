"""
Voice Activity Detection analyzer.

Detects speech vs silence using energy-based heuristics.
Can be upgraded to use WebRTC VAD or neural models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

from talkytalk.analyzers.base import Analyzer, AnalysisResult

if TYPE_CHECKING:
    from talkytalk.core.stream import AudioFrame, FrameBuffer
    from talkytalk.core.pipeline import PipelineState


@dataclass
class VADResult(AnalysisResult):
    """VAD-specific result."""
    is_speech: bool = False
    speech_probability: float = 0.0
    energy_db: float = -100.0
    zero_crossing_rate: float = 0.0


class VADAnalyzer(Analyzer):
    """
    Energy-based Voice Activity Detection.
    
    Heuristics:
    - RMS energy threshold (adaptive)
    - Zero-crossing rate (speech vs noise discrimination)
    - Temporal smoothing (hangover to prevent choppy detection)
    
    Parameters:
        energy_threshold_db: Minimum energy for speech (default -40 dB)
        hangover_frames: Frames to keep speech active after drop (default 5)
        adaptive: Whether to adapt threshold based on noise floor
    """
    
    def __init__(
        self,
        energy_threshold_db: float = -40.0,
        hangover_frames: int = 5,
        adaptive: bool = True,
    ) -> None:
        self._energy_threshold_db = energy_threshold_db
        self._hangover_frames = hangover_frames
        self._adaptive = adaptive
        
        self._noise_floor_db: float = -60.0
        self._hangover_counter: int = 0
        self._speech_active: bool = False
        self._frame_count: int = 0
    
    @property
    def name(self) -> str:
        return "vad"
    
    def analyze(
        self,
        frame: AudioFrame,
        buffer: FrameBuffer,
        state: PipelineState,
    ) -> VADResult:
        self._frame_count += 1
        
        rms = frame.rms
        energy_db = 20 * np.log10(rms + 1e-10)
        
        zcr = self._zero_crossing_rate(frame.data)
        
        if self._adaptive:
            self._update_noise_floor(energy_db)
        
        threshold = max(
            self._energy_threshold_db,
            self._noise_floor_db + 10
        )
        
        raw_speech = energy_db > threshold and zcr < 0.5
        
        if raw_speech:
            self._hangover_counter = self._hangover_frames
            self._speech_active = True
        elif self._hangover_counter > 0:
            self._hangover_counter -= 1
        else:
            self._speech_active = False
        
        speech_prob = self._calculate_speech_probability(energy_db, zcr, threshold)
        
        return VADResult(
            analyzer_name=self.name,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            is_speech=self._speech_active,
            speech_probability=speech_prob,
            energy_db=energy_db,
            zero_crossing_rate=zcr,
            data={
                "is_speech": self._speech_active,
                "speech_probability": speech_prob,
                "energy_db": energy_db,
                "threshold_db": threshold,
                "noise_floor_db": self._noise_floor_db,
            },
        )
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """Calculate zero-crossing rate."""
        if len(data) < 2:
            return 0.0
        signs = np.sign(data)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings / len(data))
    
    def _update_noise_floor(self, energy_db: float) -> None:
        """Slowly adapt noise floor estimate."""
        alpha = 0.01 if energy_db < self._noise_floor_db + 5 else 0.001
        self._noise_floor_db = (1 - alpha) * self._noise_floor_db + alpha * energy_db
    
    def _calculate_speech_probability(
        self,
        energy_db: float,
        zcr: float,
        threshold: float,
    ) -> float:
        """Estimate speech probability as continuous value."""
        if energy_db < threshold - 20:
            return 0.0
        if energy_db > threshold + 10:
            energy_prob = 1.0
        else:
            energy_prob = (energy_db - (threshold - 20)) / 30
        
        zcr_prob = max(0.0, 1.0 - zcr * 2)
        
        return min(1.0, energy_prob * 0.7 + zcr_prob * 0.3)
    
    def reset(self) -> None:
        self._noise_floor_db = -60.0
        self._hangover_counter = 0
        self._speech_active = False
        self._frame_count = 0
