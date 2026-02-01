"""
Prosody analyzer.

Extracts rhythm, tempo, pitch contour, and pause patterns.
These are behavioral signals that precede semantic content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from talkytalk.analyzers.base import Analyzer, AnalysisResult

if TYPE_CHECKING:
    from talkytalk.core.stream import AudioFrame, FrameBuffer
    from talkytalk.core.pipeline import PipelineState


@dataclass
class ProsodyResult(AnalysisResult):
    """Prosody analysis result."""
    pitch_hz: float = 0.0
    pitch_variance: float = 0.0
    tempo: float = 0.0
    pause_duration_ms: int = 0
    is_rising_intonation: bool = False
    is_falling_intonation: bool = False
    speech_rate: float = 0.0


class ProsodyAnalyzer(Analyzer):
    """
    Prosodic feature extractor.
    
    Extracts:
    - Fundamental frequency (F0) via autocorrelation
    - Pitch contour direction (rising/falling)
    - Speaking rate estimation
    - Pause detection and duration
    
    These features indicate:
    - Questions (rising intonation)
    - Commands (falling intonation, faster rate)
    - Hesitation (pauses, pitch variance)
    - Emotional state (tempo, pitch range)
    """
    
    def __init__(
        self,
        min_pitch_hz: float = 50.0,
        max_pitch_hz: float = 500.0,
        pause_threshold_ms: int = 200,
    ) -> None:
        self._min_pitch_hz = min_pitch_hz
        self._max_pitch_hz = max_pitch_hz
        self._pause_threshold_ms = pause_threshold_ms
        
        self._pitch_history: list[float] = []
        self._last_speech_timestamp: int = 0
        self._current_pause_start: int | None = None
    
    @property
    def name(self) -> str:
        return "prosody"
    
    def analyze(
        self,
        frame: AudioFrame,
        buffer: FrameBuffer,
        state: PipelineState,
    ) -> ProsodyResult:
        vad_result = state.analysis_results.get("vad")
        is_speech = vad_result.data.get("is_speech", True) if vad_result else True
        
        pitch_hz = 0.0
        if is_speech and frame.rms > 0.01:
            pitch_hz = self._estimate_pitch(frame.data, frame.config.sample_rate)
        
        if pitch_hz > 0:
            self._pitch_history.append(pitch_hz)
            if len(self._pitch_history) > 25:
                self._pitch_history.pop(0)
        
        pitch_variance = np.var(self._pitch_history) if len(self._pitch_history) > 2 else 0.0
        
        is_rising, is_falling = self._detect_intonation()
        
        pause_duration_ms = self._track_pauses(frame.timestamp_ms, is_speech)
        
        tempo = self._estimate_tempo(buffer)
        
        return ProsodyResult(
            analyzer_name=self.name,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            pitch_hz=pitch_hz,
            pitch_variance=float(pitch_variance),
            tempo=tempo,
            pause_duration_ms=pause_duration_ms,
            is_rising_intonation=is_rising,
            is_falling_intonation=is_falling,
            speech_rate=tempo,
            data={
                "pitch_hz": pitch_hz,
                "pitch_variance": float(pitch_variance),
                "tempo": tempo,
                "pause_duration_ms": pause_duration_ms,
                "is_rising_intonation": is_rising,
                "is_falling_intonation": is_falling,
                "pitch_history_len": len(self._pitch_history),
            },
        )
    
    def _estimate_pitch(self, data: NDArray[np.float32], sample_rate: int) -> float:
        """
        Estimate fundamental frequency using autocorrelation.
        
        This is a simple but effective method for real-time use.
        """
        if len(data) < 100:
            return 0.0
        
        min_lag = int(sample_rate / self._max_pitch_hz)
        max_lag = int(sample_rate / self._min_pitch_hz)
        max_lag = min(max_lag, len(data) - 1)
        
        if min_lag >= max_lag:
            return 0.0
        
        data_normalized = data - np.mean(data)
        
        autocorr = np.correlate(data_normalized, data_normalized, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        if len(autocorr) <= max_lag:
            return 0.0
        
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.0
        
        peak_idx = np.argmax(search_region) + min_lag
        
        if autocorr[0] > 0 and autocorr[peak_idx] / autocorr[0] < 0.3:
            return 0.0
        
        pitch_hz = sample_rate / peak_idx
        return float(pitch_hz)
    
    def _detect_intonation(self) -> tuple[bool, bool]:
        """Detect rising or falling intonation from pitch history."""
        if len(self._pitch_history) < 5:
            return False, False
        
        recent = self._pitch_history[-5:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        threshold = 5.0
        return slope > threshold, slope < -threshold
    
    def _track_pauses(self, timestamp_ms: int, is_speech: bool) -> int:
        """Track pause duration."""
        if is_speech:
            if self._current_pause_start is not None:
                pause_duration = timestamp_ms - self._current_pause_start
                self._current_pause_start = None
                return pause_duration
            self._last_speech_timestamp = timestamp_ms
            return 0
        else:
            if self._current_pause_start is None:
                self._current_pause_start = timestamp_ms
            return timestamp_ms - self._current_pause_start
        
    def _estimate_tempo(self, buffer: FrameBuffer) -> float:
        """
        Estimate speaking tempo from energy envelope.
        
        Returns approximate syllables per second.
        """
        if buffer.duration_ms < 500:
            return 0.0
        
        data = buffer.concatenate()
        if len(data) < 1000:
            return 0.0
        
        window_size = 160
        envelope = np.array([
            np.sqrt(np.mean(data[i:i+window_size] ** 2))
            for i in range(0, len(data) - window_size, window_size // 2)
        ])
        
        if len(envelope) < 10:
            return 0.0
        
        envelope_smooth = np.convolve(envelope, np.ones(3) / 3, mode='valid')
        
        threshold = np.mean(envelope_smooth) * 0.5
        above = envelope_smooth > threshold
        transitions = np.diff(above.astype(int))
        peaks = np.sum(transitions == 1)
        
        duration_sec = buffer.duration_ms / 1000.0
        tempo = peaks / duration_sec if duration_sec > 0 else 0.0
        
        return float(tempo)
    
    def reset(self) -> None:
        self._pitch_history.clear()
        self._last_speech_timestamp = 0
        self._current_pause_start = None
