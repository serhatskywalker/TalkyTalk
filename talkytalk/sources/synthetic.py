"""Synthetic audio sources for testing."""

from __future__ import annotations

from typing import Iterator
import numpy as np

from talkytalk.core.stream import AudioConfig, AudioFrame, AudioSource


class ArraySource(AudioSource):
    """Audio source from numpy array."""
    
    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
    ) -> None:
        self._config = AudioConfig(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
        )
        self._data = data.astype(np.float32)
        if self._data.max() > 1.0 or self._data.min() < -1.0:
            self._data = self._data / max(abs(self._data.max()), abs(self._data.min()))
        self._position = 0
        self._frame_id = 0
        self._closed = False
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        samples_per_frame = self._config.samples_per_frame
        
        while self._position + samples_per_frame <= len(self._data) and not self._closed:
            frame_data = self._data[self._position:self._position + samples_per_frame]
            timestamp_ms = int(self._position / self._config.sample_rate * 1000)
            
            yield AudioFrame(
                data=frame_data,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
                config=self._config,
            )
            
            self._position += samples_per_frame
            self._frame_id += 1
    
    def close(self) -> None:
        self._closed = True


class SineSource(AudioSource):
    """Generate sine wave audio."""
    
    def __init__(
        self,
        frequency_hz: float = 440.0,
        amplitude: float = 0.5,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
    ) -> None:
        self._config = AudioConfig(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
        )
        
        total_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, total_samples, dtype=np.float32)
        self._data = (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
        
        self._position = 0
        self._frame_id = 0
        self._closed = False
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        samples_per_frame = self._config.samples_per_frame
        
        while self._position + samples_per_frame <= len(self._data) and not self._closed:
            frame_data = self._data[self._position:self._position + samples_per_frame]
            timestamp_ms = int(self._position / self._config.sample_rate * 1000)
            
            yield AudioFrame(
                data=frame_data,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
                config=self._config,
            )
            
            self._position += samples_per_frame
            self._frame_id += 1
    
    def close(self) -> None:
        self._closed = True


class NoiseSource(AudioSource):
    """Generate white noise audio."""
    
    def __init__(
        self,
        amplitude: float = 0.1,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        seed: int | None = None,
    ) -> None:
        self._config = AudioConfig(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
        )
        
        rng = np.random.default_rng(seed)
        total_samples = int(sample_rate * duration_ms / 1000)
        self._data = (amplitude * rng.standard_normal(total_samples)).astype(np.float32)
        
        self._position = 0
        self._frame_id = 0
        self._closed = False
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        samples_per_frame = self._config.samples_per_frame
        
        while self._position + samples_per_frame <= len(self._data) and not self._closed:
            frame_data = self._data[self._position:self._position + samples_per_frame]
            timestamp_ms = int(self._position / self._config.sample_rate * 1000)
            
            yield AudioFrame(
                data=frame_data,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
                config=self._config,
            )
            
            self._position += samples_per_frame
            self._frame_id += 1
    
    def close(self) -> None:
        self._closed = True


class SilenceSource(AudioSource):
    """Generate silence."""
    
    def __init__(
        self,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
    ) -> None:
        self._config = AudioConfig(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
        )
        
        total_samples = int(sample_rate * duration_ms / 1000)
        self._data = np.zeros(total_samples, dtype=np.float32)
        
        self._position = 0
        self._frame_id = 0
        self._closed = False
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        samples_per_frame = self._config.samples_per_frame
        
        while self._position + samples_per_frame <= len(self._data) and not self._closed:
            frame_data = self._data[self._position:self._position + samples_per_frame]
            timestamp_ms = int(self._position / self._config.sample_rate * 1000)
            
            yield AudioFrame(
                data=frame_data,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
                config=self._config,
            )
            
            self._position += samples_per_frame
            self._frame_id += 1
    
    def close(self) -> None:
        self._closed = True
