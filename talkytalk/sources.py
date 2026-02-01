"""
Audio source implementations.

Provides concrete AudioSource implementations for common use cases.
"""

from __future__ import annotations

from typing import Iterator
import numpy as np

from talkytalk.core.stream import AudioFrame, AudioConfig, AudioSource


class ArraySource(AudioSource):
    """
    Audio source from numpy array.
    
    Useful for testing and batch processing.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        config: AudioConfig | None = None,
    ) -> None:
        self._config = config or AudioConfig()
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if data.max() > 1.0 or data.min() < -1.0:
            max_val = max(abs(data.max()), abs(data.min()))
            if max_val > 0:
                data = data / max_val
        
        self._data = data
        self._position = 0
        self._frame_id = 0
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        frame_size = self._config.frame_size
        
        while self._position + frame_size <= len(self._data):
            frame_data = self._data[self._position:self._position + frame_size]
            timestamp_ms = int(self._frame_id * self._config.frame_duration_ms)
            
            yield AudioFrame(
                data=frame_data,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
                config=self._config,
            )
            
            self._position += frame_size
            self._frame_id += 1
    
    def close(self) -> None:
        pass
    
    def reset(self) -> None:
        """Reset to beginning."""
        self._position = 0
        self._frame_id = 0


class SineSource(AudioSource):
    """
    Generate sine wave audio for testing.
    """
    
    def __init__(
        self,
        frequency_hz: float = 440.0,
        duration_ms: int = 1000,
        amplitude: float = 0.5,
        config: AudioConfig | None = None,
    ) -> None:
        self._config = config or AudioConfig()
        self._frequency = frequency_hz
        self._duration_ms = duration_ms
        self._amplitude = amplitude
        
        total_samples = int(self._config.sample_rate * duration_ms / 1000)
        t = np.arange(total_samples) / self._config.sample_rate
        self._data = (self._amplitude * np.sin(2 * np.pi * self._frequency * t)).astype(np.float32)
        
        self._source = ArraySource(self._data, self._config)
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        return self._source.frames()
    
    def close(self) -> None:
        self._source.close()


class NoiseSource(AudioSource):
    """
    Generate white noise for testing.
    """
    
    def __init__(
        self,
        duration_ms: int = 1000,
        amplitude: float = 0.1,
        config: AudioConfig | None = None,
    ) -> None:
        self._config = config or AudioConfig()
        self._duration_ms = duration_ms
        self._amplitude = amplitude
        
        total_samples = int(self._config.sample_rate * duration_ms / 1000)
        self._data = (self._amplitude * np.random.randn(total_samples)).astype(np.float32)
        
        self._source = ArraySource(self._data, self._config)
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        return self._source.frames()
    
    def close(self) -> None:
        self._source.close()


class SilenceSource(AudioSource):
    """
    Generate silence for testing.
    """
    
    def __init__(
        self,
        duration_ms: int = 1000,
        config: AudioConfig | None = None,
    ) -> None:
        self._config = config or AudioConfig()
        self._duration_ms = duration_ms
        
        total_samples = int(self._config.sample_rate * duration_ms / 1000)
        self._data = np.zeros(total_samples, dtype=np.float32)
        
        self._source = ArraySource(self._data, self._config)
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def frames(self) -> Iterator[AudioFrame]:
        return self._source.frames()
    
    def close(self) -> None:
        self._source.close()
