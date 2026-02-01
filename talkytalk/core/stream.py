"""
Audio stream abstractions.

Frame-based processing: 20-40ms chunks.
No dependency on specific audio libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio stream configuration."""
    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 20
    dtype: str = "float32"
    
    @property
    def frame_size(self) -> int:
        """Samples per frame."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)
    
    @property
    def bytes_per_frame(self) -> int:
        """Bytes per frame (assuming 16-bit PCM for raw)."""
        return self.frame_size * self.channels * 2


@dataclass(slots=True)
class AudioFrame:
    """
    Single audio frame for processing.
    
    Attributes:
        data: Audio samples as float32 numpy array, normalized to [-1.0, 1.0]
        frame_id: Monotonically increasing frame identifier
        timestamp_ms: Timestamp in milliseconds from stream start
        config: Audio configuration
    """
    data: NDArray[np.float32]
    frame_id: int
    timestamp_ms: int
    config: AudioConfig
    
    @property
    def duration_ms(self) -> int:
        """Frame duration in milliseconds."""
        return self.config.frame_duration_ms
    
    @property
    def rms(self) -> float:
        """Root mean square energy."""
        return float(np.sqrt(np.mean(self.data ** 2)))
    
    @property
    def peak(self) -> float:
        """Peak absolute amplitude."""
        return float(np.max(np.abs(self.data)))
    
    @property
    def is_silent(self, threshold: float = 0.01) -> bool:
        """Quick silence check based on RMS."""
        return self.rms < threshold
    
    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        frame_id: int,
        timestamp_ms: int,
        config: AudioConfig,
    ) -> AudioFrame:
        """Create frame from raw PCM16 bytes."""
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return cls(data=samples, frame_id=frame_id, timestamp_ms=timestamp_ms, config=config)
    
    @classmethod
    def silence(cls, frame_id: int, timestamp_ms: int, config: AudioConfig) -> AudioFrame:
        """Create a silent frame."""
        return cls(
            data=np.zeros(config.frame_size, dtype=np.float32),
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            config=config,
        )


@runtime_checkable
class AudioSource(Protocol):
    """Protocol for audio sources."""
    
    @property
    def config(self) -> AudioConfig:
        """Return audio configuration."""
        ...
    
    def frames(self) -> Iterator[AudioFrame]:
        """Yield audio frames."""
        ...
    
    def close(self) -> None:
        """Close the source."""
        ...


class FrameBuffer:
    """
    Sliding window buffer for audio frames.
    
    Maintains a fixed-size window of recent frames for
    analysis that requires temporal context.
    """
    
    def __init__(self, max_frames: int = 50, max_duration_ms: int = 1000) -> None:
        self._frames: list[AudioFrame] = []
        self._max_frames = max_frames
        self._max_duration_ms = max_duration_ms
    
    def push(self, frame: AudioFrame) -> None:
        """Add frame to buffer, evicting old frames if necessary."""
        self._frames.append(frame)
        self._evict()
    
    def _evict(self) -> None:
        """Remove frames that exceed limits."""
        while len(self._frames) > self._max_frames:
            self._frames.pop(0)
        
        if len(self._frames) > 1:
            latest = self._frames[-1].timestamp_ms
            while self._frames and (latest - self._frames[0].timestamp_ms) > self._max_duration_ms:
                self._frames.pop(0)
    
    @property
    def frames(self) -> list[AudioFrame]:
        """Get all buffered frames."""
        return list(self._frames)
    
    @property
    def duration_ms(self) -> int:
        """Total buffered duration in ms."""
        if len(self._frames) < 2:
            return 0
        return self._frames[-1].timestamp_ms - self._frames[0].timestamp_ms
    
    def concatenate(self) -> NDArray[np.float32]:
        """Concatenate all frame data."""
        if not self._frames:
            return np.array([], dtype=np.float32)
        return np.concatenate([f.data for f in self._frames])
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._frames.clear()
