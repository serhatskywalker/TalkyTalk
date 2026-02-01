"""
Real-time microphone audio source.

Requires: pip install sounddevice
"""

from __future__ import annotations

from typing import Iterator
from collections import deque
from threading import Thread, Event
import numpy as np

from talkytalk.core.stream import AudioConfig, AudioFrame, AudioSource


class MicrophoneSource(AudioSource):
    """
    Real-time microphone input using sounddevice.
    
    Usage:
        source = MicrophoneSource()
        
        for frame in source.frames():
            packet = pipeline.process_frame(frame)
            print(packet.intent)
        
        # Or with timeout
        source = MicrophoneSource(max_duration_s=10.0)
    
    Press Ctrl+C to stop recording.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        channels: int = 1,
        device: int | str | None = None,
        max_duration_s: float | None = None,
        buffer_size: int = 100,
    ) -> None:
        """
        Initialize microphone source.
        
        Args:
            sample_rate: Audio sample rate (default 16kHz)
            frame_duration_ms: Frame duration in ms (default 20ms)
            channels: Number of audio channels (default 1 = mono)
            device: Audio device index or name (None = default)
            max_duration_s: Maximum recording duration (None = unlimited)
            buffer_size: Internal frame buffer size
        """
        self._config = AudioConfig(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            channels=channels,
        )
        self._device = device
        self._max_duration_s = max_duration_s
        self._buffer_size = buffer_size
        
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._frame_id = 0
        self._timestamp_ms = 0
        self._stop_event = Event()
        self._stream = None
        self._thread: Thread | None = None
        self._error: Exception | None = None
    
    @property
    def config(self) -> AudioConfig:
        return self._config
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}")
        
        audio_data = indata[:, 0].copy().astype(np.float32)
        self._frame_buffer.append(audio_data)
    
    def _start_stream(self):
        """Start the audio stream in background."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone input.\n"
                "Install with: pip install sounddevice"
            )
        
        samples_per_frame = self._config.samples_per_frame
        
        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            blocksize=samples_per_frame,
            channels=self._config.channels,
            dtype=np.float32,
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()
    
    def frames(self) -> Iterator[AudioFrame]:
        """
        Yield audio frames from microphone.
        
        This is a blocking generator that yields frames as they arrive.
        Use Ctrl+C or call close() to stop.
        """
        self._start_stream()
        
        start_time_ms = 0
        max_ms = int(self._max_duration_s * 1000) if self._max_duration_s else None
        
        try:
            while not self._stop_event.is_set():
                if self._frame_buffer:
                    frame_data = self._frame_buffer.popleft()
                    
                    frame = AudioFrame(
                        data=frame_data,
                        frame_id=self._frame_id,
                        timestamp_ms=self._timestamp_ms,
                        config=self._config,
                    )
                    
                    yield frame
                    
                    self._frame_id += 1
                    self._timestamp_ms += self._config.frame_duration_ms
                    
                    if max_ms and self._timestamp_ms >= max_ms:
                        break
                else:
                    import time
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
    
    def close(self) -> None:
        """Stop recording and clean up."""
        self._stop_event.set()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def list_audio_devices() -> None:
    """Print available audio devices."""
    try:
        import sounddevice as sd
        print(sd.query_devices())
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")


def get_default_device() -> dict:
    """Get default input device info."""
    try:
        import sounddevice as sd
        return sd.query_devices(kind='input')
    except ImportError:
        return {"error": "sounddevice not installed"}
