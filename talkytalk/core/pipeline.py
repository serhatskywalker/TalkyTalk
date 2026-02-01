"""
Core processing pipeline.

The pipeline orchestrates analyzers and predictors to produce IntentPackets.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, AsyncIterator
import time

from talkytalk.core.packet import IntentPacket, Emotion, Timing, Intent
from talkytalk.core.stream import AudioFrame, AudioConfig, FrameBuffer, AudioSource
from talkytalk.analyzers.base import Analyzer, AnalysisResult
from talkytalk.predictors.base import Predictor, PredictionContext


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    buffer_duration_ms: int = 1000
    emit_interval_ms: int = 100
    min_confidence_to_emit: float = 0.0


class PipelineState:
    """Mutable state shared across pipeline components."""
    
    def __init__(self) -> None:
        self.current_intent: Intent = Intent.UNKNOWN
        self.intent_confidence: float = 0.0
        self.language: str = "unknown"
        self.target_language: str | None = None
        self.emotion: Emotion = Emotion()
        self.timing: Timing = Timing()
        self.last_speech_frame_id: int = 0
        self.speech_active: bool = False
        self.analysis_results: dict[str, AnalysisResult] = {}
    
    def to_packet(self, frame_id: int, timestamp_ms: int) -> IntentPacket:
        """Convert current state to IntentPacket."""
        return IntentPacket(
            intent=self.current_intent,
            confidence=self.intent_confidence,
            language=self.language,
            target_language=self.target_language,
            emotion=self.emotion,
            timing=self.timing,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )


class Pipeline:
    """
    Main processing pipeline.
    
    Processes audio frames through analyzers and predictors,
    emitting IntentPackets at configured intervals.
    
    Usage:
        pipeline = Pipeline(config)
        pipeline.add_analyzer(VADAnalyzer())
        pipeline.add_predictor(IntentPredictor())
        
        async for packet in pipeline.run(audio_source):
            handle_packet(packet)
    """
    
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._analyzers: list[Analyzer] = []
        self._predictors: list[Predictor] = []
        self._buffer = FrameBuffer(max_duration_ms=self._config.buffer_duration_ms)
        self._state = PipelineState()
        self._callbacks: list[Callable[[IntentPacket], None]] = []
        self._running = False
        self._last_emit_ms = 0
    
    @property
    def config(self) -> PipelineConfig:
        return self._config
    
    @property
    def state(self) -> PipelineState:
        return self._state
    
    def add_analyzer(self, analyzer: Analyzer) -> Pipeline:
        """Add an analyzer to the pipeline. Returns self for chaining."""
        self._analyzers.append(analyzer)
        return self
    
    def add_predictor(self, predictor: Predictor) -> Pipeline:
        """Add a predictor to the pipeline. Returns self for chaining."""
        self._predictors.append(predictor)
        return self
    
    def on_packet(self, callback: Callable[[IntentPacket], None]) -> Pipeline:
        """Register a callback for emitted packets. Returns self for chaining."""
        self._callbacks.append(callback)
        return self
    
    def process_frame(self, frame: AudioFrame) -> IntentPacket | None:
        """
        Process a single frame synchronously.
        
        Returns IntentPacket if emit interval has passed, None otherwise.
        """
        self._buffer.push(frame)
        
        for analyzer in self._analyzers:
            result = analyzer.analyze(frame, self._buffer, self._state)
            self._state.analysis_results[analyzer.name] = result
        
        context = PredictionContext(
            frame=frame,
            buffer=self._buffer,
            state=self._state,
            analysis_results=self._state.analysis_results,
        )
        
        for predictor in self._predictors:
            predictor.predict(context, self._state)
        
        should_emit = (frame.timestamp_ms - self._last_emit_ms) >= self._config.emit_interval_ms
        
        if should_emit and self._state.intent_confidence >= self._config.min_confidence_to_emit:
            self._last_emit_ms = frame.timestamp_ms
            packet = self._state.to_packet(frame.frame_id, frame.timestamp_ms)
            
            for callback in self._callbacks:
                callback(packet)
            
            return packet
        
        return None
    
    async def run(self, source: AudioSource) -> AsyncIterator[IntentPacket]:
        """
        Run the pipeline on an audio source.
        
        Yields IntentPackets as they are produced.
        """
        self._running = True
        
        try:
            for frame in source.frames():
                if not self._running:
                    break
                
                packet = self.process_frame(frame)
                if packet is not None:
                    yield packet
                
                await asyncio.sleep(0)
        finally:
            self._running = False
            source.close()
    
    def run_sync(self, source: AudioSource) -> list[IntentPacket]:
        """
        Run the pipeline synchronously (for testing/batch processing).
        
        Returns all emitted packets.
        """
        packets: list[IntentPacket] = []
        
        for frame in source.frames():
            packet = self.process_frame(frame)
            if packet is not None:
                packets.append(packet)
        
        source.close()
        return packets
    
    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self._buffer.clear()
        self._state = PipelineState()
        self._last_emit_ms = 0
