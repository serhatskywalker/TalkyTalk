"""Tests for the processing pipeline."""

import pytest
import numpy as np

from talkytalk.core.pipeline import Pipeline, PipelineConfig
from talkytalk.core.stream import AudioConfig
from talkytalk.core.packet import Intent
from talkytalk.analyzers import VADAnalyzer, ProsodyAnalyzer, EmotionAnalyzer
from talkytalk.predictors import IntentPredictor, TimingPredictor
from talkytalk.sources import SineSource, SilenceSource, ArraySource


class TestPipeline:
    def test_pipeline_creation(self):
        pipeline = Pipeline()
        assert pipeline.config is not None
        assert len(pipeline._analyzers) == 0
        assert len(pipeline._predictors) == 0
    
    def test_add_analyzer_chaining(self):
        pipeline = (
            Pipeline()
            .add_analyzer(VADAnalyzer())
            .add_analyzer(ProsodyAnalyzer())
        )
        assert len(pipeline._analyzers) == 2
    
    def test_add_predictor_chaining(self):
        pipeline = (
            Pipeline()
            .add_predictor(IntentPredictor())
            .add_predictor(TimingPredictor())
        )
        assert len(pipeline._predictors) == 2
    
    def test_process_sine_wave(self):
        """Process a sine wave and verify packets are emitted."""
        pipeline = (
            Pipeline(PipelineConfig(emit_interval_ms=100))
            .add_analyzer(VADAnalyzer())
            .add_analyzer(ProsodyAnalyzer())
            .add_predictor(IntentPredictor())
            .add_predictor(TimingPredictor())
        )
        
        source = SineSource(frequency_hz=200, duration_ms=500, amplitude=0.5)
        packets = pipeline.run_sync(source)
        
        assert len(packets) > 0
        for packet in packets:
            assert packet.frame_id >= 0
            assert packet.timestamp_ms >= 0
    
    def test_process_silence(self):
        """Process silence and verify low speech probability."""
        pipeline = (
            Pipeline(PipelineConfig(emit_interval_ms=50))
            .add_analyzer(VADAnalyzer())
            .add_predictor(TimingPredictor())
        )
        
        source = SilenceSource(duration_ms=300)
        packets = pipeline.run_sync(source)
        
        assert len(packets) > 0
        last_packet = packets[-1]
        assert last_packet.timing.speech_likelihood < 0.5
    
    def test_callback_invocation(self):
        """Verify callbacks are called for each packet."""
        received_packets = []
        
        pipeline = (
            Pipeline(PipelineConfig(emit_interval_ms=50))
            .add_analyzer(VADAnalyzer())
            .on_packet(lambda p: received_packets.append(p))
        )
        
        source = SineSource(frequency_hz=200, duration_ms=200)
        packets = pipeline.run_sync(source)
        
        assert len(received_packets) == len(packets)
    
    def test_reset(self):
        """Verify pipeline reset clears state."""
        pipeline = Pipeline()
        pipeline.add_analyzer(VADAnalyzer())
        
        source = SineSource(duration_ms=100)
        pipeline.run_sync(source)
        
        pipeline.reset()
        
        assert pipeline.state.current_intent == Intent.UNKNOWN
        assert pipeline.state.intent_confidence == 0.0


class TestVADAnalyzer:
    def test_silence_detection(self):
        vad = VADAnalyzer()
        config = AudioConfig()
        
        from talkytalk.core.stream import AudioFrame, FrameBuffer
        from talkytalk.core.pipeline import PipelineState
        
        silent_data = np.zeros(config.frame_size, dtype=np.float32)
        frame = AudioFrame(data=silent_data, frame_id=0, timestamp_ms=0, config=config)
        buffer = FrameBuffer()
        state = PipelineState()
        
        result = vad.analyze(frame, buffer, state)
        
        assert result.data.get("is_speech") is False or result.speech_probability < 0.5
    
    def test_speech_detection(self):
        vad = VADAnalyzer()
        config = AudioConfig()
        
        from talkytalk.core.stream import AudioFrame, FrameBuffer
        from talkytalk.core.pipeline import PipelineState
        
        t = np.arange(config.frame_size) / config.sample_rate
        loud_data = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        frame = AudioFrame(data=loud_data, frame_id=0, timestamp_ms=0, config=config)
        buffer = FrameBuffer()
        state = PipelineState()
        
        result = vad.analyze(frame, buffer, state)
        
        assert result.energy_db > -40
