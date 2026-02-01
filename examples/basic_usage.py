"""
talkytalk Basic Usage Example

Demonstrates core pipeline usage with synthetic audio.
"""

import asyncio
from talkytalk import Pipeline, PipelineConfig, AudioConfig
from talkytalk.analyzers import VADAnalyzer, ProsodyAnalyzer, EmotionAnalyzer, LanguageAnalyzer
from talkytalk.predictors import IntentPredictor, TimingPredictor
from talkytalk.sources import SineSource, SilenceSource, NoiseSource
from talkytalk.adapters.base import DictAdapter
import json


def create_default_pipeline() -> Pipeline:
    """Create a fully configured pipeline with all analyzers and predictors."""
    config = PipelineConfig(
        audio=AudioConfig(sample_rate=16000, frame_duration_ms=20),
        buffer_duration_ms=1000,
        emit_interval_ms=100,
    )
    
    return (
        Pipeline(config)
        .add_analyzer(VADAnalyzer(energy_threshold_db=-40, hangover_frames=5))
        .add_analyzer(ProsodyAnalyzer(min_pitch_hz=50, max_pitch_hz=500))
        .add_analyzer(EmotionAnalyzer(smoothing_alpha=0.3))
        .add_analyzer(LanguageAnalyzer(default_language="unknown"))
        .add_predictor(IntentPredictor(decay_rate=0.95))
        .add_predictor(TimingPredictor(pause_threshold_ms=300, turn_end_threshold_ms=700))
    )


def example_sync_processing():
    """Synchronous processing example."""
    print("=" * 60)
    print("Synchronous Processing Example")
    print("=" * 60)
    
    pipeline = create_default_pipeline()
    adapter = DictAdapter()
    
    source = SineSource(frequency_hz=200, duration_ms=500, amplitude=0.5)
    
    packets = pipeline.run_sync(source)
    
    print(f"\nProcessed {len(packets)} packets:\n")
    
    for i, packet in enumerate(packets):
        output = adapter.transform(packet)
        print(f"Packet {i + 1}:")
        print(f"  Intent: {output['intent']} (confidence: {output['confidence']:.2f})")
        print(f"  Emotion: arousal={output['emotion']['arousal']:.2f}, valence={output['emotion']['valence']:.2f}")
        print(f"  Timing: paused={output['timing']['user_paused']}, interrupt_safe={output['timing']['interrupt_safe']}")
        print()


def example_callback_processing():
    """Callback-based processing example."""
    print("=" * 60)
    print("Callback Processing Example")
    print("=" * 60)
    
    def on_packet(packet):
        if packet.timing.interrupt_safe:
            print(f"[{packet.timestamp_ms}ms] INTERRUPT SAFE - Intent: {packet.intent.value}")
        elif packet.is_actionable:
            print(f"[{packet.timestamp_ms}ms] Actionable intent: {packet.intent.value} ({packet.confidence:.2f})")
    
    pipeline = create_default_pipeline()
    pipeline.on_packet(on_packet)
    
    source = SineSource(frequency_hz=150, duration_ms=300, amplitude=0.4)
    pipeline.run_sync(source)
    
    print()


def example_mixed_audio():
    """Process mixed audio (speech-like + silence)."""
    print("=" * 60)
    print("Mixed Audio Example (Speech → Silence → Speech)")
    print("=" * 60)
    
    import numpy as np
    from talkytalk.sources import ArraySource
    from talkytalk.core.stream import AudioConfig
    
    config = AudioConfig(sample_rate=16000)
    
    duration_speech1 = 0.3
    duration_silence = 0.5
    duration_speech2 = 0.2
    
    t1 = np.arange(int(config.sample_rate * duration_speech1)) / config.sample_rate
    speech1 = (0.4 * np.sin(2 * np.pi * 180 * t1)).astype(np.float32)
    
    silence = np.zeros(int(config.sample_rate * duration_silence), dtype=np.float32)
    
    t2 = np.arange(int(config.sample_rate * duration_speech2)) / config.sample_rate
    speech2 = (0.5 * np.sin(2 * np.pi * 220 * t2)).astype(np.float32)
    
    combined = np.concatenate([speech1, silence, speech2])
    
    pipeline = create_default_pipeline()
    source = ArraySource(combined, config)
    
    packets = pipeline.run_sync(source)
    
    print(f"\nTotal duration: {(duration_speech1 + duration_silence + duration_speech2) * 1000:.0f}ms")
    print(f"Packets emitted: {len(packets)}\n")
    
    for packet in packets:
        status = "SPEECH" if packet.timing.speech_likelihood > 0.5 else "PAUSE"
        interrupt = " [INTERRUPT OK]" if packet.timing.interrupt_safe else ""
        print(f"[{packet.timestamp_ms:4d}ms] {status} | likelihood={packet.timing.speech_likelihood:.2f}{interrupt}")


async def example_async_processing():
    """Async processing example."""
    print("=" * 60)
    print("Async Processing Example")
    print("=" * 60)
    
    pipeline = create_default_pipeline()
    source = SineSource(frequency_hz=200, duration_ms=400, amplitude=0.5)
    
    count = 0
    async for packet in pipeline.run(source):
        count += 1
        print(f"Async packet {count}: intent={packet.intent.value}, confidence={packet.confidence:.2f}")
    
    print(f"\nTotal async packets: {count}")


if __name__ == "__main__":
    example_sync_processing()
    print("\n")
    example_callback_processing()
    print("\n")
    example_mixed_audio()
    print("\n")
    asyncio.run(example_async_processing())
