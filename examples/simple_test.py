#!/usr/bin/env python3
"""
talkytalk Simple Test

Mikrofon olmadan temel pipeline testini çalıştırır.
Sentetik ses ile pipeline'ın çalıştığını doğrular.

Kullanım:
    python examples/simple_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talkytalk import Pipeline, PipelineConfig, AudioConfig
from talkytalk.analyzers import VADAnalyzer, ProsodyAnalyzer, EmotionAnalyzer
from talkytalk.predictors import IntentPredictor, TimingPredictor
from talkytalk.sources.synthetic import SineSource, NoiseSource, SilenceSource
from talkytalk.benchmark import BenchmarkSuite


def test_basic_pipeline():
    """Test basic pipeline with synthetic audio."""
    print("\n" + "="*50)
    print("  [TEST] Temel Pipeline Testi")
    print("="*50)
    
    pipeline = (
        Pipeline(PipelineConfig(emit_interval_ms=100))
        .add_analyzer(VADAnalyzer())
        .add_analyzer(ProsodyAnalyzer())
        .add_analyzer(EmotionAnalyzer())
        .add_predictor(IntentPredictor())
        .add_predictor(TimingPredictor())
    )
    
    source = SineSource(frequency_hz=200, duration_ms=500)
    
    packets = list(pipeline.run_sync(source))
    
    print(f"\n  [OK] {len(packets)} paket üretildi")
    
    if packets:
        p = packets[-1]
        print(f"\n  Son paket:")
        print(f"    Intent: {p.intent.value}")
        print(f"    Confidence: {p.confidence:.2f}")
        print(f"    Arousal: {p.emotion.arousal:.2f}")
        print(f"    Valence: {p.emotion.valence:.2f}")
        print(f"    Interrupt safe: {p.timing.interrupt_safe}")
    
    return len(packets) > 0


def test_silence_detection():
    """Test silence vs speech detection."""
    print("\n" + "="*50)
    print("  [TEST] Sessizlik Algılama Testi")
    print("="*50)
    
    pipeline = (
        Pipeline(PipelineConfig(emit_interval_ms=100))
        .add_analyzer(VADAnalyzer())
        .add_predictor(TimingPredictor())
    )
    
    silence_source = SilenceSource(duration_ms=500)
    noise_source = NoiseSource(amplitude=0.3, duration_ms=500)
    
    silence_packets = list(pipeline.run_sync(silence_source))
    pipeline.reset()
    noise_packets = list(pipeline.run_sync(noise_source))
    
    silence_speech = sum(1 for p in silence_packets if p.timing.speech_likelihood > 0.5)
    noise_speech = sum(1 for p in noise_packets if p.timing.speech_likelihood > 0.5)
    
    print(f"\n  Sessizlik: {silence_speech}/{len(silence_packets)} frame konuşma algılandı")
    print(f"  Gürültü: {noise_speech}/{len(noise_packets)} frame konuşma algılandı")
    
    return silence_speech < noise_speech


def test_benchmark():
    """Run basic benchmark."""
    print("\n" + "="*50)
    print("  [TEST] Benchmark Testi")
    print("="*50)
    
    pipeline = (
        Pipeline(PipelineConfig(emit_interval_ms=50))
        .add_analyzer(VADAnalyzer())
        .add_analyzer(ProsodyAnalyzer())
        .add_predictor(IntentPredictor())
        .add_predictor(TimingPredictor())
    )
    
    source = SineSource(frequency_hz=300, duration_ms=2000)
    
    suite = BenchmarkSuite(latency_budget_ms=10.0)
    result = suite.run("basic_test", pipeline, source)
    
    print(f"\n{result.summary()}")
    
    return result.realtime_factor > 1.0


def main():
    print("\n" + "="*50)
    print("  [*] talkytalk Test Suite")
    print("="*50)
    
    results = {
        "Temel Pipeline": test_basic_pipeline(),
        "Sessizlik Algılama": test_silence_detection(),
        "Benchmark": test_benchmark(),
    }
    
    print("\n" + "="*50)
    print("  [RESULTS] Test Sonuclari")
    print("="*50)
    
    all_passed = True
    for name, passed in results.items():
        icon = "[OK]" if passed else "[FAIL]"
        print(f"  {icon} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("  [SUCCESS] Tum testler basarili!")
    else:
        print("  [WARNING] Bazi testler basarisiz.")
    print("="*50 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
