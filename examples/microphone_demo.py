#!/usr/bin/env python3
"""
talkytalk Microphone Demo

Gerçek zamanlı mikrofon ile talkytalk'ı test edin.
Konuşun ve pipeline'ın ürettiği sinyalleri görün.

Kullanım:
    python examples/microphone_demo.py

Gerekli:
    pip install sounddevice

Ctrl+C ile durdurun.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talkytalk import Pipeline, PipelineConfig, AudioConfig
from talkytalk.analyzers import VADAnalyzer, ProsodyAnalyzer, EmotionAnalyzer
from talkytalk.predictors import (
    IntentPredictor,
    TimingPredictor,
    EarlyIntentPredictor,
    TurnTakingPredictor,
)
from talkytalk.behavior import BehaviorMapper, BehaviorMode


def clear_line():
    """Clear current line in terminal."""
    print("\r" + " " * 80 + "\r", end="", flush=True)


def color(text: str, code: int) -> str:
    """Add ANSI color to text."""
    return f"\033[{code}m{text}\033[0m"


def format_bar(value: float, width: int = 20, filled: str = "#", empty: str = ".") -> str:
    """Create a visual bar."""
    filled_count = int(value * width)
    return filled * filled_count + empty * (width - filled_count)


def print_packet_live(packet, behavior_signal=None):
    """Print packet info in a live-updating format."""
    
    intent_colors = {
        "query": 36,       # cyan
        "command": 31,     # red
        "conversation": 32, # green
        "unknown": 90,     # gray
    }
    
    intent_str = packet.intent.value
    intent_color = intent_colors.get(intent_str, 37)
    
    arousal_bar = format_bar(packet.emotion.arousal)
    valence_bar = format_bar(packet.emotion.valence)
    confidence_bar = format_bar(packet.confidence)
    speech_bar = format_bar(packet.timing.speech_likelihood)
    
    interrupt_icon = "[OK]" if packet.timing.interrupt_safe else "[..]"   
    paused_icon = "[PAUSE]" if packet.timing.user_paused else "[MIC]"
    
    quadrant = packet.emotion.quadrant
    quadrant_emoji = {
        "calm_positive": "[+]",
        "calm_negative": "[-]",
        "tense_positive": "[!+]",
        "tense_negative": "[!-]",
    }.get(quadrant, "[?]")
    
    print(f"\n{'='*60}")
    print(f"  Frame: {packet.frame_id:5d}  |  Time: {packet.timestamp_ms/1000:.1f}s")
    print(f"{'='*60}")
    
    print(f"\n  {paused_icon} Intent: {color(intent_str.upper(), intent_color)}")
    print(f"     Confidence: [{confidence_bar}] {packet.confidence:.0%}")
    
    print(f"\n  {quadrant_emoji} Emotion ({quadrant})")
    print(f"     Arousal:   [{arousal_bar}] {packet.emotion.arousal:.0%}")
    print(f"     Valence:   [{valence_bar}] {packet.emotion.valence:.0%}")
    
    print(f"\n  {interrupt_icon} Timing")
    print(f"     Speech:    [{speech_bar}] {packet.timing.speech_likelihood:.0%}")
    print(f"     Silence:   {packet.timing.silence_duration_ms}ms")
    print(f"     Interrupt: {'SAFE' if packet.timing.interrupt_safe else 'WAIT'}")
    
    early = packet.analysis_results.get("early_intent")
    if early:
        interruptibility = early.get("interruptibility", 0)
        stable = early.get("hypothesis_stable", False)
        
        int_bar = format_bar(interruptibility)
        stable_icon = "[LOCK]" if stable else "[SPIN]"
        
        print(f"\n  {stable_icon} Early Intent")
        print(f"     Interruptibility: [{int_bar}] {interruptibility:.0%}")
        print(f"     Hypothesis: {'STABLE' if stable else 'UNSTABLE'}")
    
    turn = packet.analysis_results.get("turn_taking")
    if turn:
        state = turn.get("state", "unknown")
        overlap = turn.get("overlap_probability", 0)
        wait = turn.get("suggested_wait_ms", 0)
        
        state_emoji = {
            "user_speaking": "[SPEAK]",
            "user_pausing": "[THINK]",
            "turn_yielded": "[YIELD]",
            "system_can_speak": "[GO]",
        }.get(state, "[?]")
        
        print(f"\n  {state_emoji} Turn-Taking: {state}")
        print(f"     Overlap risk: {overlap:.0%}")
        print(f"     Suggested wait: {wait}ms")
    
    if behavior_signal:
        strategy = behavior_signal.response_strategy.value
        tone = behavior_signal.suggested_tone
        
        print(f"\n  [BEHAVIOR] Signal")
        print(f"     Strategy: {strategy}")
        print(f"     Tone: {tone}")
        print(f"     Soften: {behavior_signal.should_soften}")
        print(f"     Wait: {behavior_signal.should_wait}")


def main():
    print("\n" + "="*60)
    print("  [MIC] talkytalk Microphone Demo")
    print("="*60)
    print("\n  Mikrofona konuşun, pipeline'ın sinyallerini görün.")
    print("  Ctrl+C ile durdurun.\n")
    
    try:
        from talkytalk.sources.microphone import MicrophoneSource, list_audio_devices
    except ImportError as e:
        print(f"\n  [ERROR] Hata: {e}")
        print("\n  sounddevice yüklü değil. Yüklemek için:")
        print("    pip install sounddevice\n")
        return
    
    print("  Mevcut ses cihazları:")
    print("-" * 40)
    list_audio_devices()
    print("-" * 40)
    
    pipeline = (
        Pipeline(PipelineConfig(
            audio=AudioConfig(sample_rate=16000, frame_duration_ms=20),
            emit_interval_ms=200,
        ))
        .add_analyzer(VADAnalyzer(energy_threshold_db=-45.0))
        .add_analyzer(ProsodyAnalyzer())
        .add_analyzer(EmotionAnalyzer())
        .add_predictor(IntentPredictor())
        .add_predictor(TimingPredictor())
        .add_predictor(EarlyIntentPredictor())
        .add_predictor(TurnTakingPredictor())
    )
    
    mapper = BehaviorMapper(mode=BehaviorMode.ASSISTANT)
    
    print("\n  [OK] Pipeline hazir!")
    print("  [REC] Kayit basliyor... (Ctrl+C ile durdurun)\n")
    
    try:
        source = MicrophoneSource(
            sample_rate=16000,
            frame_duration_ms=20,
        )
        
        packet_count = 0
        
        for packet in pipeline.run_sync(source):
            if packet is not None:
                behavior = mapper.map(packet)
                print_packet_live(packet, behavior)
                packet_count += 1
                
    except KeyboardInterrupt:
        print("\n\n  [STOP] Kayit durduruldu.")
        print(f"  [INFO] Toplam {packet_count} paket islendi.\n")
    except Exception as e:
        print(f"\n  [ERROR] Hata: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
