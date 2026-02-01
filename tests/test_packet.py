"""Tests for IntentPacket and related data structures."""

import pytest
from talkytalk.core.packet import IntentPacket, Emotion, Timing, Intent


class TestEmotion:
    def test_default_values(self):
        emotion = Emotion()
        assert emotion.arousal == 0.5
        assert emotion.valence == 0.5
    
    def test_clamping(self):
        emotion = Emotion(arousal=1.5, valence=-0.5)
        assert emotion.arousal == 1.0
        assert emotion.valence == 0.0
    
    def test_quadrant_calm_positive(self):
        emotion = Emotion(arousal=0.3, valence=0.7)
        assert emotion.quadrant == "calm_positive"
    
    def test_quadrant_tense_negative(self):
        emotion = Emotion(arousal=0.8, valence=0.2)
        assert emotion.quadrant == "tense_negative"


class TestTiming:
    def test_default_values(self):
        timing = Timing()
        assert timing.user_paused is False
        assert timing.interrupt_safe is False
        assert timing.speech_likelihood == 1.0
        assert timing.silence_duration_ms == 0
    
    def test_speech_likelihood_clamping(self):
        timing = Timing(speech_likelihood=1.5)
        assert timing.speech_likelihood == 1.0


class TestIntentPacket:
    def test_default_values(self):
        packet = IntentPacket()
        assert packet.intent == Intent.UNKNOWN
        assert packet.confidence == 0.0
        assert packet.language == "unknown"
        assert packet.target_language is None
    
    def test_confidence_clamping(self):
        packet = IntentPacket(confidence=1.5)
        assert packet.confidence == 1.0
    
    def test_is_actionable_true(self):
        packet = IntentPacket(intent=Intent.QUERY, confidence=0.7)
        assert packet.is_actionable is True
    
    def test_is_actionable_false_low_confidence(self):
        packet = IntentPacket(intent=Intent.QUERY, confidence=0.4)
        assert packet.is_actionable is False
    
    def test_is_actionable_false_unknown(self):
        packet = IntentPacket(intent=Intent.UNKNOWN, confidence=0.8)
        assert packet.is_actionable is False
    
    def test_needs_translation(self):
        packet = IntentPacket(
            intent=Intent.TRANSLATE,
            language="en",
            target_language="tr",
        )
        assert packet.needs_translation is True
    
    def test_with_updates(self):
        packet = IntentPacket(intent=Intent.QUERY, confidence=0.5)
        updated = packet.with_updates(confidence=0.8)
        
        assert packet.confidence == 0.5
        assert updated.confidence == 0.8
        assert updated.intent == Intent.QUERY
