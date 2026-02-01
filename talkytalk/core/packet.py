"""
IntentPacket - The sole output of talkytalk.

This is not a decision. This is not a command.
This is an early, probabilistic, retractable signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Intent(str, Enum):
    """Known intent categories."""
    PLAY_MUSIC = "play_music"
    TRANSLATE = "translate"
    QUERY = "query"
    CONVERSATION = "conversation"
    COMMAND = "command"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Emotion:
    """
    Dimensional emotion representation.
    
    Based on Russell's circumplex model:
    - arousal: calm (0.0) ↔ excited/tense (1.0)
    - valence: negative (0.0) ↔ positive (1.0)
    """
    arousal: float = 0.5
    valence: float = 0.5
    
    def __post_init__(self) -> None:
        if not (0.0 <= self.arousal <= 1.0):
            object.__setattr__(self, 'arousal', max(0.0, min(1.0, self.arousal)))
        if not (0.0 <= self.valence <= 1.0):
            object.__setattr__(self, 'valence', max(0.0, min(1.0, self.valence)))
    
    @property
    def quadrant(self) -> Literal["calm_positive", "calm_negative", "tense_positive", "tense_negative"]:
        """Return emotional quadrant."""
        if self.arousal >= 0.5:
            return "tense_positive" if self.valence >= 0.5 else "tense_negative"
        return "calm_positive" if self.valence >= 0.5 else "calm_negative"


@dataclass(frozen=True, slots=True)
class Timing:
    """
    Temporal signals for downstream systems.
    
    - user_paused: meaningful pause detected (not just breathing)
    - interrupt_safe: system intervention is appropriate now
    - speech_likelihood: probability that user is still speaking (0.0-1.0)
    - silence_duration_ms: milliseconds since last speech
    """
    user_paused: bool = False
    interrupt_safe: bool = False
    speech_likelihood: float = 1.0
    silence_duration_ms: int = 0
    
    def __post_init__(self) -> None:
        if not (0.0 <= self.speech_likelihood <= 1.0):
            object.__setattr__(self, 'speech_likelihood', max(0.0, min(1.0, self.speech_likelihood)))


@dataclass(frozen=True, slots=True)
class IntentPacket:
    """
    The single, unified output of talkytalk.
    
    Every field is probabilistic. Nothing is final.
    Downstream systems consume this and make their own decisions.
    """
    intent: Intent | str = Intent.UNKNOWN
    confidence: float = 0.0
    language: str = "unknown"
    target_language: str | None = None
    emotion: Emotion = field(default_factory=Emotion)
    timing: Timing = field(default_factory=Timing)
    frame_id: int = 0
    timestamp_ms: int = 0
    
    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            object.__setattr__(self, 'confidence', max(0.0, min(1.0, self.confidence)))
        if isinstance(self.intent, str) and self.intent not in Intent.__members__.values():
            try:
                object.__setattr__(self, 'intent', Intent(self.intent))
            except ValueError:
                pass
    
    @property
    def is_actionable(self) -> bool:
        """Heuristic: confidence > 0.6 and not unknown."""
        return self.confidence > 0.6 and self.intent != Intent.UNKNOWN
    
    @property
    def needs_translation(self) -> bool:
        """Check if translation intent is detected."""
        return (
            self.intent == Intent.TRANSLATE 
            and self.target_language is not None
            and self.target_language != self.language
        )
    
    def with_updates(self, **kwargs) -> IntentPacket:
        """Create a new packet with updated fields."""
        current = {
            'intent': self.intent,
            'confidence': self.confidence,
            'language': self.language,
            'target_language': self.target_language,
            'emotion': self.emotion,
            'timing': self.timing,
            'frame_id': self.frame_id,
            'timestamp_ms': self.timestamp_ms,
        }
        current.update(kwargs)
        return IntentPacket(**current)
