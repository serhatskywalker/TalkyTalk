"""
Base adapter protocol.

Adapters transform IntentPackets for downstream systems.
talkytalk does not implement specific integrations -
adapters are provided by consumers or as separate packages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from talkytalk.core.packet import IntentPacket


T = TypeVar("T")


@dataclass
class AdapterConfig:
    """Base configuration for adapters."""
    pass


class Adapter(ABC, Generic[T]):
    """
    Abstract base for output adapters.
    
    Adapters transform IntentPackets into formats consumable
    by downstream systems (LLMs, game engines, search systems, etc.)
    
    talkytalk provides the protocol; implementations are external.
    
    Example implementations (not included):
    - OpenAIAdapter: Transforms to system prompt modifications
    - WebSocketAdapter: Streams packets over WebSocket
    - UnityAdapter: Formats for Unity game engine consumption
    
    Usage:
        class MyAdapter(Adapter[MyOutputType]):
            def transform(self, packet: IntentPacket) -> MyOutputType:
                return MyOutputType(...)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter name."""
        ...
    
    @abstractmethod
    def transform(self, packet: IntentPacket) -> T:
        """
        Transform an IntentPacket to target format.
        
        Args:
            packet: The IntentPacket to transform
            
        Returns:
            Transformed output in target format
        """
        ...
    
    def batch_transform(self, packets: list[IntentPacket]) -> list[T]:
        """Transform multiple packets. Override for optimization."""
        return [self.transform(p) for p in packets]


class DictAdapter(Adapter[dict[str, Any]]):
    """
    Simple adapter that converts IntentPacket to dictionary.
    
    Useful for JSON serialization or simple integrations.
    """
    
    @property
    def name(self) -> str:
        return "dict"
    
    def transform(self, packet: IntentPacket) -> dict[str, Any]:
        return {
            "intent": packet.intent.value if hasattr(packet.intent, 'value') else str(packet.intent),
            "confidence": packet.confidence,
            "language": packet.language,
            "target_language": packet.target_language,
            "emotion": {
                "arousal": packet.emotion.arousal,
                "valence": packet.emotion.valence,
                "quadrant": packet.emotion.quadrant,
            },
            "timing": {
                "user_paused": packet.timing.user_paused,
                "interrupt_safe": packet.timing.interrupt_safe,
                "speech_likelihood": packet.timing.speech_likelihood,
                "silence_duration_ms": packet.timing.silence_duration_ms,
            },
            "frame_id": packet.frame_id,
            "timestamp_ms": packet.timestamp_ms,
            "is_actionable": packet.is_actionable,
        }


class CallbackAdapter(Adapter[None]):
    """
    Adapter that invokes a callback for each packet.
    
    Useful for event-driven architectures.
    """
    
    def __init__(self, callback: callable) -> None:
        self._callback = callback
    
    @property
    def name(self) -> str:
        return "callback"
    
    def transform(self, packet: IntentPacket) -> None:
        self._callback(packet)
