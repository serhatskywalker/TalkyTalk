"""
Emotion → Behavior Mapping Layer.

"Emotion ölçmek yetmez, kullanmak gerekir"

Translates emotional state into behavioral signals
that downstream systems can act upon.

This layer is LLM-independent and system-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from talkytalk.core.packet import IntentPacket, Emotion, Timing


class BehaviorMode(str, Enum):
    """Predefined behavior modes for different contexts."""
    ASSISTANT = "assistant"
    TEACHER = "teacher"
    GAME_NPC = "game_npc"
    COMPANION = "companion"
    CUSTOMER_SERVICE = "customer_service"
    CUSTOM = "custom"


class ResponseStrategy(str, Enum):
    """How the system should respond."""
    IMMEDIATE = "immediate"
    GENTLE = "gentle"
    WAIT = "wait"
    MIRROR = "mirror"
    CALM_DOWN = "calm_down"
    ENERGIZE = "energize"


@dataclass(frozen=True)
class BehaviorSignal:
    """
    Behavioral signal for downstream systems.
    
    This is what gets sent to LLMs, game engines, or UI systems.
    """
    response_strategy: ResponseStrategy
    
    should_soften: bool
    should_speed_up: bool
    should_wait: bool
    
    response_delay_ms: int
    
    energy_level: float
    formality_level: float
    empathy_level: float
    
    suggested_tone: str
    
    raw_emotion: Emotion
    raw_timing: Timing
    
    def to_prompt_prefix(self) -> str:
        """Generate a prompt prefix for LLM consumption."""
        parts = []
        
        if self.should_soften:
            parts.append("Respond gently and with understanding.")
        if self.should_speed_up:
            parts.append("Be concise and direct.")
        if self.empathy_level > 0.7:
            parts.append("Show empathy in your response.")
        
        if self.suggested_tone:
            parts.append(f"Tone: {self.suggested_tone}.")
        
        return " ".join(parts) if parts else ""


class BehaviorMapper:
    """
    Maps emotional and timing signals to behavioral recommendations.
    
    Different modes produce different mappings:
    - TEACHER: More patience, slower responses, encouraging
    - GAME_NPC: Quick reactions, match user energy
    - ASSISTANT: Balanced, professional
    - COMPANION: High empathy, mirroring
    
    Usage:
        mapper = BehaviorMapper(mode=BehaviorMode.ASSISTANT)
        signal = mapper.map(packet)
        
        if signal.should_wait:
            await asyncio.sleep(signal.response_delay_ms / 1000)
        
        llm_prompt = signal.to_prompt_prefix() + user_message
    """
    
    def __init__(
        self,
        mode: BehaviorMode = BehaviorMode.ASSISTANT,
        custom_mapper: Callable[[IntentPacket], BehaviorSignal] | None = None,
    ) -> None:
        self._mode = mode
        self._custom_mapper = custom_mapper
        
        self._mode_configs = {
            BehaviorMode.ASSISTANT: {
                "base_delay": 100,
                "formality": 0.6,
                "empathy_weight": 0.5,
                "energy_matching": 0.3,
            },
            BehaviorMode.TEACHER: {
                "base_delay": 300,
                "formality": 0.7,
                "empathy_weight": 0.8,
                "energy_matching": 0.2,
            },
            BehaviorMode.GAME_NPC: {
                "base_delay": 50,
                "formality": 0.3,
                "empathy_weight": 0.4,
                "energy_matching": 0.9,
            },
            BehaviorMode.COMPANION: {
                "base_delay": 200,
                "formality": 0.3,
                "empathy_weight": 0.9,
                "energy_matching": 0.7,
            },
            BehaviorMode.CUSTOMER_SERVICE: {
                "base_delay": 150,
                "formality": 0.8,
                "empathy_weight": 0.7,
                "energy_matching": 0.2,
            },
        }
    
    @property
    def mode(self) -> BehaviorMode:
        return self._mode
    
    def set_mode(self, mode: BehaviorMode) -> None:
        """Change behavior mode."""
        self._mode = mode
    
    def map(self, packet: IntentPacket) -> BehaviorSignal:
        """Map an IntentPacket to a BehaviorSignal."""
        if self._mode == BehaviorMode.CUSTOM and self._custom_mapper:
            return self._custom_mapper(packet)
        
        config = self._mode_configs.get(
            self._mode,
            self._mode_configs[BehaviorMode.ASSISTANT]
        )
        
        emotion = packet.emotion
        timing = packet.timing
        
        strategy = self._determine_strategy(emotion, timing)
        
        should_soften = (
            emotion.valence < 0.4 or
            emotion.arousal > 0.7
        )
        
        should_speed_up = (
            emotion.arousal > 0.6 and
            packet.intent.value in ["command", "query"]
        )
        
        should_wait = (
            not timing.interrupt_safe or
            timing.speech_likelihood > 0.5
        )
        
        base_delay = config["base_delay"]
        if should_soften:
            base_delay += 100
        if emotion.arousal > 0.7:
            base_delay -= 50
        if not timing.interrupt_safe:
            base_delay += 200
        response_delay_ms = max(0, base_delay)
        
        energy_matching = config["energy_matching"]
        energy_level = (
            0.5 * (1 - energy_matching) +
            emotion.arousal * energy_matching
        )
        
        empathy_level = config["empathy_weight"]
        if emotion.valence < 0.4:
            empathy_level = min(1.0, empathy_level + 0.2)
        
        suggested_tone = self._suggest_tone(emotion, self._mode)
        
        return BehaviorSignal(
            response_strategy=strategy,
            should_soften=should_soften,
            should_speed_up=should_speed_up,
            should_wait=should_wait,
            response_delay_ms=response_delay_ms,
            energy_level=energy_level,
            formality_level=config["formality"],
            empathy_level=empathy_level,
            suggested_tone=suggested_tone,
            raw_emotion=emotion,
            raw_timing=timing,
        )
    
    def _determine_strategy(
        self,
        emotion: Emotion,
        timing: Timing,
    ) -> ResponseStrategy:
        """Determine response strategy from emotional state."""
        
        if emotion.arousal > 0.7 and emotion.valence < 0.4:
            return ResponseStrategy.CALM_DOWN
        
        if emotion.arousal < 0.3 and emotion.valence < 0.4:
            return ResponseStrategy.ENERGIZE
        
        if not timing.interrupt_safe:
            return ResponseStrategy.WAIT
        
        if emotion.arousal > 0.6:
            return ResponseStrategy.IMMEDIATE
        
        if emotion.valence < 0.4:
            return ResponseStrategy.GENTLE
        
        return ResponseStrategy.MIRROR
    
    def _suggest_tone(self, emotion: Emotion, mode: BehaviorMode) -> str:
        """Suggest a tone based on emotion and mode."""
        quadrant = emotion.quadrant
        
        tone_map = {
            "calm_positive": {
                BehaviorMode.ASSISTANT: "friendly and helpful",
                BehaviorMode.TEACHER: "encouraging and warm",
                BehaviorMode.GAME_NPC: "relaxed and welcoming",
                BehaviorMode.COMPANION: "warm and content",
                BehaviorMode.CUSTOMER_SERVICE: "professional and pleasant",
            },
            "calm_negative": {
                BehaviorMode.ASSISTANT: "understanding and supportive",
                BehaviorMode.TEACHER: "patient and reassuring",
                BehaviorMode.GAME_NPC: "concerned but hopeful",
                BehaviorMode.COMPANION: "gentle and caring",
                BehaviorMode.CUSTOMER_SERVICE: "empathetic and solution-focused",
            },
            "tense_positive": {
                BehaviorMode.ASSISTANT: "enthusiastic and efficient",
                BehaviorMode.TEACHER: "energetic and motivating",
                BehaviorMode.GAME_NPC: "excited and engaged",
                BehaviorMode.COMPANION: "matching enthusiasm",
                BehaviorMode.CUSTOMER_SERVICE: "proactive and upbeat",
            },
            "tense_negative": {
                BehaviorMode.ASSISTANT: "calm and reassuring",
                BehaviorMode.TEACHER: "steady and grounding",
                BehaviorMode.GAME_NPC: "de-escalating",
                BehaviorMode.COMPANION: "soothing and present",
                BehaviorMode.CUSTOMER_SERVICE: "apologetic and resolution-focused",
            },
        }
        
        return tone_map.get(quadrant, {}).get(mode, "neutral")
