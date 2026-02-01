"""
talkytalk - Real-time behavioral speech signal processor

talkytalk produces probabilistic intent signals from streaming audio.
It does not speak, decide, or respond. It only signals.
"""

from talkytalk.core.packet import IntentPacket, Emotion, Timing
from talkytalk.core.stream import AudioFrame, AudioConfig
from talkytalk.core.pipeline import Pipeline, PipelineConfig
from talkytalk.analyzers.base import Analyzer
from talkytalk.predictors.base import Predictor
from talkytalk.adapters.base import Adapter

__version__ = "0.1.0"
__all__ = [
    # Core data structures
    "IntentPacket",
    "Emotion",
    "Timing",
    "AudioFrame",
    "AudioConfig",
    # Pipeline
    "Pipeline",
    "PipelineConfig",
    # Extension protocols
    "Analyzer",
    "Predictor",
    "Adapter",
]
