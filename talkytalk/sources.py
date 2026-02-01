"""
Audio sources for talkytalk.

Re-exports from talkytalk.sources module for backward compatibility.
"""

from talkytalk.sources.synthetic import ArraySource, SineSource, NoiseSource, SilenceSource
from talkytalk.sources.microphone import MicrophoneSource

__all__ = [
    "ArraySource",
    "SineSource",
    "NoiseSource",
    "SilenceSource",
    "MicrophoneSource",
]
