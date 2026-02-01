"""
Base analyzer protocol.

Analyzers extract features from audio frames.
They do not make decisions—they produce signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from talkytalk.core.stream import AudioFrame, FrameBuffer
    from talkytalk.core.pipeline import PipelineState


@dataclass
class AnalysisResult:
    """Base result from an analyzer."""
    analyzer_name: str
    frame_id: int
    timestamp_ms: int
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class Analyzer(ABC):
    """
    Abstract base for audio analyzers.
    
    Analyzers are stateless feature extractors.
    They receive frames and produce AnalysisResults.
    
    Implementation requirements:
    - Must be fast (< 5ms per frame ideally)
    - Should not block
    - May use buffer for temporal context
    - Must not modify pipeline state directly (return results only)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique analyzer name."""
        ...
    
    @abstractmethod
    def analyze(
        self,
        frame: AudioFrame,
        buffer: FrameBuffer,
        state: PipelineState,
    ) -> AnalysisResult:
        """
        Analyze a single frame.
        
        Args:
            frame: Current audio frame
            buffer: Recent frame history
            state: Current pipeline state (read-only)
        
        Returns:
            AnalysisResult with extracted features
        """
        ...
    
    def reset(self) -> None:
        """Reset analyzer state (if any)."""
        pass
