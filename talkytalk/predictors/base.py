"""
Base predictor protocol.

Predictors make probabilistic predictions based on analysis results.
They update pipeline state with their predictions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from talkytalk.core.stream import AudioFrame, FrameBuffer
    from talkytalk.core.pipeline import PipelineState
    from talkytalk.analyzers.base import AnalysisResult


@dataclass
class PredictionContext:
    """Context passed to predictors."""
    frame: AudioFrame
    buffer: FrameBuffer
    state: PipelineState
    analysis_results: dict[str, AnalysisResult]


class Predictor(ABC):
    """
    Abstract base for predictors.
    
    Predictors consume analysis results and update pipeline state
    with probabilistic predictions. Unlike analyzers, predictors
    are allowed to modify state.
    
    Key principle: Predictions are probabilistic, retractable,
    and may be wrong. Early wrong > late right.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique predictor name."""
        ...
    
    @abstractmethod
    def predict(self, context: PredictionContext, state: PipelineState) -> None:
        """
        Make predictions and update state.
        
        Args:
            context: Current prediction context
            state: Pipeline state to update
        """
        ...
    
    def reset(self) -> None:
        """Reset predictor state (if any)."""
        pass
