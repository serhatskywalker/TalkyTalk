"""
Benchmark & Latency Measurement.

Comprehensive performance tracking for real-time requirements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Any
from collections import deque
import statistics


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    name: str
    start_ns: int
    end_ns: int
    
    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000
    
    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000


class LatencyTracker:
    """
    Real-time latency tracking with budget enforcement.
    
    Tracks:
    - Frame processing latency
    - End-to-end pipeline latency
    - Worst-case spikes
    - Jitter (latency variance)
    
    Usage:
        tracker = LatencyTracker(budget_ms=10.0)
        
        with tracker.measure("frame_process"):
            process_frame(frame)
        
        if tracker.is_over_budget:
            log_warning("Latency budget exceeded")
    """
    
    def __init__(
        self,
        budget_ms: float = 10.0,
        history_size: int = 1000,
        spike_threshold_multiplier: float = 3.0,
    ) -> None:
        self._budget_ms = budget_ms
        self._history_size = history_size
        self._spike_threshold_multiplier = spike_threshold_multiplier
        
        self._measurements: dict[str, deque[float]] = {}
        self._current_start: dict[str, int] = {}
        self._spike_count: dict[str, int] = {}
        self._total_count: dict[str, int] = {}
    
    @property
    def budget_ms(self) -> float:
        return self._budget_ms
    
    def start(self, name: str) -> None:
        """Start timing a named operation."""
        self._current_start[name] = time.perf_counter_ns()
    
    def stop(self, name: str) -> float:
        """Stop timing and record measurement. Returns duration in ms."""
        if name not in self._current_start:
            return 0.0
        
        end_ns = time.perf_counter_ns()
        start_ns = self._current_start.pop(name)
        duration_ms = (end_ns - start_ns) / 1_000_000
        
        if name not in self._measurements:
            self._measurements[name] = deque(maxlen=self._history_size)
            self._spike_count[name] = 0
            self._total_count[name] = 0
        
        self._measurements[name].append(duration_ms)
        self._total_count[name] += 1
        
        if len(self._measurements[name]) > 10:
            mean = statistics.mean(self._measurements[name])
            if duration_ms > mean * self._spike_threshold_multiplier:
                self._spike_count[name] += 1
        
        return duration_ms
    
    def measure(self, name: str):
        """Context manager for measuring an operation."""
        return _MeasureContext(self, name)
    
    def get_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a named operation."""
        if name not in self._measurements or len(self._measurements[name]) == 0:
            return {}
        
        data = list(self._measurements[name])
        
        return {
            "mean_ms": statistics.mean(data),
            "median_ms": statistics.median(data),
            "min_ms": min(data),
            "max_ms": max(data),
            "stddev_ms": statistics.stdev(data) if len(data) > 1 else 0.0,
            "p95_ms": self._percentile(data, 95),
            "p99_ms": self._percentile(data, 99),
            "jitter_ms": self._calculate_jitter(data),
            "spike_rate": self._spike_count.get(name, 0) / max(1, self._total_count.get(name, 1)),
            "over_budget_rate": sum(1 for x in data if x > self._budget_ms) / len(data),
            "sample_count": len(data),
        }
    
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all tracked operations."""
        return {name: self.get_stats(name) for name in self._measurements}
    
    def is_over_budget(self, name: str) -> bool:
        """Check if last measurement exceeded budget."""
        if name not in self._measurements or len(self._measurements[name]) == 0:
            return False
        return self._measurements[name][-1] > self._budget_ms
    
    def _percentile(self, data: list[float], p: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    def _calculate_jitter(self, data: list[float]) -> float:
        """Calculate jitter (variation between consecutive samples)."""
        if len(data) < 2:
            return 0.0
        diffs = [abs(data[i] - data[i-1]) for i in range(1, len(data))]
        return statistics.mean(diffs)
    
    def reset(self) -> None:
        """Reset all measurements."""
        self._measurements.clear()
        self._current_start.clear()
        self._spike_count.clear()
        self._total_count.clear()


class _MeasureContext:
    """Context manager for latency measurement."""
    
    def __init__(self, tracker: LatencyTracker, name: str) -> None:
        self._tracker = tracker
        self._name = name
    
    def __enter__(self):
        self._tracker.start(self._name)
        return self
    
    def __exit__(self, *args):
        self._tracker.stop(self._name)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    duration_seconds: float
    frames_processed: int
    packets_emitted: int
    
    latency_stats: dict[str, dict[str, float]]
    
    intent_accuracy: float | None = None
    interrupt_success_rate: float | None = None
    false_silence_rate: float | None = None
    early_intent_precision: float | None = None
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def frames_per_second(self) -> float:
        return self.frames_processed / self.duration_seconds if self.duration_seconds > 0 else 0
    
    @property
    def realtime_factor(self) -> float:
        """How much faster than realtime. >1 means faster than realtime."""
        frame_duration_ms = self.metadata.get("frame_duration_ms", 20)
        expected_duration = (self.frames_processed * frame_duration_ms) / 1000
        return expected_duration / self.duration_seconds if self.duration_seconds > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "frames_processed": self.frames_processed,
            "packets_emitted": self.packets_emitted,
            "frames_per_second": self.frames_per_second,
            "realtime_factor": self.realtime_factor,
            "latency_stats": self.latency_stats,
            "intent_accuracy": self.intent_accuracy,
            "interrupt_success_rate": self.interrupt_success_rate,
            "false_silence_rate": self.false_silence_rate,
            "early_intent_precision": self.early_intent_precision,
            "metadata": self.metadata,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Benchmark: {self.name}",
            f"  Duration: {self.duration_seconds:.2f}s",
            f"  Frames: {self.frames_processed} ({self.frames_per_second:.1f} fps)",
            f"  Realtime factor: {self.realtime_factor:.2f}x",
            f"  Packets emitted: {self.packets_emitted}",
        ]
        
        if "frame_process" in self.latency_stats:
            stats = self.latency_stats["frame_process"]
            lines.extend([
                f"  Frame latency:",
                f"    Mean: {stats.get('mean_ms', 0):.2f}ms",
                f"    P95: {stats.get('p95_ms', 0):.2f}ms",
                f"    P99: {stats.get('p99_ms', 0):.2f}ms",
                f"    Max: {stats.get('max_ms', 0):.2f}ms",
                f"    Jitter: {stats.get('jitter_ms', 0):.2f}ms",
                f"    Over budget: {stats.get('over_budget_rate', 0)*100:.1f}%",
            ])
        
        if self.intent_accuracy is not None:
            lines.append(f"  Intent accuracy: {self.intent_accuracy*100:.1f}%")
        if self.interrupt_success_rate is not None:
            lines.append(f"  Interrupt success: {self.interrupt_success_rate*100:.1f}%")
        
        return "\n".join(lines)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for talkytalk.
    
    Measures:
    - End-to-end latency
    - Frame processing time
    - Worst-case spikes
    - Jitter tolerance
    - Intent accuracy (if ground truth provided)
    - Interrupt success rate
    - False silence detection
    - Early intent precision
    """
    
    def __init__(self, latency_budget_ms: float = 10.0) -> None:
        self._latency_budget_ms = latency_budget_ms
        self._results: list[BenchmarkResult] = []
    
    def run(
        self,
        name: str,
        pipeline,
        source,
        ground_truth: dict | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark on a pipeline with given source."""
        from talkytalk.core.stream import AudioSource
        
        tracker = LatencyTracker(budget_ms=self._latency_budget_ms)
        
        frames_processed = 0
        packets: list = []
        
        start_time = time.perf_counter()
        
        for frame in source.frames():
            with tracker.measure("frame_process"):
                packet = pipeline.process_frame(frame)
            
            frames_processed += 1
            if packet is not None:
                packets.append(packet)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        intent_accuracy = None
        interrupt_success = None
        false_silence = None
        early_precision = None
        
        if ground_truth:
            intent_accuracy = self._calculate_intent_accuracy(packets, ground_truth)
            interrupt_success = self._calculate_interrupt_success(packets, ground_truth)
            false_silence = self._calculate_false_silence(packets, ground_truth)
            early_precision = self._calculate_early_precision(packets, ground_truth)
        
        result = BenchmarkResult(
            name=name,
            duration_seconds=duration,
            frames_processed=frames_processed,
            packets_emitted=len(packets),
            latency_stats=tracker.get_all_stats(),
            intent_accuracy=intent_accuracy,
            interrupt_success_rate=interrupt_success,
            false_silence_rate=false_silence,
            early_intent_precision=early_precision,
            metadata={
                "frame_duration_ms": source.config.frame_duration_ms,
                "sample_rate": source.config.sample_rate,
                "latency_budget_ms": self._latency_budget_ms,
            },
        )
        
        self._results.append(result)
        source.close()
        
        return result
    
    def _calculate_intent_accuracy(self, packets, ground_truth) -> float:
        """Calculate intent prediction accuracy."""
        expected_intents = ground_truth.get("intents", [])
        if not expected_intents or not packets:
            return 0.0
        
        correct = sum(
            1 for p in packets
            if p.intent.value in expected_intents
        )
        return correct / len(packets)
    
    def _calculate_interrupt_success(self, packets, ground_truth) -> float:
        """Calculate interrupt timing success rate."""
        safe_windows = ground_truth.get("safe_interrupt_windows", [])
        if not safe_windows:
            return 0.0
        
        successful = 0
        for p in packets:
            if p.timing.interrupt_safe:
                for start, end in safe_windows:
                    if start <= p.timestamp_ms <= end:
                        successful += 1
                        break
        
        interrupt_count = sum(1 for p in packets if p.timing.interrupt_safe)
        return successful / max(1, interrupt_count)
    
    def _calculate_false_silence(self, packets, ground_truth) -> float:
        """Calculate false silence detection rate."""
        speech_windows = ground_truth.get("speech_windows", [])
        if not speech_windows:
            return 0.0
        
        false_silences = 0
        for p in packets:
            if p.timing.speech_likelihood < 0.3:
                for start, end in speech_windows:
                    if start <= p.timestamp_ms <= end:
                        false_silences += 1
                        break
        
        return false_silences / max(1, len(packets))
    
    def _calculate_early_precision(self, packets, ground_truth) -> float:
        """Calculate early intent prediction precision."""
        final_intent = ground_truth.get("final_intent")
        if not final_intent or not packets:
            return 0.0
        
        early_packets = packets[:len(packets)//2]
        if not early_packets:
            return 0.0
        
        correct_early = sum(
            1 for p in early_packets
            if p.intent.value == final_intent and p.confidence > 0.5
        )
        return correct_early / len(early_packets)
    
    @property
    def results(self) -> list[BenchmarkResult]:
        return list(self._results)
    
    def summary(self) -> str:
        """Get summary of all benchmark results."""
        return "\n\n".join(r.summary() for r in self._results)
