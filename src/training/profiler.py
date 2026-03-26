"""
Benchmark utilities for profiling training pipeline bottlenecks.

Usage:
    from src.training.profiler import benchmark, PhaseTimer, profile_training_step

Decorator:
    @benchmark
    def my_function(): ...

Phase timer (context manager for granular step profiling):
    with PhaseTimer("forward") as t:
        output = model(inputs)
    print(t.elapsed)
"""

import logging
import time
from collections import defaultdict
from functools import wraps

import torch

LOGGER = logging.getLogger(__name__)


def benchmark(func):
    """Decorator that logs wall-clock time for each call."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        LOGGER.info("[benchmark] %s: %.4fs", func.__name__, elapsed)
        return result

    return wrapper


class PhaseTimer:
    """Context manager that records elapsed time for a named phase.

    Automatically calls ``torch.cuda.synchronize()`` on CUDA so that
    GPU work is included in the measurement.
    """

    def __init__(self, name: str):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
        return False


class StepProfiler:
    """Accumulates per-phase timings across many training steps.

    Example::

        sp = StepProfiler()
        for batch in loader:
            with sp.phase("data_load"):
                inputs, targets = prepare(batch)
            with sp.phase("forward"):
                output = model(inputs)
            with sp.phase("backward"):
                loss.backward()
            with sp.phase("optim_step"):
                optimizer.step()
            sp.step_done()

        sp.report()
    """

    def __init__(self):
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._step = 0

    def phase(self, name: str) -> PhaseTimer:
        """Return a PhaseTimer that auto-records into this profiler."""
        return _RecordingPhaseTimer(name, self)

    def _record(self, name: str, elapsed: float) -> None:
        self._totals[name] += elapsed
        self._counts[name] += 1

    def step_done(self) -> None:
        self._step += 1

    def report(self) -> dict[str, dict[str, float]]:
        """Log and return a summary of per-phase timings."""
        total_time = sum(self._totals.values())
        summary: dict[str, dict[str, float]] = {}

        LOGGER.info("=" * 60)
        LOGGER.info("  STEP PROFILER REPORT  (%d steps)", self._step)
        LOGGER.info("=" * 60)
        LOGGER.info("%-20s %10s %10s %8s", "Phase", "Total (s)", "Avg (ms)", "% Time")
        LOGGER.info("-" * 60)

        for name in self._totals:
            t = self._totals[name]
            c = self._counts[name]
            avg_ms = (t / c) * 1000 if c else 0
            pct = (t / total_time) * 100 if total_time else 0

            summary[name] = {"total_s": t, "avg_ms": avg_ms, "pct": pct, "count": c}
            LOGGER.info("%-20s %10.3f %10.3f %7.1f%%", name, t, avg_ms, pct)

        LOGGER.info("-" * 60)
        LOGGER.info("%-20s %10.3f", "TOTAL", total_time)

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            current_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            LOGGER.info("GPU peak memory: %.1f MB  |  current: %.1f MB", peak_mb, current_mb)

        LOGGER.info("=" * 60)
        return summary


class _RecordingPhaseTimer(PhaseTimer):
    """PhaseTimer that records into a StepProfiler on exit."""

    def __init__(self, name: str, profiler: StepProfiler):
        super().__init__(name)
        self._profiler = profiler

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self._profiler._record(self.name, self.elapsed)
        return False
