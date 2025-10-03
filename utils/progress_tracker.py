"""
Elite Progress Tracking Framework (Synthesized Edition)
Combines the best features from Claude and Gemini upgrades while maintaining
backward compatibility with existing harvesting engine components.

Key Features:
• Observer pattern for decoupled, extensible reporting
• Dependency injection for testability and flexibility  
• Robust time handling with monotonic clock
• Phase tracking with automatic duration calculation
• Context manager support for safe lifecycle management
• Type-safe immutable progress reports
φ = 1.618 033 988 749 895
"""
import abc
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Core Data Structures (Immutable DTOs)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProgressReport:
    """Immutable snapshot of progress state - prevents observer interference."""
    phase: str
    total_items: int
    processed_items: int
    failed_items: int
    progress_pct: float
    elapsed_seconds: float
    rate_per_second: float
    eta: Optional[datetime]
    phase_durations: Dict[str, float]

@dataclass
class ProgressState:
    """Internal mutable state container - pure data, no logic."""
    start_time: float = 0.0
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    current_phase: str = "initializing"
    phase_start_time: float = 0.0
    phase_durations: Dict[str, float] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# Abstractions for Dependency Injection
# ─────────────────────────────────────────────────────────────────────────────
class Clock(Protocol):
    """Time abstraction for testing and monotonic time handling."""
    def now(self) -> float: ...

class MonotonicClock(Clock):
    """Production clock using time.monotonic for reliable intervals."""
    def now(self) -> float:
        return time.monotonic()

class ProgressObserver(abc.ABC):
    """Observer interface for progress event notifications."""
    @abc.abstractmethod
    def on_update(self, report: ProgressReport): ...
        
    @abc.abstractmethod
    def on_finish(self, report: ProgressReport): ...

# ─────────────────────────────────────────────────────────────────────────────
# Concrete Observer Implementations
# ─────────────────────────────────────────────────────────────────────────────
class LoggingObserver(ProgressObserver):
    """Observer that logs progress with intelligent, monotonic-clock-based throttling."""
    
    def __init__(self, logger: logging.Logger, clock: Clock, log_interval_seconds: int = 10):
        self.logger = logger
        self._clock = clock
        self._log_interval = log_interval_seconds
        self._last_log_time = -float('inf')  # Use monotonic time for tracking

    def on_update(self, report: ProgressReport):
        now = self._clock.now()
        if now - self._last_log_time > self._log_interval:
            eta_str = report.eta.strftime('%Y-%m-%d %H:%M:%S') if report.eta else "N/A"
            success_items = report.processed_items - report.failed_items
            
            self.logger.info(
                f"📊 Phase '{report.phase}': {success_items}/{report.total_items} "
                f"({report.progress_pct:.1f}%) | "
                f"⚡ {report.rate_per_second:.1f}/s | "
                f"🕒 ETA: {eta_str}"
            )
            self._last_log_time = now

    def on_finish(self, report: ProgressReport):
        total_duration = timedelta(seconds=int(report.elapsed_seconds))
        success_items = report.processed_items - report.failed_items
        success_rate = (success_items / report.processed_items * 100) if report.processed_items > 0 else 0

        summary_lines = [
            "\n" + "="*50,
            "🎉 HARVESTING COMPLETE",
            "="*50,
            f"  📁 Total items:     {report.total_items}",
            f"  ✅ Processed:       {report.processed_items}",
            f"  ❌ Failed:          {report.failed_items}",
            f"  📈 Success Rate:    {success_rate:.1f}%",
            f"  ⏱️  Total Duration:  {total_duration}",
            f"  ⚡ Average Rate:    {report.rate_per_second:.2f} items/sec",
        ]
        
        if report.phase_durations:
            summary_lines.append("  🔄 Phase Breakdown:")
            for phase, duration in report.phase_durations.items():
                summary_lines.append(f"     • {phase}: {timedelta(seconds=int(duration))}")
        
        summary_lines.append("="*50)
        self.logger.info("\n".join(summary_lines))

class StatisticsObserver(ProgressObserver):
    """Observer that tracks running statistics without storing all reports."""
    
    def __init__(self):
        self.peak_rate: float = 0.0
        self.min_rate: float = float('inf')
        self.total_updates: int = 0
        self.rate_sum: float = 0.0
        self.final_report: Optional[ProgressReport] = None
    
    def on_update(self, report: ProgressReport):
        self.total_updates += 1
        self.rate_sum += report.rate_per_second
        
        if report.rate_per_second > self.peak_rate:
            self.peak_rate = report.rate_per_second
        # Only update min_rate if the rate is meaningful (greater than zero)
        if 0 < report.rate_per_second < self.min_rate:
            self.min_rate = report.rate_per_second
    
    def on_finish(self, report: ProgressReport):
        self.final_report = report
        avg_rate = (self.rate_sum / self.total_updates) if self.total_updates > 0 else 0.0
        min_rate_str = f"{self.min_rate:.2f}/s" if self.min_rate != float('inf') else "N/A"
        
        logger.info(
            f"📊 Statistics Summary: "
            f"Peak Rate: {self.peak_rate:.2f}/s, "
            f"Min Rate: {min_rate_str}, "
            f"Avg Update Rate: {avg_rate:.2f}/s, "
            f"Total Updates: {self.total_updates}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Pure Logic Components (Testable Business Logic)
# ─────────────────────────────────────────────────────────────────────────────
class ReportGenerator:
    """Calculates progress reports from state - pure function, no side effects."""
    
    def __init__(self, clock: Clock):
        self._clock = clock

    def generate(self, state: ProgressState) -> ProgressReport:
        elapsed = self._clock.now() - state.start_time
        progress_pct = (state.processed_items / state.total_items * 100) if state.total_items > 0 else 0.0

        # Calculate rate with smoothing for stability
        rate = 0.0
        if elapsed > 2:  # Avoid unstable initial rates
            rate = state.processed_items / elapsed

        # Calculate ETA
        eta: Optional[datetime] = None
        if rate > 0 and state.processed_items < state.total_items:
            remaining_items = state.total_items - state.processed_items
            remaining_seconds = remaining_items / rate
            eta = datetime.now() + timedelta(seconds=remaining_seconds)

        return ProgressReport(
            phase=state.current_phase,
            total_items=state.total_items,
            processed_items=state.processed_items,
            failed_items=state.failed_items,
            progress_pct=progress_pct,
            elapsed_seconds=elapsed,
            rate_per_second=rate,
            eta=eta,
            phase_durations=state.phase_durations.copy()
        )

# ─────────────────────────────────────────────────────────────────────────────
# Elite Progress Tracker (Orchestrator with Observer Pattern)
# ─────────────────────────────────────────────────────────────────────────────
class ProgressTracker:
    """Thread-safe, elite progress tracker with a complete lifecycle, observer pattern, and dependency injection."""

    def __init__(
        self,
        total_items: Optional[int] = None,
        observers: Optional[List[ProgressObserver]] = None,
        clock: Clock = MonotonicClock()
    ):
        # Handle legacy usage where total_items is set later
        self._state = ProgressState(total_items=total_items or 0)
        self._observers = observers or []
        self._clock = clock
        self._report_generator = ReportGenerator(self._clock)
        self._is_active = False
        self._lock = threading.Lock()  # For thread safety
        self._initialized = total_items is not None
        
        if total_items is not None:
            self._initialize_start_state()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
    
    def _initialize_start_state(self):
        """Initializes the start times. Must be called when total_items is known."""
        now = self._clock.now()
        self._state.start_time = now
        self._state.phase_start_time = now
        self._is_active = True

    def start(self, total_items: Optional[int] = None):
        """Explicitly starts the tracker if not already started."""
        with self._lock:
            if total_items is not None:
                self._state.total_items = total_items
                self._initialized = True
            
            if not self._is_active:
                if self._state.total_items <= 0 and self._initialized:
                    raise ValueError("Total items must be set to a positive integer before starting.")
                if not self._initialized:
                    self._setup_default_observers()
                    self._initialized = True
                self._initialize_start_state()
                self._notify_observers()
    
    def _setup_default_observers(self):
        """Set up default logging observer for legacy compatibility."""
        if not self._observers:
            default_logger = logging.getLogger("harvest.progress")
            self._observers = [LoggingObserver(default_logger, self._clock, log_interval_seconds=5)]

    def set_phase(self, phase: str):
        """Transitions to a new phase, calculating the duration of the old one."""
        with self._lock:
            if not self._is_active:
                self.start()  # Auto-start if phase is set first

            now = self._clock.now()
            if self._state.phase_start_time > 0:
                phase_duration = now - self._state.phase_start_time
                self._state.phase_durations[self._state.current_phase] = phase_duration
            
            self._state.current_phase = phase
            self._state.phase_start_time = now
            logger.info(f"🔄 Transitioned to phase: {phase}")
            self._notify_observers()

    def update(self, processed: int = 1, failed: int = 0):
        """Increments the processed items and notifies observers."""
        with self._lock:
            if not self._is_active:
                logger.debug("Tracker incremented before it was started. Ignoring.")
                return

            self._state.processed_items += processed
            self._state.failed_items += failed
            self._notify_observers()

    def finish(self):
        """Finalizes the tracking, calculates final durations, and notifies observers."""
        with self._lock:
            if not self._is_active:
                return  # Avoid multiple finish calls

            # Finalize the duration of the last active phase
            now = self._clock.now()
            if self._state.phase_start_time > 0:
                phase_duration = now - self._state.phase_start_time
                self._state.phase_durations[self._state.current_phase] = phase_duration

            self._is_active = False
            
            report = self._report_generator.generate(self._state)
            for observer in self._observers:
                observer.on_finish(report)

    def _notify_observers(self):
        """Generate report and notify all observers."""
        if not self._observers:
            return
        
        report = self._report_generator.generate(self._state)
        for observer in self._observers:
            observer.on_update(report)

    def get_current_report(self) -> ProgressReport:
        """Get current progress snapshot without notifications."""
        return self._report_generator.generate(self._state)
    
    def set_total_items(self, total_items: int):
        """Sets the total number of items, typically used if not set at init."""
        with self._lock:
            if not self._is_active:
                self._state.total_items = total_items
                self._initialized = True
                if not self._observers:
                    self._setup_default_observers()
                self._initialize_start_state()
    
    def __call__(self, processed_count: int = 1):
        """Make the tracker callable as a progress callback function."""
        if not self._initialized:
            # Auto-initialize with default observers if not already done
            self._setup_default_observers()
            self._initialized = True
            
        self.update(processed_count)

# ─────────────────────────────────────────────────────────────────────────────
# Legacy Compatibility Layer (Preserves Existing API)
# ─────────────────────────────────────────────────────────────────────────────
class LegacyProgressTracker:
    """Legacy progress tracker - PRESERVED for harvesting engine compatibility."""
    
    def __init__(self, total_files: int):
        # Set up default observers for backward compatibility
        self._logger = logging.getLogger("harvest.progress")
        observers = [LoggingObserver(self._logger, log_interval_seconds=5)]
        
        # Use the elite implementation under the hood
        self._elite_tracker = ProgressTracker(
            total_items=total_files,
            observers=observers
        )
        
        # Legacy state tracking for API compatibility
        self._total_files = total_files
        self._processed_files = 0
        self._failed_files = 0
        self._phase = "initializing"

    def __enter__(self):
        self._elite_tracker.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._elite_tracker.finish()

    def start(self):
        """Legacy start method."""
        self._elite_tracker.start()

    def update(self, processed: int = 1, failed: int = 0):
        """Legacy update method."""
        self._processed_files += processed
        self._failed_files += failed
        self._elite_tracker.update(processed, failed)

    def set_phase(self, phase: str):
        """Legacy phase setter."""
        self._phase = phase
        self._elite_tracker.set_phase(phase)

    def finish(self):
        """Legacy finish method."""
        self._elite_tracker.finish()

    def get_progress(self) -> Dict[str, Any]:
        """Legacy progress getter - returns dict for backward compatibility."""
        report = self._elite_tracker.get_current_report()
        return {
            'total_files': report.total_items,
            'processed_files': report.processed_items,
            'failed_files': report.failed_items,
            'progress_percentage': report.progress_pct,
            'current_phase': report.phase,
            'elapsed_seconds': report.elapsed_seconds,
            'rate_per_second': report.rate_per_second
        }

# ─────────────────────────────────────────────────────────────────────────────
# Factory for Easy Setup
# ─────────────────────────────────────────────────────────────────────────────
class ProgressTrackerFactory:
    """Factory for creating configured progress trackers."""
    
    @staticmethod
    def create_harvesting_tracker(total_files: int, enable_statistics: bool = False) -> ProgressTracker:
        """Create a tracker configured for harvesting operations."""
        harvest_logger = logging.getLogger("harvest.progress")
        clock = MonotonicClock()
        observers = [LoggingObserver(harvest_logger, clock, log_interval_seconds=3)]
        
        if enable_statistics:
            observers.append(StatisticsObserver())
        
        return ProgressTracker(
            total_items=total_files,
            observers=observers,
            clock=clock
        )
    
    @staticmethod
    def create_silent_tracker(total_items: int) -> ProgressTracker:
        """Create a tracker with no output - useful for testing."""
        return ProgressTracker(total_items=total_items, observers=[])

# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility Aliases
# ─────────────────────────────────────────────────────────────────────────────
# For existing imports, make ProgressTracker available at module level
# The elite ProgressTracker is the default, LegacyProgressTracker for full compatibility
__all__ = ['ProgressTracker', 'LegacyProgressTracker', 'ProgressTrackerFactory']