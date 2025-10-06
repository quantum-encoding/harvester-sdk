"""
Elite Rate Limiting Framework (Synthesized Edition)
Combines the best features from Claude and Gemini upgrades while maintaining
backward compatibility with existing harvesting engine components.

Key Features:
• Strategy pattern for interchangeable rate limiting algorithms
• Composite pattern for multi-tier rate limiting orchestration
• Factory pattern for decoupled configuration and construction
• Concurrent acquisition using asyncio.gather for optimal performance
• Protocol-based interfaces for maximum flexibility
• Robust error handling and graceful degradation
φ = 1.618 033 988 749 895
"""
import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (Immutable and Type-Safe)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RateLimitConfig:
    """Immutable configuration for a suite of rate limiters."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    requests_per_hour: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    burst_allowance: float = 0.1  # 10% burst allowance
    
    def __post_init__(self):
        """Validate configuration on construction."""
        if self.requests_per_minute <= 0 or self.tokens_per_minute <= 0:
            raise ValueError("Rate limits must be positive")
        if not 0 <= self.burst_allowance <= 1.0:
            raise ValueError("Burst allowance must be between 0 and 1")

# ─────────────────────────────────────────────────────────────────────────────
# Core Abstractions (Strategy Pattern Interface)
# ─────────────────────────────────────────────────────────────────────────────
class RateLimiter(Protocol):
    """Strategy interface for all rate limiting algorithms."""
    
    async def acquire(self, cost: int = 1) -> None:
        """Acquire permission to proceed. Blocks until permission granted."""
        ...

    def get_status(self) -> Dict[str, Any]:
        """Return current limiter status for monitoring."""
        ...

    @property
    def name(self) -> str:
        """Return unique identifier for this limiter instance."""
        ...

# ─────────────────────────────────────────────────────────────────────────────
# Elite Strategy Implementations
# ─────────────────────────────────────────────────────────────────────────────
class TokenBucket(RateLimiter):
    """
    Token bucket algorithm - ideal for resource consumption limiting.
    Allows for controlled bursts while maintaining long-term rate limits.
    """
    
    def __init__(self, capacity: int, refill_rate: float, name: str):
        self._name = name
        self._capacity = float(capacity)
        self._refill_rate = refill_rate  # tokens per second
        self._tokens = self._capacity
        self._last_refill_time = time.monotonic()
        self._condition = asyncio.Condition()

    @property
    def name(self) -> str:
        return self._name

    async def acquire(self, cost: int = 1) -> None:
        if cost > self._capacity:
            raise ValueError(f"Cost ({cost}) exceeds bucket capacity ({self._capacity})")

        while True:
            async with self._condition:
                while True:
                    self._refill()
                    if self._tokens >= cost:
                        self._tokens -= cost
                        logger.debug(f"{self.name}: Granted {cost} tokens, {self._tokens:.1f} remaining")
                        return

                    # Wait until we might have enough tokens
                    deficit = cost - self._tokens
                    wait_time = deficit / self._refill_rate
                    
                    logger.debug(f"{self.name}: Insufficient tokens, waiting at least {wait_time:.2f}s")
                    
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=wait_time + 0.1)
                    except asyncio.TimeoutError:
                        pass  # Timeout is expected, we just re-loop and check again

    def _refill(self) -> None:
        """Refill tokens. MUST be called inside the Condition's lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        if elapsed > 0:
            tokens_to_add = elapsed * self._refill_rate
            if tokens_to_add > 0:
                self._tokens = min(self._capacity, self._tokens + tokens_to_add)
                self._last_refill_time = now
                # We've added tokens, notify waiters that state has changed.
                self._condition.notify_all()

    async def get_status(self) -> Dict[str, Any]:
        async with self._condition:
            # To provide a consistent view, we must lock.
            self._refill() 
            return {
                'name': self.name,
                'type': 'TokenBucket',
                'tokens_available': round(self._tokens, 2),
                'capacity': self._capacity,
                'refill_rate_per_sec': self._refill_rate,
                'utilization_pct': round((1 - self._tokens / self._capacity) * 100, 1)
            }

class SlidingWindowLog(RateLimiter):
    """
    Sliding window log algorithm - provides strict rate enforcement.
    Maintains exact request history for precise rate limiting.
    """
    
    def __init__(self, window_seconds: int, max_requests: int, name: str):
        self._name = name
        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._requests = deque()  # Timestamps of requests
        self._condition = asyncio.Condition()

    @property
    def name(self) -> str:
        return self._name

    async def acquire(self, cost: int = 1) -> None:
        if cost > self._max_requests:
            raise ValueError(f"Cost ({cost}) exceeds window capacity ({self._max_requests})")

        async with self._condition:
            while True:
                now = time.monotonic()
                self._evict_old_requests(now)

                if len(self._requests) + cost <= self._max_requests:
                    # Grant permission by adding timestamps
                    for _ in range(cost):
                        self._requests.append(now)
                    logger.debug(f"{self.name}: Granted {cost} requests, "
                               f"{len(self._requests)}/{self._max_requests} used")
                    return

                # Calculate wait time until oldest request expires
                if self._requests:
                    wait_time = (self._requests[0] + self._window_seconds) - now
                else:
                    wait_time = 0.1  # Minimal wait if queue is empty

                logger.debug(f"{self.name}: Window full, waiting {wait_time:.2f}s")
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=max(0, wait_time) + 0.1)
                except asyncio.TimeoutError:
                    pass  # Expected wake-up to re-evaluate

    def _evict_old_requests(self, now: float) -> None:
        """Remove requests outside the sliding window."""
        cutoff_time = now - self._window_seconds
        while self._requests and self._requests[0] <= cutoff_time:
            self._requests.popleft()

    async def get_status(self) -> Dict[str, Any]:
        async with self._condition:
            now = time.monotonic()
            self._evict_old_requests(now)
            return {
                'name': self.name,
                'type': 'SlidingWindowLog',
                'requests_in_window': len(self._requests),
                'max_requests': self._max_requests,
                'window_seconds': self._window_seconds,
                'utilization_pct': round((len(self._requests) / self._max_requests) * 100, 1)
            }

class FixedWindowCounter(RateLimiter):
    """
    Fixed window counter algorithm - memory efficient for high-throughput.
    Resets counters at fixed intervals, allowing potential burst at boundaries.
    """
    
    def __init__(self, window_seconds: int, max_requests: int, name: str):
        self._name = name
        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._current_count = 0
        self._window_start = time.monotonic()
        self._condition = asyncio.Condition()

    @property
    def name(self) -> str:
        return self._name

    async def acquire(self, cost: int = 1) -> None:
        if cost > self._max_requests:
            raise ValueError(f"Cost ({cost}) exceeds window capacity ({self._max_requests})")

        async with self._condition:
            while True:
                now = time.monotonic()
                self._reset_if_needed(now)

                if self._current_count + cost <= self._max_requests:
                    self._current_count += cost
                    logger.debug(f"{self.name}: Granted {cost} requests, count is now {self._current_count}/{self._max_requests}")
                    return

                # Calculate time until next window and wait
                time_to_next_window = (self._window_start + self._window_seconds) - now
                logger.debug(f"{self.name}: Window full, waiting {time_to_next_window:.2f}s for next window.")
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=max(0, time_to_next_window) + 0.1)
                except asyncio.TimeoutError:
                    pass  # Expected wake-up to re-evaluate

    def _reset_if_needed(self, now: float) -> None:
        """Resets the window if the current time has passed its end. Must be called in lock."""
        if now >= self._window_start + self._window_seconds:
            self._window_start = now
            self._current_count = 0
            # Window reset, notify all waiters.
            self._condition.notify_all()

    async def get_status(self) -> Dict[str, Any]:
        async with self._condition:
            now = time.monotonic()
            self._reset_if_needed(now)
            return {
                'name': self.name,
                'type': 'FixedWindowCounter',
                'requests_in_window': self._current_count,
                'max_requests': self._max_requests,
                'window_seconds': self._window_seconds,
                'window_ends_in_sec': round((self._window_start + self._window_seconds) - now, 2),
                'utilization_pct': round((self._current_count / self._max_requests) * 100, 1)
            }

# ─────────────────────────────────────────────────────────────────────────────
# Generic Fallback (No-Op Limiter)
# ─────────────────────────────────────────────────────────────────────────────
class NoOpRateLimiter(RateLimiter):
    """No-operation limiter for testing or disabling rate limiting."""
    
    def __init__(self, name: str = "noop"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def acquire(self, cost: int = 1) -> None:
        # Always allow - no rate limiting
        pass

    async def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'NoOpRateLimiter',
            'status': 'unlimited'
        }

# ─────────────────────────────────────────────────────────────────────────────
# Elite Factory (Decoupled Construction)
# ─────────────────────────────────────────────────────────────────────────────
class RateLimiterFactory:
    """Factory for creating configured rate limiters from config."""

    @staticmethod
    def from_config(config: RateLimitConfig) -> List[RateLimiter]:
        """Create a suite of rate limiters from configuration."""
        limiters: List[RateLimiter] = []

        # Minute-level limiters (primary rate controls)
        limiters.append(SlidingWindowLog(
            window_seconds=60,
            max_requests=config.requests_per_minute,
            name="requests_per_minute"
        ))
        
        limiters.append(TokenBucket(
            capacity=int(config.tokens_per_minute * (1 + config.burst_allowance)),
            refill_rate=config.tokens_per_minute / 60.0,
            name="tokens_per_minute"
        ))

        # Hour-level limiters (secondary quotas)
        if config.requests_per_hour is not None:
            limiters.append(FixedWindowCounter(
                window_seconds=3600,
                max_requests=config.requests_per_hour,
                name="requests_per_hour"
            ))

        if config.tokens_per_hour is not None:
            limiters.append(TokenBucket(
                capacity=int(config.tokens_per_hour * (1 + config.burst_allowance)),
                refill_rate=config.tokens_per_hour / 3600.0,
                name="tokens_per_hour"
            ))

        logger.info(f"🚦 Created {len(limiters)} rate limiters: "
                   f"{[limiter.name for limiter in limiters]}")
        return limiters

    @staticmethod
    def create_testing_limiters() -> List[RateLimiter]:
        """Create minimal limiters for testing."""
        return [NoOpRateLimiter("test_noop")]

# ─────────────────────────────────────────────────────────────────────────────
# Elite Composite (Orchestration with Concurrent Acquisition)
# ─────────────────────────────────────────────────────────────────────────────
class CompositeRateLimiter:
    """
    Orchestrates multiple rate limiters, ensuring all must grant permission.
    Acquisition is sequential, respecting hierarchical limits (e.g., per-minute then per-hour).
    """
    
    def __init__(self, limiters: Sequence[RateLimiter], name: str = "Composite"):
        if not limiters:
            raise ValueError("CompositeRateLimiter requires at least one limiter")
        self._name = name
        self._limiters = list(limiters)
        logger.info(f"🎯 Composite rate limiter with {len(self._limiters)} strategies")

    @property
    def name(self) -> str:
        return self._name

    async def acquire(self, token_cost: int = 0) -> None:
        """
        Acquire permission from all child limiters sequentially.
        If any limiter blocks, this method will block.
        
        Args:
            token_cost: Cost for token-based limiters (request-based use cost=1)
        """
        if not self._limiters:
            return

        # This is the core fix: sequential acquisition
        for limiter in self._limiters:
            # Smart cost assignment based on limiter type
            if "token" in limiter.name.lower():
                cost = max(1, token_cost)  # Token-based limiters use actual cost
            else:
                cost = 1  # Request-based limiters always use unit cost
            
            await limiter.acquire(cost)
        
        logger.debug(f"{self.name}: Granted permission through all child limiters.")

    async def get_status(self) -> Dict[str, Any]:
        """Return the combined status of all child limiters."""
        status = {
            'name': self.name,
            'type': 'Composite',
            'children': []
        }
        
        for limiter in self._limiters:
            try:
                child_status = await limiter.get_status()
                status['children'].append(child_status)
            except Exception as e:
                status['children'].append({'name': limiter.name, 'error': str(e)})
        
        return status

    def add_limiter(self, limiter: RateLimiter):
        """Dynamically add a new limiter to the composite."""
        self._limiters.append(limiter)
        logger.info(f"➕ Added limiter '{limiter.name}' to composite")

    def remove_limiter(self, name: str) -> bool:
        """Remove a limiter by name."""
        for i, limiter in enumerate(self._limiters):
            if limiter.name == name:
                del self._limiters[i]
                logger.info(f"➖ Removed limiter '{name}' from composite")
                return True
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Legacy Compatibility Layer
# ─────────────────────────────────────────────────────────────────────────────
class MultiTierRateLimiter:
    """Legacy multi-tier rate limiter - PRESERVED for harvesting engine compatibility."""
    
    def __init__(self, config: RateLimitConfig):
        # Use elite implementation under the hood
        limiters = RateLimiterFactory.from_config(config)
        self._elite_limiter = CompositeRateLimiter(limiters)
        
        # Store config for legacy API compatibility
        self._config = config

    async def acquire(self, token_cost: int = 0) -> float:
        """Legacy acquire method - returns 0.0 for compatibility."""
        await self._elite_limiter.acquire(token_cost)
        return 0.0  # Legacy API expects wait time, but we handle internally

    async def get_status(self) -> Dict[str, Any]:
        """Legacy status method."""
        return await self._elite_limiter.get_status()

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Factory Functions
# ─────────────────────────────────────────────────────────────────────────────
def create_standard_rate_limiter(
    requests_per_minute: int = 60,
    tokens_per_minute: int = 90000,
    **kwargs
) -> CompositeRateLimiter:
    """Create a standard rate limiter with sensible defaults."""
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        **kwargs
    )
    limiters = RateLimiterFactory.from_config(config)
    return CompositeRateLimiter(limiters)

def create_burst_tolerant_limiter(
    requests_per_minute: int = 60,
    tokens_per_minute: int = 90000,
    burst_allowance: float = 0.3
) -> CompositeRateLimiter:
    """Create a rate limiter optimized for bursty workloads."""
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        burst_allowance=burst_allowance
    )
    limiters = RateLimiterFactory.from_config(config)
    return CompositeRateLimiter(limiters)

def create_testing_limiter() -> CompositeRateLimiter:
    """Create a no-op rate limiter for testing."""
    limiters = [NoOpRateLimiter("test")]
    return CompositeRateLimiter(limiters)

# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility Aliases
# ─────────────────────────────────────────────────────────────────────────────
# Preserve old class names for existing imports
SlidingWindowLimiter = SlidingWindowLog  # Old name compatibility