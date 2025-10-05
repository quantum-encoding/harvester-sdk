"""
Usage Tracking - Monitor token usage and costs

Automatically tracks token usage for every agent run.
Useful for monitoring costs, enforcing limits, and analytics.
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field


@dataclass
class UsageStats:
    """
    Token usage statistics for an agent run.

    Tracks requests, tokens, and detailed breakdowns.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: 'UsageStats') -> 'UsageStats':
        """Combine usage stats from multiple runs."""
        return UsageStats(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens
        )

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens
        }


def get_usage_from_result(result) -> UsageStats:
    """
    Extract usage statistics from a RunResult.

    Args:
        result: RunResult from Runner.run()

    Returns:
        UsageStats object

    Example:
        result = await Runner.run(agent, "Hello")
        usage = get_usage_from_result(result)
        print(f"Used {usage.total_tokens} tokens")
    """
    try:
        usage = result.context_wrapper.usage

        return UsageStats(
            requests=usage.requests,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=getattr(usage.details, "cached_tokens", 0)
            if hasattr(usage, "details")
            else 0,
            reasoning_tokens=getattr(usage.details, "reasoning_tokens", 0)
            if hasattr(usage, "details")
            else 0,
        )
    except Exception as e:
        # Return empty stats if usage not available
        return UsageStats()


class UsageTracker:
    """
    Track usage across multiple agent runs.

    Useful for monitoring costs and enforcing budgets.
    """

    def __init__(self, cost_per_1k_input: float = 0.0, cost_per_1k_output: float = 0.0):
        """
        Initialize usage tracker.

        Args:
            cost_per_1k_input: Cost per 1K input tokens (USD)
            cost_per_1k_output: Cost per 1K output tokens (USD)

        Example:
            # GPT-5 pricing (example rates)
            tracker = UsageTracker(
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.06
            )
        """
        self.total_usage = UsageStats()
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.runs: list = []

    def record(self, result, metadata: Optional[Dict[str, Any]] = None):
        """
        Record usage from an agent run.

        Args:
            result: RunResult from Runner.run()
            metadata: Optional metadata to attach to this run
        """
        usage = get_usage_from_result(result)
        self.total_usage += usage
        self.runs.append({"usage": usage, "metadata": metadata or {}})

    def get_cost(self, usage: Optional[UsageStats] = None) -> float:
        """
        Calculate cost for usage.

        Args:
            usage: UsageStats to calculate cost for (default: total usage)

        Returns:
            Cost in USD
        """
        u = usage or self.total_usage
        input_cost = (u.input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (u.output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """
        Get usage summary.

        Returns:
            Dict with total usage, cost, and run count
        """
        return {
            "total_runs": len(self.runs),
            "total_tokens": self.total_usage.total_tokens,
            "input_tokens": self.total_usage.input_tokens,
            "output_tokens": self.total_usage.output_tokens,
            "cached_tokens": self.total_usage.cached_tokens,
            "reasoning_tokens": self.total_usage.reasoning_tokens,
            "total_cost_usd": self.get_cost(),
        }

    def reset(self):
        """Reset all tracked usage."""
        self.total_usage = UsageStats()
        self.runs = []


class UsageLimiter:
    """
    Enforce usage limits on agent runs.

    Prevents exceeding token or cost budgets.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_cost: Optional[float] = None,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ):
        """
        Initialize usage limiter.

        Args:
            max_tokens: Maximum total tokens allowed
            max_cost: Maximum cost allowed (USD)
            cost_per_1k_input: Cost per 1K input tokens
            cost_per_1k_output: Cost per 1K output tokens

        Example:
            limiter = UsageLimiter(
                max_tokens=100000,
                max_cost=5.0,  # $5 limit
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.06
            )

            if limiter.can_run():
                result = await agent.run_async("Hello")
                limiter.record(result)
            else:
                print("Budget exceeded!")
        """
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.tracker = UsageTracker(cost_per_1k_input, cost_per_1k_output)

    def can_run(self) -> bool:
        """
        Check if another run is within limits.

        Returns:
            True if within limits, False otherwise
        """
        if self.max_tokens and self.tracker.total_usage.total_tokens >= self.max_tokens:
            return False

        if self.max_cost and self.tracker.get_cost() >= self.max_cost:
            return False

        return True

    def record(self, result, metadata: Optional[Dict[str, Any]] = None):
        """Record usage from a run."""
        self.tracker.record(result, metadata)

    def get_remaining(self) -> Dict[str, Any]:
        """
        Get remaining budget.

        Returns:
            Dict with remaining tokens and cost
        """
        current_tokens = self.tracker.total_usage.total_tokens
        current_cost = self.tracker.get_cost()

        return {
            "remaining_tokens": self.max_tokens - current_tokens
            if self.max_tokens
            else None,
            "remaining_cost": self.max_cost - current_cost if self.max_cost else None,
            "tokens_used": current_tokens,
            "cost_used": current_cost,
        }


# Model pricing (example rates - update based on actual pricing)
MODEL_PRICING = {
    "gpt-5": {"input": 0.015, "output": 0.06},
    "gpt-5-mini": {"input": 0.003, "output": 0.012},
    "gpt-5-nano": {"input": 0.0015, "output": 0.006},
    "gpt-4.1": {"input": 0.01, "output": 0.03},
    "gpt-4.1-mini": {"input": 0.002, "output": 0.008},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing for a model.

    Args:
        model: Model name

    Returns:
        Dict with input and output prices per 1K tokens

    Example:
        pricing = get_model_pricing("gpt-5")
        tracker = UsageTracker(
            cost_per_1k_input=pricing["input"],
            cost_per_1k_output=pricing["output"]
        )
    """
    # Extract base model name (remove provider prefix if present)
    base_model = model.split("/")[-1]

    return MODEL_PRICING.get(
        base_model, {"input": 0.0, "output": 0.0}  # Default to free if unknown
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    from ..openai_agent import OpenAIAgent

    async def main():
        # Example 1: Basic usage tracking
        print("Example 1: Basic usage tracking")

        agent = OpenAIAgent(name="Assistant", model="gpt-5")

        # Simulate a run
        print("Tracking usage from agent runs...")
        print("(In real usage, you'd get actual RunResult objects)")

        # Example 2: Usage tracker
        print("\nExample 2: Usage tracker")
        pricing = get_model_pricing("gpt-5")
        tracker = UsageTracker(
            cost_per_1k_input=pricing["input"], cost_per_1k_output=pricing["output"]
        )

        print(f"Tracker initialized with GPT-5 pricing")
        print(f"Input: ${pricing['input']}/1K, Output: ${pricing['output']}/1K")

        # Example 3: Usage limiter
        print("\nExample 3: Usage limiter")
        limiter = UsageLimiter(
            max_tokens=100000,
            max_cost=5.0,
            cost_per_1k_input=pricing["input"],
            cost_per_1k_output=pricing["output"],
        )

        if limiter.can_run():
            print("Within budget - can run")
        else:
            print("Budget exceeded!")

        remaining = limiter.get_remaining()
        print(f"Remaining tokens: {remaining['remaining_tokens']}")
        print(f"Remaining cost: ${remaining['remaining_cost']:.2f}")

        print("\nUsage examples created successfully!")

    asyncio.run(main())
