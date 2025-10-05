"""
Guardrails - Input/Output validation and safety checks

Run checks in parallel with agents to validate user input and agent output.
Useful for preventing misuse, ensuring quality, and enforcing policies.
"""

from typing import Optional, Callable, Any, List, Union
from dataclasses import dataclass


def create_input_guardrail(
    guardrail_fn: Callable,
    name: Optional[str] = None,
):
    """
    Create an input guardrail that validates user input.

    Input guardrails run before the agent processes the input.
    If the guardrail triggers, an exception is raised and the agent
    doesn't run (saving time and money).

    Args:
        guardrail_fn: Function that checks input and returns GuardrailFunctionOutput
        name: Optional name for the guardrail

    Returns:
        Input guardrail decorator

    Example:
        from agents import create_input_guardrail, GuardrailFunctionOutput
        from pydantic import BaseModel

        class MathCheck(BaseModel):
            is_math: bool
            reasoning: str

        @create_input_guardrail
        async def no_math_homework(ctx, agent, input):
            # Use a cheap/fast model to check
            check_agent = OpenAIAgent(
                name="Math Checker",
                model="gpt-4o-mini",
                output_type=MathCheck
            )
            result = await check_agent.run_async(str(input))

            return GuardrailFunctionOutput(
                output_info=result.final_output,
                tripwire_triggered=result.final_output.is_math
            )

        agent = OpenAIAgent(
            name="Assistant",
            input_guardrails=[no_math_homework]
        )
    """
    try:
        from agents import input_guardrail
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    return input_guardrail(guardrail_fn)


def create_output_guardrail(
    guardrail_fn: Callable,
    name: Optional[str] = None,
):
    """
    Create an output guardrail that validates agent output.

    Output guardrails run after the agent produces output.
    If the guardrail triggers, an exception is raised.

    Args:
        guardrail_fn: Function that checks output and returns GuardrailFunctionOutput
        name: Optional name for the guardrail

    Returns:
        Output guardrail decorator

    Example:
        from agents import create_output_guardrail, GuardrailFunctionOutput

        @create_output_guardrail
        async def no_profanity(ctx, agent, output):
            # Check if output contains profanity
            has_profanity = check_for_profanity(output)

            return GuardrailFunctionOutput(
                output_info={"checked": True},
                tripwire_triggered=has_profanity
            )

        agent = OpenAIAgent(
            name="Assistant",
            output_guardrails=[no_profanity]
        )
    """
    try:
        from agents import output_guardrail
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    return output_guardrail(guardrail_fn)


class CommonGuardrails:
    """
    Common guardrail patterns for typical use cases.

    Provides pre-built guardrails for common scenarios like
    content filtering, topic validation, and safety checks.
    """

    @staticmethod
    def topic_validator(
        allowed_topics: List[str],
        model: str = "gpt-4o-mini"
    ):
        """
        Create an input guardrail that validates topic relevance.

        Args:
            allowed_topics: List of allowed topics
            model: Model to use for checking (default: gpt-4o-mini for speed)

        Returns:
            Input guardrail function

        Example:
            guardrail = CommonGuardrails.topic_validator(
                allowed_topics=["billing", "refunds", "shipping"]
            )

            agent = OpenAIAgent(
                name="Support Agent",
                input_guardrails=[guardrail]
            )
        """
        from pydantic import BaseModel

        class TopicCheck(BaseModel):
            is_on_topic: bool
            detected_topic: str
            reasoning: str

        async def topic_check(ctx, agent, input):
            from agents import GuardrailFunctionOutput
            from ..openai_agent import OpenAIAgent

            topics_str = ", ".join(allowed_topics)

            checker = OpenAIAgent(
                name="Topic Checker",
                model=model,
                instructions=f"""Check if the input is about one of these topics: {topics_str}.
                Respond with whether it's on-topic and which topic was detected.""",
                output_type=TopicCheck
            )

            result = await checker.run_async(str(input))

            return GuardrailFunctionOutput(
                output_info=result.final_output,
                tripwire_triggered=not result.final_output.is_on_topic
            )

        return create_input_guardrail(topic_check)

    @staticmethod
    def content_safety_check(
        categories: Optional[List[str]] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Create an input guardrail for content safety.

        Checks for harmful, offensive, or inappropriate content.

        Args:
            categories: Specific categories to check (harassment, hate, violence, etc.)
            model: Model to use for checking

        Returns:
            Input guardrail function

        Example:
            guardrail = CommonGuardrails.content_safety_check(
                categories=["harassment", "hate_speech", "violence"]
            )
        """
        from pydantic import BaseModel

        class SafetyCheck(BaseModel):
            is_safe: bool
            flagged_categories: List[str]
            reasoning: str

        async def safety_check(ctx, agent, input):
            from agents import GuardrailFunctionOutput
            from ..openai_agent import OpenAIAgent

            categories_str = ", ".join(categories) if categories else "any harmful content"

            checker = OpenAIAgent(
                name="Safety Checker",
                model=model,
                instructions=f"""Check if the input contains {categories_str}.
                Identify any safety concerns.""",
                output_type=SafetyCheck
            )

            result = await checker.run_async(str(input))

            return GuardrailFunctionOutput(
                output_info=result.final_output,
                tripwire_triggered=not result.final_output.is_safe
            )

        return create_input_guardrail(safety_check)

    @staticmethod
    def output_quality_check(
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required_elements: Optional[List[str]] = None,
    ):
        """
        Create an output guardrail for quality validation.

        Checks output length, required elements, and formatting.

        Args:
            min_length: Minimum acceptable output length
            max_length: Maximum acceptable output length
            required_elements: List of required elements (keywords, sections, etc.)

        Returns:
            Output guardrail function

        Example:
            guardrail = CommonGuardrails.output_quality_check(
                min_length=50,
                required_elements=["reasoning", "conclusion"]
            )
        """
        async def quality_check(ctx, agent, output):
            from agents import GuardrailFunctionOutput

            output_str = str(output)
            issues = []

            if min_length and len(output_str) < min_length:
                issues.append(f"Output too short (min: {min_length})")

            if max_length and len(output_str) > max_length:
                issues.append(f"Output too long (max: {max_length})")

            if required_elements:
                for element in required_elements:
                    if element.lower() not in output_str.lower():
                        issues.append(f"Missing required element: {element}")

            return GuardrailFunctionOutput(
                output_info={"issues": issues},
                tripwire_triggered=len(issues) > 0
            )

        return create_output_guardrail(quality_check)


@dataclass
class GuardrailConfig:
    """Configuration for a guardrail."""

    name: str
    guardrail_fn: Callable
    is_input: bool
    enabled: bool = True


class GuardrailBuilder:
    """
    Builder for creating complex guardrail chains.

    Allows you to combine multiple guardrails with different
    conditions and error handling.
    """

    def __init__(self):
        self.input_guardrails = []
        self.output_guardrails = []

    def add_input_check(self, name: str, check_fn: Callable):
        """Add an input guardrail to the chain."""
        guardrail = create_input_guardrail(check_fn)
        self.input_guardrails.append((name, guardrail))
        return self

    def add_output_check(self, name: str, check_fn: Callable):
        """Add an output guardrail to the chain."""
        guardrail = create_output_guardrail(check_fn)
        self.output_guardrails.append((name, guardrail))
        return self

    def build(self):
        """Build and return the guardrail lists."""
        return {
            "input_guardrails": [g for _, g in self.input_guardrails],
            "output_guardrails": [g for _, g in self.output_guardrails]
        }


# Example usage
if __name__ == "__main__":
    from ..openai_agent import OpenAIAgent
    from pydantic import BaseModel

    # Example 1: Topic validation
    print("Example 1: Topic validation guardrail")
    topic_guardrail = CommonGuardrails.topic_validator(
        allowed_topics=["billing", "refunds", "account"]
    )

    support_agent = OpenAIAgent(
        name="Support Agent",
        instructions="Help customers with their questions",
        input_guardrails=[topic_guardrail]
    )
    print("Created support agent with topic validation")

    # Example 2: Safety check
    print("\nExample 2: Content safety guardrail")
    safety_guardrail = CommonGuardrails.content_safety_check(
        categories=["harassment", "violence"]
    )

    safe_agent = OpenAIAgent(
        name="Safe Agent",
        input_guardrails=[safety_guardrail]
    )
    print("Created agent with safety guardrail")

    # Example 3: Output quality
    print("\nExample 3: Output quality guardrail")
    quality_guardrail = CommonGuardrails.output_quality_check(
        min_length=50,
        required_elements=["reasoning", "answer"]
    )

    quality_agent = OpenAIAgent(
        name="Quality Agent",
        output_guardrails=[quality_guardrail]
    )
    print("Created agent with output quality check")

    # Example 4: Guardrail builder
    print("\nExample 4: Guardrail builder")
    builder = GuardrailBuilder()
    builder.add_input_check("topic", topic_guardrail)
    builder.add_input_check("safety", safety_guardrail)
    builder.add_output_check("quality", quality_guardrail)

    guardrails = builder.build()
    print(f"Built {len(guardrails['input_guardrails'])} input and {len(guardrails['output_guardrails'])} output guardrails")

    print("\nGuardrail examples created successfully!")
