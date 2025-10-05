"""
Agent Orchestration - Patterns for coordinating multiple agents

Provides utilities for orchestrating agent workflows:
- LLM-driven orchestration (autonomous planning)
- Code-driven orchestration (deterministic flows)
- Parallel execution
- Sequential chaining
- Evaluation loops
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class AgentChainResult:
    """Result from a chain of agents."""

    steps: List[Dict[str, Any]]
    final_output: Any
    metadata: Dict[str, Any]


class AgentChain:
    """
    Chain multiple agents sequentially.

    Output of one agent becomes input to the next.
    Useful for decomposing complex tasks into steps.
    """

    def __init__(self, agents: List[Any], transform_fn: Optional[Callable] = None):
        """
        Initialize an agent chain.

        Args:
            agents: List of agents to chain
            transform_fn: Optional function to transform output between agents

        Example:
            # Research -> Outline -> Write -> Edit
            research_agent = OpenAIAgent(name="Research")
            outline_agent = OpenAIAgent(name="Outline")
            write_agent = OpenAIAgent(name="Write")
            edit_agent = OpenAIAgent(name="Edit")

            chain = AgentChain([
                research_agent,
                outline_agent,
                write_agent,
                edit_agent
            ])

            result = await chain.run("Write a blog post about AI")
        """
        self.agents = agents
        self.transform_fn = transform_fn or (lambda x: x)

    async def run(self, initial_input: str, context: Optional[Any] = None) -> AgentChainResult:
        """
        Run the agent chain.

        Args:
            initial_input: Initial input to first agent
            context: Optional context object

        Returns:
            AgentChainResult with all steps and final output
        """
        from ..openai_agent import OpenAIAgent

        steps = []
        current_input = initial_input

        for i, agent in enumerate(self.agents):
            # Handle OpenAIAgent wrapper
            if isinstance(agent, OpenAIAgent):
                result = await agent.run_async(current_input, context=context)
                output = result.final_output
            else:
                # Raw Agent from SDK
                from agents import Runner
                result = await Runner.run(agent, current_input, context=context)
                output = result.final_output

            steps.append({
                "agent": agent.name if hasattr(agent, "name") else f"Agent {i}",
                "input": current_input,
                "output": output
            })

            # Transform output for next agent
            current_input = self.transform_fn(output)

        return AgentChainResult(
            steps=steps,
            final_output=output,
            metadata={"total_steps": len(steps)}
        )


class ParallelAgents:
    """
    Run multiple agents in parallel.

    Useful when you have independent tasks that don't depend on each other.
    Significantly faster than sequential execution.
    """

    @staticmethod
    async def run_all(
        agents: List[Any],
        inputs: Union[str, List[str]],
        context: Optional[Any] = None
    ) -> List[Any]:
        """
        Run multiple agents in parallel.

        Args:
            agents: List of agents to run
            inputs: Single input for all agents, or list of inputs per agent
            context: Optional context object

        Returns:
            List of outputs from each agent

        Example:
            # Translate to multiple languages simultaneously
            spanish = OpenAIAgent(name="Spanish", instructions="Translate to Spanish")
            french = OpenAIAgent(name="French", instructions="Translate to French")
            german = OpenAIAgent(name="German", instructions="Translate to German")

            results = await ParallelAgents.run_all(
                [spanish, french, german],
                "Hello, how are you?"
            )
        """
        from ..openai_agent import OpenAIAgent

        # Normalize inputs
        if isinstance(inputs, str):
            inputs = [inputs] * len(agents)

        # Create tasks
        tasks = []
        for agent, input_text in zip(agents, inputs):
            if isinstance(agent, OpenAIAgent):
                tasks.append(agent.run_async(input_text, context=context))
            else:
                from agents import Runner
                tasks.append(Runner.run(agent, input_text, context=context))

        # Run in parallel
        results = await asyncio.gather(*tasks)

        # Extract final outputs
        return [r.final_output for r in results]


class EvaluationLoop:
    """
    Run an agent in a loop with an evaluator until criteria are met.

    The worker agent performs the task, the evaluator judges the output,
    and the loop continues until the evaluator approves or max iterations reached.
    """

    def __init__(
        self,
        worker_agent: Any,
        evaluator_agent: Any,
        max_iterations: int = 5,
        feedback_prompt: str = "Improve this based on feedback: {feedback}\n\nOriginal: {output}"
    ):
        """
        Initialize an evaluation loop.

        Args:
            worker_agent: Agent that performs the task
            evaluator_agent: Agent that evaluates the output
            max_iterations: Maximum iterations before stopping
            feedback_prompt: Template for feedback to worker

        Example:
            writer = OpenAIAgent(
                name="Writer",
                instructions="Write high-quality content"
            )

            evaluator = OpenAIAgent(
                name="Evaluator",
                instructions="Evaluate quality. Respond with 'APPROVED' or feedback.",
                model="gpt-4o-mini"  # Cheaper model for evaluation
            )

            loop = EvaluationLoop(writer, evaluator, max_iterations=3)
            result = await loop.run("Write a blog post about AI")
        """
        self.worker_agent = worker_agent
        self.evaluator_agent = evaluator_agent
        self.max_iterations = max_iterations
        self.feedback_prompt = feedback_prompt

    async def run(self, initial_input: str, context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run the evaluation loop.

        Returns:
            Dict with final output, iterations, and feedback history
        """
        from ..openai_agent import OpenAIAgent

        current_input = initial_input
        iterations = []

        for i in range(self.max_iterations):
            # Worker produces output
            if isinstance(self.worker_agent, OpenAIAgent):
                worker_result = await self.worker_agent.run_async(current_input, context=context)
            else:
                from agents import Runner
                worker_result = await Runner.run(self.worker_agent, current_input, context=context)

            output = worker_result.final_output

            # Evaluator judges output
            eval_input = f"Evaluate this output:\n\n{output}"

            if isinstance(self.evaluator_agent, OpenAIAgent):
                eval_result = await self.evaluator_agent.run_async(eval_input, context=context)
            else:
                from agents import Runner
                eval_result = await Runner.run(self.evaluator_agent, eval_input, context=context)

            feedback = eval_result.final_output

            iterations.append({
                "iteration": i + 1,
                "output": output,
                "feedback": feedback
            })

            # Check if approved
            if "APPROVED" in str(feedback).upper():
                return {
                    "final_output": output,
                    "iterations": iterations,
                    "approved": True,
                    "total_iterations": i + 1
                }

            # Prepare next iteration with feedback
            current_input = self.feedback_prompt.format(
                feedback=feedback,
                output=output
            )

        return {
            "final_output": output,
            "iterations": iterations,
            "approved": False,
            "total_iterations": self.max_iterations
        }


class ClassificationRouter:
    """
    Route tasks to specialized agents based on classification.

    Uses structured output to classify the task, then routes
    to the appropriate specialized agent.
    """

    def __init__(
        self,
        classifier_agent: Any,
        routes: Dict[str, Any],
        default_agent: Optional[Any] = None
    ):
        """
        Initialize a classification router.

        Args:
            classifier_agent: Agent that classifies tasks
            routes: Dict mapping categories to agents
            default_agent: Fallback agent if category not found

        Example:
            from pydantic import BaseModel

            class TaskCategory(BaseModel):
                category: str  # "billing", "technical", "general"
                confidence: float

            classifier = OpenAIAgent(
                name="Classifier",
                output_type=TaskCategory,
                model="gpt-4o-mini"  # Fast classifier
            )

            router = ClassificationRouter(
                classifier,
                routes={
                    "billing": billing_agent,
                    "technical": tech_support_agent,
                    "general": general_agent
                }
            )

            result = await router.route("I have a billing question")
        """
        self.classifier_agent = classifier_agent
        self.routes = routes
        self.default_agent = default_agent

    async def route(self, user_input: str, context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Classify and route the task to appropriate agent.

        Returns:
            Dict with category, agent used, and final output
        """
        from ..openai_agent import OpenAIAgent

        # Classify
        if isinstance(self.classifier_agent, OpenAIAgent):
            classifier_result = await self.classifier_agent.run_async(user_input, context=context)
        else:
            from agents import Runner
            classifier_result = await Runner.run(self.classifier_agent, user_input, context=context)

        category = classifier_result.final_output

        # Get category string
        if hasattr(category, "category"):
            category_str = category.category
        else:
            category_str = str(category)

        # Route to appropriate agent
        agent = self.routes.get(category_str, self.default_agent)

        if agent is None:
            raise ValueError(f"No agent found for category '{category_str}' and no default agent provided")

        # Run the specialized agent
        if isinstance(agent, OpenAIAgent):
            result = await agent.run_async(user_input, context=context)
        else:
            from agents import Runner
            result = await Runner.run(agent, user_input, context=context)

        return {
            "category": category_str,
            "agent": agent.name if hasattr(agent, "name") else "Unknown",
            "output": result.final_output,
            "classification_result": category
        }


# Example usage
if __name__ == "__main__":
    from ..openai_agent import OpenAIAgent

    async def main():
        # Example 1: Sequential chain
        print("Example 1: Sequential agent chain")
        research = OpenAIAgent(name="Research", instructions="Research topics")
        outline = OpenAIAgent(name="Outline", instructions="Create outlines")
        writer = OpenAIAgent(name="Writer", instructions="Write content")

        chain = AgentChain([research, outline, writer])
        print("Created 3-agent chain: Research -> Outline -> Write")

        # Example 2: Parallel execution
        print("\nExample 2: Parallel agents")
        spanish = OpenAIAgent(name="Spanish", instructions="Translate to Spanish")
        french = OpenAIAgent(name="French", instructions="Translate to French")

        print("Created parallel translation agents")

        # Example 3: Evaluation loop
        print("\nExample 3: Evaluation loop")
        writer = OpenAIAgent(name="Writer", instructions="Write quality content")
        evaluator = OpenAIAgent(
            name="Evaluator",
            instructions="Evaluate quality",
            model="gpt-4o-mini"
        )

        loop = EvaluationLoop(writer, evaluator, max_iterations=3)
        print("Created evaluation loop with max 3 iterations")

        # Example 4: Classification router
        print("\nExample 4: Classification router")
        from pydantic import BaseModel

        class Category(BaseModel):
            category: str

        classifier = OpenAIAgent(
            name="Classifier",
            output_type=Category,
            model="gpt-4o-mini"
        )

        billing = OpenAIAgent(name="Billing")
        tech = OpenAIAgent(name="Technical")

        router = ClassificationRouter(
            classifier,
            routes={"billing": billing, "technical": tech}
        )
        print("Created classification router with 2 routes")

        print("\nOrchestration examples created successfully!")

    asyncio.run(main())
