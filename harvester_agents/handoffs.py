"""
Handoffs - Agent delegation and task routing

Allows agents to delegate tasks to specialized agents.
Useful for multi-agent systems where agents have different expertise.
"""

from typing import Optional, Callable, Any, Union, List
from dataclasses import dataclass


def create_handoff(
    agent,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    on_handoff: Optional[Callable] = None,
    input_type: Optional[type] = None,
    input_filter: Optional[Callable] = None,
    is_enabled: Union[bool, Callable] = True,
):
    """
    Create a handoff to another agent.

    Handoffs allow agents to delegate tasks to specialized agents.
    They appear as tools to the LLM (e.g., "transfer_to_billing_agent").

    Args:
        agent: Agent or OpenAIAgent to hand off to
        tool_name: Override default tool name (default: "transfer_to_<agent_name>")
        tool_description: Override default tool description
        on_handoff: Callback executed when handoff is invoked
        input_type: Type of input data expected from LLM (e.g., Pydantic model)
        input_filter: Function to filter/modify input sent to new agent
        is_enabled: Whether handoff is enabled (bool or function)

    Returns:
        Handoff object

    Example:
        from agents import OpenAIAgent, create_handoff

        # Basic handoff
        billing_agent = OpenAIAgent(name="Billing", instructions="Handle billing")
        refund_agent = OpenAIAgent(name="Refund", instructions="Handle refunds")

        triage = OpenAIAgent(
            name="Triage",
            instructions="Route customer requests",
            handoffs=[
                billing_agent,  # Simple handoff
                create_handoff(  # Customized handoff
                    refund_agent,
                    tool_name="escalate_to_refunds",
                    on_handoff=lambda ctx: print("Refund handoff!")
                )
            ]
        )
    """
    try:
        from agents import handoff
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    # Handle OpenAIAgent wrapper
    from .openai_agent import OpenAIAgent
    if isinstance(agent, OpenAIAgent):
        raw_agent = agent.agent
    else:
        raw_agent = agent

    kwargs = {"agent": raw_agent}

    if tool_name:
        kwargs["tool_name_override"] = tool_name
    if tool_description:
        kwargs["tool_description_override"] = tool_description
    if on_handoff:
        kwargs["on_handoff"] = on_handoff
    if input_type:
        kwargs["input_type"] = input_type
    if input_filter:
        kwargs["input_filter"] = input_filter
    if is_enabled is not True:
        kwargs["is_enabled"] = is_enabled

    return handoff(**kwargs)


class HandoffFilters:
    """
    Common input filters for handoffs.

    Filters modify the conversation history that gets passed
    to the next agent during a handoff.
    """

    @staticmethod
    def remove_all_tools():
        """
        Remove all tool calls from the conversation history.

        Useful when you want the next agent to start fresh
        without seeing previous tool interactions.

        Returns:
            Input filter function

        Example:
            handoff = create_handoff(
                agent,
                input_filter=HandoffFilters.remove_all_tools()
            )
        """
        try:
            from agents.extensions import handoff_filters
            return handoff_filters.remove_all_tools
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

    @staticmethod
    def keep_only_user_messages():
        """
        Keep only user messages in the conversation history.

        Removes all assistant messages and tool interactions.

        Returns:
            Input filter function
        """
        def filter_fn(input_data):
            from agents import HandoffInputData

            filtered_items = [
                item for item in input_data.input
                if item.get("role") == "user"
            ]

            return HandoffInputData(
                input=filtered_items,
                agent=input_data.agent
            )

        return filter_fn

    @staticmethod
    def custom_filter(filter_fn: Callable):
        """
        Create a custom input filter.

        Args:
            filter_fn: Function that receives HandoffInputData and returns
                      modified HandoffInputData

        Returns:
            Input filter function

        Example:
            def my_filter(input_data):
                # Custom logic to modify input_data
                return input_data

            handoff = create_handoff(
                agent,
                input_filter=HandoffFilters.custom_filter(my_filter)
            )
        """
        return filter_fn


class HandoffPrompts:
    """
    Recommended prompts for agents with handoffs.

    Including handoff instructions in your agent prompts helps
    the LLM understand when and how to use handoffs.
    """

    @staticmethod
    def get_recommended_prefix() -> str:
        """
        Get the recommended prompt prefix for agents with handoffs.

        Returns:
            Prompt prefix string

        Example:
            from agents import OpenAIAgent, HandoffPrompts

            agent = OpenAIAgent(
                name="Triage",
                instructions=f'''{HandoffPrompts.get_recommended_prefix()}

                You are a customer service triage agent.
                Route requests to the appropriate specialist.
                '''
            )
        """
        try:
            from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
            return RECOMMENDED_PROMPT_PREFIX
        except ImportError:
            return """You can transfer conversations to other agents when appropriate.
Use the transfer tools when you encounter requests outside your expertise."""

    @staticmethod
    def with_handoff_instructions(base_instructions: str) -> str:
        """
        Add handoff instructions to existing prompt.

        Args:
            base_instructions: Your base agent instructions

        Returns:
            Instructions with handoff guidance prepended

        Example:
            instructions = HandoffPrompts.with_handoff_instructions(
                "You are a billing agent. Help with invoices and payments."
            )
        """
        try:
            from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
            return prompt_with_handoff_instructions(base_instructions)
        except ImportError:
            prefix = HandoffPrompts.get_recommended_prefix()
            return f"{prefix}\n\n{base_instructions}"


@dataclass
class HandoffConfig:
    """Configuration for a handoff."""

    agent: Any
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None
    on_handoff: Optional[Callable] = None
    input_type: Optional[type] = None
    input_filter: Optional[Callable] = None
    is_enabled: Union[bool, Callable] = True


def create_handoff_with_data(
    agent,
    data_model: type,
    on_handoff: Optional[Callable] = None,
    **kwargs
):
    """
    Create a handoff that requires structured data from the LLM.

    Useful when you want the LLM to provide context when handing off.

    Args:
        agent: Agent to hand off to
        data_model: Pydantic model defining required data
        on_handoff: Callback to handle the data
        **kwargs: Additional handoff parameters

    Returns:
        Handoff object

    Example:
        from pydantic import BaseModel
        from agents import OpenAIAgent, create_handoff_with_data

        class EscalationData(BaseModel):
            reason: str
            priority: str

        async def log_escalation(ctx, data: EscalationData):
            print(f"Escalation: {data.reason} (Priority: {data.priority})")

        escalation_agent = OpenAIAgent(name="Escalation")

        handoff = create_handoff_with_data(
            escalation_agent,
            data_model=EscalationData,
            on_handoff=log_escalation
        )
    """
    return create_handoff(
        agent,
        input_type=data_model,
        on_handoff=on_handoff,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    from .openai_agent import OpenAIAgent
    from pydantic import BaseModel

    # Example 1: Basic handoffs
    print("Example 1: Basic handoffs")
    billing_agent = OpenAIAgent(
        name="Billing Agent",
        instructions="Handle billing questions"
    )

    refund_agent = OpenAIAgent(
        name="Refund Agent",
        instructions="Handle refund requests"
    )

    triage_agent = OpenAIAgent(
        name="Triage Agent",
        instructions=HandoffPrompts.with_handoff_instructions(
            "Route customer requests to the right specialist"
        ),
        handoffs=[billing_agent, refund_agent]
    )

    print(f"Created triage agent with {len(triage_agent.agent.handoffs)} handoffs")

    # Example 2: Handoff with data
    print("\nExample 2: Handoff with data")

    class EscalationData(BaseModel):
        reason: str
        severity: str

    async def on_escalation(ctx, data: EscalationData):
        print(f"Escalation: {data.reason} (Severity: {data.severity})")

    escalation_agent = OpenAIAgent(
        name="Escalation Agent",
        instructions="Handle escalated issues"
    )

    handoff_with_data = create_handoff_with_data(
        escalation_agent,
        data_model=EscalationData,
        on_handoff=on_escalation
    )

    print("Created handoff with structured data requirement")

    # Example 3: Conditional handoff
    print("\nExample 3: Conditional handoff")

    def is_business_hours(ctx, agent) -> bool:
        from datetime import datetime
        hour = datetime.now().hour
        return 9 <= hour <= 17

    after_hours_agent = OpenAIAgent(
        name="After Hours Agent",
        instructions="Handle requests outside business hours"
    )

    conditional_handoff = create_handoff(
        after_hours_agent,
        is_enabled=is_business_hours
    )

    print("Created conditional handoff (business hours)")

    print("\nHandoff examples created successfully!")
