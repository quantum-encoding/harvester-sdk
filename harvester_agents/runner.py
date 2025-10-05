"""Agent runner for executing workflows with automatic conversation management."""

import asyncio
from typing import Any, Optional, List, TypeVar, Generic, Callable
from dataclasses import dataclass, field


TContext = TypeVar('TContext')
DEFAULT_MAX_TURNS = 100


class TResponseInputItem:
    """Type for response input items."""
    pass


class Session:
    """Protocol for session implementations."""

    async def get_items(self, limit: Optional[int] = None) -> List[dict]:
        """Retrieve conversation history."""
        raise NotImplementedError

    async def add_items(self, items: List[dict]) -> None:
        """Add items to conversation history."""
        raise NotImplementedError

    async def pop_item(self) -> Optional[dict]:
        """Remove and return the most recent item."""
        raise NotImplementedError

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        raise NotImplementedError


class Agent(Generic[TContext]):
    """Base Agent class."""

    def __init__(self, name: str, instructions: str, output_type: Any = None):
        self.name = name
        self.instructions = instructions
        self.output_type = output_type


class RunHooks(Generic[TContext]):
    """Lifecycle hooks for agent runs."""

    def on_turn_start(self, agent: Agent, turn: int):
        """Called at the start of each turn."""
        pass

    def on_turn_end(self, agent: Agent, turn: int, result: Any):
        """Called at the end of each turn."""
        pass

    def on_handoff(self, from_agent: Agent, to_agent: Agent):
        """Called when a handoff occurs."""
        pass

    def on_tool_call(self, tool_name: str, args: dict):
        """Called when a tool is called."""
        pass


class Model:
    """Base model class."""
    pass


class ModelProvider:
    """Base model provider class."""

    def resolve_model(self, model_name: str) -> Model:
        """Resolve a model name to a Model instance."""
        raise NotImplementedError


class MultiProvider(ModelProvider):
    """Multi-provider implementation."""
    pass


class ModelSettings:
    """Model configuration settings."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class InputGuardrail(Generic[TContext]):
    """Input guardrail for validation."""
    pass


class OutputGuardrail(Generic[TContext]):
    """Output guardrail for validation."""
    pass


HandoffInputFilter = Callable[[List[dict]], List[dict]]
SessionInputCallback = Callable[[List[dict], List[dict]], List[dict]]
CallModelInputFilter = Callable[[Agent, TContext, dict], dict]


def _default_trace_include_sensitive_data() -> bool:
    """Default value for trace_include_sensitive_data."""
    return True


@dataclass
class RunConfig:
    """Configures settings for the entire agent run."""

    model: Optional[str | Model] = None
    """The model to use for the entire agent run. If set, will override the model
    set on every agent. The model_provider passed in below must be able to resolve
    this model name."""

    model_provider: ModelProvider = field(default_factory=MultiProvider)
    """The model provider to use when looking up string model names. Defaults to OpenAI."""

    model_settings: Optional[ModelSettings] = None
    """Configure global model settings. Any non-null values will override the
    agent-specific model settings."""

    handoff_input_filter: Optional[HandoffInputFilter] = None
    """A global input filter to apply to all handoffs. If Handoff.input_filter is set,
    then that will take precedence. The input filter allows you to edit the inputs that
    are sent to the new agent."""

    input_guardrails: Optional[List[InputGuardrail[Any]]] = None
    """A list of input guardrails to run on the initial run input."""

    output_guardrails: Optional[List[OutputGuardrail[Any]]] = None
    """A list of output guardrails to run on the final output of the run."""

    tracing_disabled: bool = False
    """Whether tracing is disabled for the agent run. If disabled, we will not trace
    the agent run."""

    trace_include_sensitive_data: bool = field(
        default_factory=_default_trace_include_sensitive_data
    )
    """Whether we include potentially sensitive data (for example: inputs/outputs of
    tool calls or LLM generations) in traces. If False, we'll still create spans for
    these events, but the sensitive data will not be included."""

    workflow_name: str = 'Agent workflow'
    """The name of the run, used for tracing. Should be a logical name for the run,
    like "Code generation workflow" or "Customer support agent"."""

    trace_id: Optional[str] = None
    """A custom trace ID to use for tracing. If not provided, we will generate a new
    trace ID."""

    group_id: Optional[str] = None
    """A grouping identifier to use for tracing, to link multiple traces from the same
    conversation or process. For example, you might use a chat thread ID."""

    trace_metadata: Optional[dict[str, Any]] = None
    """An optional dictionary of additional metadata to include with the trace."""

    session_input_callback: Optional[SessionInputCallback] = None
    """Defines how to handle session history when new input is provided.
    - None (default): The new input is appended to the session history.
    - SessionInputCallback: A custom function that receives the history and new input,
      and returns the desired combined list of items."""

    call_model_input_filter: Optional[CallModelInputFilter] = None
    """Optional callback that is invoked immediately before calling the model. It receives
    the current agent, context and the model input (instructions and input items), and must
    return a possibly modified ModelInputData to use for the model call.

    This allows you to edit the input sent to the model e.g. to stay within a token limit.
    For example, you can use this to add a system prompt to the input."""


@dataclass
class RunResult:
    """Result of a completed agent run."""

    output: Any
    """The final output from the agent."""

    inputs: List[dict]
    """All inputs provided during the run."""

    guardrail_results: List[dict]
    """Results from any guardrails that were executed."""

    agent: Agent
    """The agent that produced the final output."""

    turns: int
    """Number of turns taken."""

    response_id: Optional[str] = None
    """The response ID from the final model call."""

    conversation_id: Optional[str] = None
    """The conversation ID if one was used."""


class RunResultStreaming:
    """Streaming result for agent runs."""

    def __init__(self, run_task: asyncio.Task):
        self._run_task = run_task
        self._events = []

    async def stream_events(self):
        """Stream events as they are generated."""
        # Placeholder for streaming implementation
        yield {"type": "start"}


class MaxTurnsExceeded(Exception):
    """Raised when max_turns is exceeded."""

    def __init__(self, max_turns: int):
        super().__init__(f"Maximum number of turns ({max_turns}) exceeded")
        self.max_turns = max_turns


class GuardrailTripwireTriggered(Exception):
    """Raised when a guardrail tripwire is triggered."""

    def __init__(self, message: str):
        super().__init__(message)


class Runner:
    """Runner for executing agent workflows with automatic conversation management."""

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | List[TResponseInputItem],
        *,
        context: Optional[TContext] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Optional[RunHooks[TContext]] = None,
        run_config: Optional[RunConfig] = None,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> RunResult:
        """Run a workflow starting at the given agent.

        The agent will run in a loop until a final output is generated. The loop runs like so:

        1. The agent is invoked with the given input.
        2. If there is a final output (i.e. the agent produces something of type
           agent.output_type), the loop terminates.
        3. If there's a handoff, we run the loop again, with the new agent.
        4. Else, we run tool calls (if any), and re-run the loop.

        In two cases, the agent may raise an exception:

        - If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
        - If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.

        Note:
            Only the first agent's input guardrails are run.

        Args:
            starting_agent: The starting agent to run.
            input: The initial input to the agent. You can pass a single string for a user
                message, or a list of input items.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for. A turn is defined
                as one AI invocation (including any tool calls that might occur).
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.
            previous_response_id: The ID of the previous response. If using OpenAI models via
                the Responses API, this allows you to skip passing in input from the previous turn.
            conversation_id: The conversation ID. If provided, the conversation will be used to
                read and write items. Every agent will have access to the conversation history so
                far, and its output items will be written to the conversation.
            session: A session for automatic conversation history management.

        Returns:
            A run result containing all the inputs, guardrail results and the output of
            the last agent. Agents may perform handoffs, so we don't know the specific
            type of the output.
        """
        run_config = run_config or RunConfig()
        hooks = hooks or RunHooks()

        # Load session history if provided
        if session:
            history = await session.get_items()
            if run_config.session_input_callback:
                inputs = run_config.session_input_callback(history, [input] if isinstance(input, str) else input)
            else:
                inputs = history + ([{"role": "user", "content": input}] if isinstance(input, str) else input)
        else:
            inputs = [{"role": "user", "content": input}] if isinstance(input, str) else input

        current_agent = starting_agent
        turn_count = 0
        guardrail_results = []
        final_output = None

        while turn_count < max_turns:
            turn_count += 1
            hooks.on_turn_start(current_agent, turn_count)

            # Placeholder for actual agent execution
            # In real implementation, would call the agent with inputs
            # and handle tool calls, handoffs, etc.

            # For now, just return a simple result
            final_output = {"message": "Agent execution placeholder"}

            hooks.on_turn_end(current_agent, turn_count, final_output)

            # Check if we have a final output
            if current_agent.output_type and isinstance(final_output, current_agent.output_type):
                break

        if turn_count >= max_turns:
            raise MaxTurnsExceeded(max_turns)

        # Save to session if provided
        if session and final_output:
            await session.add_items([{"role": "assistant", "content": str(final_output)}])

        return RunResult(
            output=final_output,
            inputs=inputs,
            guardrail_results=guardrail_results,
            agent=current_agent,
            turns=turn_count,
            response_id=previous_response_id,
            conversation_id=conversation_id,
        )

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | List[TResponseInputItem],
        *,
        context: Optional[TContext] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Optional[RunHooks[TContext]] = None,
        run_config: Optional[RunConfig] = None,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> RunResult:
        """Run a workflow synchronously, starting at the given agent.

        Note:
            This just wraps the run method, so it will not work if there's already an
            event loop (e.g. inside an async function, or in a Jupyter notebook or async
            context like FastAPI). For those cases, use the run method instead.

        The agent will run in a loop until a final output is generated. The loop runs:

        1. The agent is invoked with the given input.
        2. If there is a final output (i.e. the agent produces something of type
           agent.output_type), the loop terminates.
        3. If there's a handoff, we run the loop again, with the new agent.
        4. Else, we run tool calls (if any), and re-run the loop.

        In two cases, the agent may raise an exception:

        - If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
        - If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.

        Note:
            Only the first agent's input guardrails are run.

        Args:
            starting_agent: The starting agent to run.
            input: The initial input to the agent. You can pass a single string for a user
                message, or a list of input items.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for. A turn is defined
                as one AI invocation (including any tool calls that might occur).
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.
            previous_response_id: The ID of the previous response, if using OpenAI models via
                the Responses API, this allows you to skip passing in input from the previous turn.
            conversation_id: The ID of the stored conversation, if any.
            session: A session for automatic conversation history management.

        Returns:
            A run result containing all the inputs, guardrail results and the output of
            the last agent. Agents may perform handoffs, so we don't know the specific
            type of the output.
        """
        return asyncio.run(cls.run(
            starting_agent,
            input,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            session=session,
        ))

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | List[TResponseInputItem],
        context: Optional[TContext] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Optional[RunHooks[TContext]] = None,
        run_config: Optional[RunConfig] = None,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> RunResultStreaming:
        """Run a workflow starting at the given agent in streaming mode.

        The returned result object contains a method you can use to stream semantic events
        as they are generated.

        The agent will run in a loop until a final output is generated. The loop runs like so:

        1. The agent is invoked with the given input.
        2. If there is a final output (i.e. the agent produces something of type
           agent.output_type), the loop terminates.
        3. If there's a handoff, we run the loop again, with the new agent.
        4. Else, we run tool calls (if any), and re-run the loop.

        In two cases, the agent may raise an exception:

        - If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
        - If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.

        Note:
            Only the first agent's input guardrails are run.

        Args:
            starting_agent: The starting agent to run.
            input: The initial input to the agent. You can pass a single string for a user
                message, or a list of input items.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for. A turn is defined
                as one AI invocation (including any tool calls that might occur).
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.
            previous_response_id: The ID of the previous response, if using OpenAI models via
                the Responses API, this allows you to skip passing in input from the previous turn.
            conversation_id: The ID of the stored conversation, if any.
            session: A session for automatic conversation history management.

        Returns:
            A result object that contains data about the run, as well as a method to
            stream events.
        """
        # Create an async task for the run
        run_task = asyncio.create_task(cls.run(
            starting_agent,
            input,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            session=session,
        ))

        return RunResultStreaming(run_task)
