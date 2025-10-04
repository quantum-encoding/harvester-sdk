"""
OpenAI Agents SDK Integration for Harvester SDK

Lightweight wrapper around the OpenAI Agents SDK for building agentic workflows.
Supports GPT-5 and other OpenAI models with tools, handoffs, and guardrails.
"""

from typing import Any, Callable, Optional, List, Union
from dataclasses import dataclass
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Enable LD_PRELOAD safety for all subprocess calls
_SAFE_EXEC_LIB = str(Path(__file__).parent / "safe_exec.so")
if os.path.exists(_SAFE_EXEC_LIB):
    os.environ['LD_PRELOAD'] = _SAFE_EXEC_LIB
    logger.info(f"Safe execution library loaded: {_SAFE_EXEC_LIB}")


class OpenAIAgent:
    """
    Wrapper for OpenAI Agents SDK Agent.

    Provides a simple interface to create agents with instructions, tools, and handoffs.
    """

    def __init__(
        self,
        name: str,
        instructions: Union[str, Callable],
        model: str = "gpt-5",
        tools: Optional[List] = None,
        handoffs: Optional[List] = None,
        output_type: Optional[type] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize an OpenAI Agent.

        Args:
            name: Agent identifier
            instructions: System prompt or dynamic instruction function
            model: Model name (default: gpt-5)
            tools: List of function_tool decorated functions
            handoffs: List of agents for delegation
            output_type: Pydantic model or type for structured output
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        try:
            from agents import Agent, ModelSettings
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        # Build model settings
        model_settings = None
        if temperature is not None or max_tokens is not None:
            model_settings = ModelSettings(
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Create the agent
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            tools=tools or [],
            handoffs=handoffs or [],
            model_settings=model_settings,
        )

        if output_type:
            self.agent.output_type = output_type

    def run(self, user_input: str, context: Optional[Any] = None) -> str:
        """
        Run the agent synchronously.

        Args:
            user_input: User message
            context: Optional context object

        Returns:
            Final output as string
        """
        from agents import Runner

        result = Runner.run_sync(self.agent, user_input)
        return result.final_output

    async def run_async(self, user_input: str, context: Optional[Any] = None):
        """
        Run the agent asynchronously.

        Args:
            user_input: User message
            context: Optional context object

        Returns:
            RunResult object
        """
        from agents import Runner

        result = await Runner.run(self.agent, user_input)
        return result

    async def run_streamed(self, user_input: str, context: Optional[Any] = None):
        """
        Run the agent with streaming enabled.

        Returns a RunResultStreaming object that you can iterate over
        to receive events as they happen.

        Args:
            user_input: User message
            context: Optional context object

        Returns:
            RunResultStreaming object

        Example:
            result = await agent.run_streamed("Tell me a joke")
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    # Handle token-by-token streaming
                    print(event.data.delta, end="", flush=True)
                elif event.type == "run_item_stream_event":
                    # Handle complete items (messages, tool calls, etc.)
                    print(f"Item: {event.item}")
        """
        from agents import Runner

        result = Runner.run_streamed(self.agent, user_input)
        return result

    def clone(self, **kwargs):
        """Clone the agent with optional property overrides."""
        return self.agent.clone(**kwargs)


def function_tool(func: Callable) -> Callable:
    """
    Decorator to turn a Python function into an agent tool.

    Automatically generates schema from function signature and docstring.

    Example:
        @function_tool
        def get_weather(city: str) -> str:
            '''Returns weather info for the specified city.'''
            return f"The weather in {city} is sunny"
    """
    try:
        from agents import function_tool as _function_tool
        return _function_tool(func)
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )


class AgentSession:
    """
    Manages conversation history across multiple agent runs.

    Wraps the OpenAI Agents SDK Session for automatic state management.

    Supports multiple backends:
    - SQLite (default): File-based or in-memory storage
    - OpenAI Conversations API: Cloud-hosted storage
    - SQLAlchemy: PostgreSQL, MySQL, etc.
    """

    def __init__(
        self,
        session_id: str,
        db_path: Optional[str] = None,
        backend: str = "sqlite"
    ):
        """
        Initialize a session.

        Args:
            session_id: Unique identifier for this conversation
            db_path: Path to database file (SQLite only). If None, uses in-memory DB.
            backend: Session backend - "sqlite", "openai", or "sqlalchemy"
        """
        try:
            from agents import SQLiteSession
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        self.session_id = session_id
        self.backend = backend

        if backend == "sqlite":
            if db_path:
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.session = SQLiteSession(session_id, db_path)
        elif backend == "openai":
            from agents import OpenAIConversationsSession
            self.session = OpenAIConversationsSession()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def run(self, agent: OpenAIAgent, user_input: str) -> str:
        """
        Run agent with automatic conversation history.

        Args:
            agent: OpenAIAgent instance
            user_input: User message

        Returns:
            Final output as string
        """
        from agents import Runner

        result = Runner.run_sync(agent.agent, user_input, session=self.session)
        return result.final_output

    async def run_async(self, agent: OpenAIAgent, user_input: str):
        """
        Run agent asynchronously with automatic conversation history.

        Args:
            agent: OpenAIAgent instance
            user_input: User message

        Returns:
            RunResult object
        """
        from agents import Runner

        result = await Runner.run(agent.agent, user_input, session=self.session)
        return result

    async def get_items(self, limit: Optional[int] = None):
        """
        Retrieve conversation history for this session.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of conversation items
        """
        return await self.session.get_items(limit=limit)

    async def add_items(self, items: List[dict]):
        """
        Add items to the session history.

        Args:
            items: List of conversation items to add
        """
        await self.session.add_items(items)

    async def pop_item(self):
        """
        Remove and return the most recent item from the session.

        Useful for undoing or correcting the last message.

        Returns:
            The most recent item, or None if session is empty
        """
        return await self.session.pop_item()

    async def clear_session(self):
        """Clear all items from this session."""
        await self.session.clear_session()

    @classmethod
    def from_sqlite(cls, session_id: str, db_path: Optional[str] = None):
        """
        Create a SQLite-backed session.

        Args:
            session_id: Unique identifier
            db_path: Path to database file. If None, uses in-memory DB.

        Returns:
            AgentSession instance
        """
        return cls(session_id, db_path=db_path, backend="sqlite")

    @classmethod
    def from_openai(cls, session_id: str):
        """
        Create an OpenAI Conversations API-backed session.

        Args:
            session_id: Unique identifier

        Returns:
            AgentSession instance
        """
        return cls(session_id, backend="openai")

    @classmethod
    def from_sqlalchemy(cls, session_id: str, engine_url: str, create_tables: bool = True):
        """
        Create a SQLAlchemy-backed session.

        Args:
            session_id: Unique identifier
            engine_url: SQLAlchemy database URL (e.g., "postgresql://...")
            create_tables: Whether to auto-create tables

        Returns:
            AgentSession instance with SQLAlchemy backend
        """
        try:
            from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession
        except ImportError:
            raise ImportError(
                "SQLAlchemy session support not available. "
                "Install with: pip install sqlalchemy aiosqlite"
            )

        session = cls.__new__(cls)
        session.session_id = session_id
        session.backend = "sqlalchemy"
        session.session = SQLAlchemySession.from_url(
            session_id,
            url=engine_url,
            create_tables=create_tables
        )
        return session


# Example usage
if __name__ == "__main__":
    # Simple agent
    agent = OpenAIAgent(
        name="Assistant",
        instructions="You are a helpful assistant. Reply concisely.",
        model="gpt-5",
    )

    response = agent.run("What is 2+2?")
    print(response)

    # Agent with tools
    @function_tool
    def get_weather(city: str) -> str:
        """Returns weather info for the specified city."""
        return f"The weather in {city} is sunny"

    weather_agent = OpenAIAgent(
        name="Weather Agent",
        instructions="Help users with weather information.",
        tools=[get_weather],
    )

    response = weather_agent.run("What's the weather in Paris?")
    print(response)
