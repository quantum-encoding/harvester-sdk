"""
Tool utilities for OpenAI Agents SDK.

Provides helpers for working with hosted tools (web search, file search, computer use)
and function tools.
"""

from typing import Optional, Callable, Any, List
from dataclasses import dataclass


class HostedTools:
    """
    Factory for creating OpenAI hosted tools.

    Provides convenient access to:
    - WebSearchTool: Search the web
    - FileSearchTool: Search vector stores
    - ComputerTool: Computer use automation
    - CodeInterpreterTool: Execute code in sandbox
    - ImageGenerationTool: Generate images
    - LocalShellTool: Run shell commands
    """

    @staticmethod
    def web_search(max_results: Optional[int] = None):
        """
        Create a web search tool.

        Args:
            max_results: Maximum number of search results to return

        Returns:
            WebSearchTool instance
        """
        try:
            from agents import WebSearchTool
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        kwargs = {}
        if max_results is not None:
            kwargs["max_num_results"] = max_results

        return WebSearchTool(**kwargs)

    @staticmethod
    def file_search(vector_store_ids: List[str], max_results: int = 5):
        """
        Create a file search tool for vector stores.

        Args:
            vector_store_ids: List of OpenAI vector store IDs
            max_results: Maximum number of results to return

        Returns:
            FileSearchTool instance
        """
        try:
            from agents import FileSearchTool
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        return FileSearchTool(
            vector_store_ids=vector_store_ids,
            max_num_results=max_results
        )

    @staticmethod
    def computer_use(display_width: int = 1024, display_height: int = 768):
        """
        Create a computer use tool.

        Args:
            display_width: Display width in pixels
            display_height: Display height in pixels

        Returns:
            ComputerTool instance
        """
        try:
            from agents import ComputerTool
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        return ComputerTool(
            display_width=display_width,
            display_height=display_height
        )

    @staticmethod
    def code_interpreter():
        """
        Create a code interpreter tool.

        Returns:
            CodeInterpreterTool instance
        """
        try:
            from agents import CodeInterpreterTool
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        return CodeInterpreterTool()

    @staticmethod
    def local_shell():
        """
        Create a local shell tool.

        Returns:
            LocalShellTool instance
        """
        try:
            from agents import LocalShellTool
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        return LocalShellTool()

    @staticmethod
    def image_generation():
        """
        Create an image generation tool.

        Returns:
            ImageGenerationTool instance
        """
        try:
            from agents import ImageGenerationTool
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        return ImageGenerationTool()


def create_function_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    error_handler: Optional[Callable] = None,
):
    """
    Create a function tool with custom configuration.

    Args:
        func: Python function to use as tool
        name: Override function name
        description: Override function description
        error_handler: Custom error handler function

    Returns:
        Function tool decorator

    Example:
        @create_function_tool(name="get_weather_data")
        def fetch_weather(city: str) -> str:
            '''Get weather for a city'''
            return f"Sunny in {city}"
    """
    try:
        from agents import function_tool
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    kwargs = {}
    if name:
        kwargs["name_override"] = name
    if description:
        kwargs["description_override"] = description
    if error_handler:
        kwargs["failure_error_function"] = error_handler

    return function_tool(**kwargs)(func)


@dataclass
class ToolConfig:
    """Configuration for agent tools."""

    name: str
    description: str
    enabled: bool = True
    error_handler: Optional[Callable] = None


def agent_as_tool(
    agent,
    tool_name: str,
    tool_description: str,
    enabled: bool = True,
    custom_output_extractor: Optional[Callable] = None,
):
    """
    Convert an agent into a tool for another agent.

    Allows a "manager" agent to orchestrate specialized sub-agents.

    Args:
        agent: OpenAIAgent or Agent instance
        tool_name: Name for the tool
        tool_description: Description of what the tool does
        enabled: Whether the tool is enabled
        custom_output_extractor: Function to extract custom output from results

    Returns:
        Tool that can be added to another agent

    Example:
        spanish_agent = OpenAIAgent(
            name="Spanish",
            instructions="Translate to Spanish"
        )

        orchestrator = OpenAIAgent(
            name="Orchestrator",
            tools=[agent_as_tool(
                spanish_agent,
                "translate_spanish",
                "Translate text to Spanish"
            )]
        )
    """
    from .openai_agent import OpenAIAgent

    # Handle OpenAIAgent wrapper
    if isinstance(agent, OpenAIAgent):
        raw_agent = agent.agent
    else:
        raw_agent = agent

    kwargs = {
        "tool_name": tool_name,
        "tool_description": tool_description,
    }

    if enabled is not True:
        kwargs["is_enabled"] = enabled

    if custom_output_extractor:
        kwargs["custom_output_extractor"] = custom_output_extractor

    return raw_agent.as_tool(**kwargs)


# Example usage
if __name__ == "__main__":
    from .openai_agent import OpenAIAgent, function_tool

    # Example 1: Hosted tools
    agent = OpenAIAgent(
        name="Research Assistant",
        instructions="Help users research topics",
        tools=[
            HostedTools.web_search(max_results=5),
            HostedTools.code_interpreter(),
        ]
    )

    # Example 2: Function tools
    @function_tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    weather_agent = OpenAIAgent(
        name="Weather",
        tools=[get_weather]
    )

    # Example 3: Agent as tool
    spanish_agent = OpenAIAgent(
        name="Spanish Translator",
        instructions="Translate to Spanish"
    )

    orchestrator = OpenAIAgent(
        name="Orchestrator",
        instructions="Coordinate translation tasks",
        tools=[agent_as_tool(
            spanish_agent,
            "translate_spanish",
            "Translate text to Spanish"
        )]
    )

    print("Tool examples created successfully!")
