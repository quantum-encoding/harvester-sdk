"""
Streaming utilities for OpenAI Agents SDK

Helper functions for working with streaming agent responses.
"""

from typing import AsyncIterator, Callable, Optional
import asyncio


class StreamHandler:
    """
    Helper class for handling streaming events from agents.

    Provides callbacks for different event types:
    - on_token: Called for each token/text delta
    - on_message: Called when a complete message is received
    - on_tool_call: Called when a tool is invoked
    - on_tool_output: Called when a tool returns output
    - on_agent_update: Called when the agent changes (handoffs)
    """

    def __init__(
        self,
        on_token: Optional[Callable[[str], None]] = None,
        on_message: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
        on_tool_output: Optional[Callable[[str, str], None]] = None,
        on_agent_update: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize stream handler with callbacks.

        Args:
            on_token: Callback for each token (token_text)
            on_message: Callback for complete messages (message_text)
            on_tool_call: Callback for tool calls (tool_name, arguments)
            on_tool_output: Callback for tool outputs (tool_name, output)
            on_agent_update: Callback for agent changes (agent_name)
        """
        self.on_token = on_token
        self.on_message = on_message
        self.on_tool_call = on_tool_call
        self.on_tool_output = on_tool_output
        self.on_agent_update = on_agent_update

    async def process_stream(self, result_stream):
        """
        Process a streaming result and call appropriate callbacks.

        Args:
            result_stream: RunResultStreaming object from Runner.run_streamed()

        Returns:
            The complete RunResultStreaming object after streaming finishes
        """
        try:
            from openai.types.responses import ResponseTextDeltaEvent
            from agents import ItemHelpers
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        async for event in result_stream.stream_events():
            # Handle raw token-by-token streaming
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    if self.on_token:
                        self.on_token(event.data.delta)

            # Handle agent updates (handoffs)
            elif event.type == "agent_updated_stream_event":
                if self.on_agent_update:
                    self.on_agent_update(event.new_agent.name)

            # Handle complete items
            elif event.type == "run_item_stream_event":
                item = event.item

                # Tool call started
                if item.type == "tool_call_item":
                    if self.on_tool_call:
                        self.on_tool_call(item.name, item.arguments)

                # Tool output received
                elif item.type == "tool_call_output_item":
                    if self.on_tool_output:
                        self.on_tool_output(item.name, item.output)

                # Message output
                elif item.type == "message_output_item":
                    if self.on_message:
                        message_text = ItemHelpers.text_message_output(item)
                        self.on_message(message_text)

        return result_stream


async def stream_to_console(result_stream, show_tokens: bool = True, show_items: bool = True):
    """
    Stream agent output to the console.

    Args:
        result_stream: RunResultStreaming object from Runner.run_streamed()
        show_tokens: Whether to show token-by-token output
        show_items: Whether to show item-level events (tool calls, etc.)

    Returns:
        The complete RunResultStreaming object

    Example:
        result = Runner.run_streamed(agent, "Tell me a joke")
        await stream_to_console(result)
    """
    try:
        from openai.types.responses import ResponseTextDeltaEvent
        from agents import ItemHelpers
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    print("=== Agent streaming ===\n")

    async for event in result_stream.stream_events():
        # Token-by-token streaming
        if event.type == "raw_response_event" and show_tokens:
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

        # Agent updates
        elif event.type == "agent_updated_stream_event" and show_items:
            print(f"\n[Agent: {event.new_agent.name}]")

        # Item-level events
        elif event.type == "run_item_stream_event" and show_items:
            item = event.item

            if item.type == "tool_call_item":
                print(f"\n[Calling tool: {item.name}]")

            elif item.type == "tool_call_output_item":
                print(f"[Tool output: {item.output}]")

            elif item.type == "message_output_item":
                if not show_tokens:  # Only show if not already showing tokens
                    message_text = ItemHelpers.text_message_output(item)
                    print(f"\n{message_text}")

    print("\n\n=== Stream complete ===")
    return result_stream


async def collect_stream_text(result_stream) -> str:
    """
    Collect all text output from a stream into a single string.

    Args:
        result_stream: RunResultStreaming object from Runner.run_streamed()

    Returns:
        Complete text output as a string

    Example:
        result = Runner.run_streamed(agent, "Tell me a joke")
        text = await collect_stream_text(result)
        print(text)
    """
    try:
        from openai.types.responses import ResponseTextDeltaEvent
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    text_parts = []

    async for event in result_stream.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                text_parts.append(event.data.delta)

    return "".join(text_parts)


# Example usage
if __name__ == "__main__":
    from agents import Agent, Runner

    async def main():
        agent = Agent(
            name="Assistant",
            instructions="You are helpful. Be concise."
        )

        # Example 1: Stream to console
        print("Example 1: Stream to console")
        result = Runner.run_streamed(agent, "Tell me a short joke")
        await stream_to_console(result)

        print("\n" + "=" * 50 + "\n")

        # Example 2: Custom handler
        print("Example 2: Custom handler")

        def on_token(token):
            print(token, end="", flush=True)

        def on_message(message):
            print(f"\n[Complete message received]")

        handler = StreamHandler(on_token=on_token, on_message=on_message)
        result = Runner.run_streamed(agent, "Count to 5")
        await handler.process_stream(result)

        print("\n" + "=" * 50 + "\n")

        # Example 3: Collect text
        print("Example 3: Collect text")
        result = Runner.run_streamed(agent, "What is 2+2?")
        text = await collect_stream_text(result)
        print(f"Collected text: {text}")

    asyncio.run(main())
