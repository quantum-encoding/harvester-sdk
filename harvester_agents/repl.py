"""
REPL (Read-Eval-Print Loop) utility for interactive agent testing.

Provides a quick way to test agents directly in the terminal.
"""

import asyncio
import sys
from typing import Optional, Any


async def run_demo_loop(
    agent,
    session: Optional[Any] = None,
    streaming: bool = True,
    show_items: bool = False,
):
    """
    Start an interactive chat session with an agent.

    Continuously prompts for user input, maintains conversation history,
    and streams agent responses in real-time.

    Args:
        agent: OpenAIAgent instance or raw Agent from agents SDK
        session: Optional AgentSession for persistent history
        streaming: Whether to stream responses (default: True)
        show_items: Whether to show item-level events like tool calls

    Usage:
        agent = OpenAIAgent(name="Assistant", instructions="Be helpful")
        await run_demo_loop(agent)

    Commands:
        - Type 'quit' or 'exit' to end the session
        - Press Ctrl-D to exit
        - Press Ctrl-C to interrupt
    """
    try:
        from agents import Runner
        from openai.types.responses import ResponseTextDeltaEvent
        from agents import ItemHelpers
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    # Handle both OpenAIAgent wrapper and raw Agent
    from .openai_agent import OpenAIAgent
    if isinstance(agent, OpenAIAgent):
        raw_agent = agent.agent
    else:
        raw_agent = agent

    print("=" * 60)
    print(f"🤖 Interactive Agent Session: {raw_agent.name}")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end, or press Ctrl-D")
    print("=" * 60)
    print()

    turn_count = 0

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            turn_count += 1
            print(f"\nAgent: ", end="", flush=True)

            # Run with or without streaming
            if streaming:
                # Stream the response
                result = Runner.run_streamed(
                    raw_agent,
                    user_input,
                    session=session.session if session else None
                )

                async for event in result.stream_events():
                    # Token-by-token streaming
                    if event.type == "raw_response_event":
                        if isinstance(event.data, ResponseTextDeltaEvent):
                            print(event.data.delta, end="", flush=True)

                    # Show item-level events if requested
                    elif event.type == "run_item_stream_event" and show_items:
                        item = event.item

                        if item.type == "tool_call_item":
                            print(f"\n[🔧 Calling: {item.name}]", end="", flush=True)

                        elif item.type == "tool_call_output_item":
                            print(f"\n[✓ Output: {item.output[:50]}...]", end="", flush=True)

                    # Agent handoffs
                    elif event.type == "agent_updated_stream_event" and show_items:
                        print(f"\n[→ Agent: {event.new_agent.name}]", end="", flush=True)

                print("\n")

            else:
                # Non-streaming mode
                result = await Runner.run(
                    raw_agent,
                    user_input,
                    session=session.session if session else None
                )
                print(f"{result.final_output}\n")

        except EOFError:
            # Ctrl-D pressed
            print("\n\n👋 Goodbye!")
            break

        except KeyboardInterrupt:
            # Ctrl-C pressed
            print("\n\n⚠️  Interrupted")
            continue

        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nSession ended after {turn_count} turns.")


async def run_interactive_chat(
    agent,
    greeting: str = "Hello! How can I help you today?",
    session_id: Optional[str] = None,
):
    """
    Higher-level interactive chat with automatic session management.

    Args:
        agent: OpenAIAgent instance
        greeting: Initial greeting message
        session_id: Session ID for persistent history (optional)

    Example:
        agent = OpenAIAgent(name="Assistant")
        await run_interactive_chat(agent, session_id="user_123")
    """
    from .openai_agent import AgentSession

    # Create session if ID provided
    session = None
    if session_id:
        session = AgentSession.from_sqlite(session_id, "./sessions/repl.db")

    print(f"\nAgent: {greeting}\n")

    await run_demo_loop(agent, session=session, streaming=True)


# CLI entry point
async def main():
    """CLI entry point for quick agent testing."""
    import argparse
    from .openai_agent import OpenAIAgent

    parser = argparse.ArgumentParser(description="Interactive agent REPL")
    parser.add_argument("--model", default="gpt-5", help="Model to use")
    parser.add_argument("--instructions", default="You are a helpful assistant.", help="Agent instructions")
    parser.add_argument("--name", default="Assistant", help="Agent name")
    parser.add_argument("--session-id", help="Session ID for persistent history")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")

    args = parser.parse_args()

    agent = OpenAIAgent(
        name=args.name,
        instructions=args.instructions,
        model=args.model
    )

    session = None
    if args.session_id:
        from .openai_agent import AgentSession
        session = AgentSession.from_sqlite(args.session_id, "./sessions/repl.db")

    await run_demo_loop(agent, session=session, streaming=not args.no_stream)


if __name__ == "__main__":
    asyncio.run(main())
