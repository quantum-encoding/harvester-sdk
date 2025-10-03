#!/usr/bin/env python3
"""
Interactive OpenAI Chat with Streaming (Responses API)
A terminal-based chat interface for OpenAI's GPT models using the Responses API
"""
import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Load environment variables
load_dotenv()

class OpenAIChat:
    def __init__(self):
        if not HAS_OPENAI:
            raise ImportError("OpenAI SDK not installed. Install with: pip install openai")

        # Get API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable")

        # Initialize client
        self.client = OpenAI(api_key=self.api_key)

        self.input_history = []
        self.current_model = "gpt-5"
        self.use_functions = False
        self.use_search = False
        self.reasoning_effort = "medium"
        self.verbosity = "medium"
        self.function_tools = []
        self.previous_response_id = None
        self.setup_function_tools()

        # Setup prompt session
        if HAS_PROMPT_TOOLKIT:
            chat_commands = [
                '/help', '/quit', '/exit', '/model', '/search', '/functions',
                '/reasoning', '/verbosity', '/clear', '/history', '/save', '/load', '/export', '/paste'
            ]
            completer = WordCompleter(chat_commands, ignore_case=True, sentence=True)
            history_file = os.path.expanduser('~/.openai_chat_history')

            self.prompt_session = PromptSession(
                history=FileHistory(history_file),
                completer=completer,
                auto_suggest=AutoSuggestFromHistory()
            )
        else:
            self.prompt_session = None

    def setup_function_tools(self):
        """Setup function calling tools"""
        self.function_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time in a specific timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone (e.g., 'America/New_York', 'UTC')"
                            }
                        },
                        "required": ["timezone"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name or location"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

    def execute_function(self, function_name: str, arguments: dict) -> dict:
        """Execute a function call"""
        if function_name == "calculate":
            try:
                expression = arguments.get("expression", "")
                result = eval(expression, {"__builtins__": {}}, {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len, "pow": pow, "sqrt": lambda x: x**0.5
                })
                return {"result": result, "expression": expression}
            except Exception as e:
                return {"error": str(e)}

        elif function_name == "get_current_time":
            try:
                from datetime import datetime
                import pytz
                tz = pytz.timezone(arguments.get("timezone", "UTC"))
                current_time = datetime.now(tz)
                return {
                    "timezone": arguments.get("timezone"),
                    "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": current_time.timestamp()
                }
            except Exception as e:
                return {"error": str(e)}

        elif function_name == "get_weather":
            # Mock weather function
            return {
                "location": arguments.get("location"),
                "temperature": 22,
                "unit": "celsius",
                "conditions": "partly cloudy"
            }

        return {"error": "Unknown function"}

    def print_help(self):
        """Print help information"""
        print(f"""
Available Commands:
  /help              - Show this help
  /model <name>      - Switch model
  /search            - Toggle web search on/off
  /functions         - Toggle function calling on/off
  /reasoning <level> - Set reasoning effort (minimal, low, medium, high)
  /verbosity <level> - Set text verbosity (low, medium, high)
  /clear             - Clear conversation history
  /history           - Show conversation history
  /save <file>       - Save conversation to file
  /load <file>       - Load conversation from file
  /export <file>     - Export conversation as markdown
  /paste             - Enter multi-line input mode
  /quit, /exit       - Exit the chat

Current Settings:
  Model: {self.current_model}
  Web Search: {'ON' if self.use_search else 'OFF'}
  Functions: {'ON' if self.use_functions else 'OFF'}
  Reasoning Effort: {self.reasoning_effort}
  Verbosity: {self.verbosity}

Models Available:
  - gpt-5                 : Latest GPT-5 (best)
  - gpt-5-mini            : Cost-optimized
  - gpt-5-nano            : High-throughput
  - gpt-4.1               : Fast, non-reasoning
        """)

    def print_status(self):
        """Print current status"""
        print(f"\n🤖 OpenAI Chat - Model: {self.current_model} | Search: {'ON' if self.use_search else 'OFF'} | Functions: {'ON' if self.use_functions else 'OFF'} | Reasoning: {self.reasoning_effort}")
        print("Type /help for commands or start chatting...\n")

    async def handle_command(self, command: str) -> bool:
        """Handle user commands"""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd in ['/quit', '/exit']:
            return False

        elif cmd == '/help':
            self.print_help()

        elif cmd == '/model':
            if len(parts) > 1:
                self.current_model = parts[1]
                print(f"✓ Switched to {self.current_model}")
            else:
                print(f"Current model: {self.current_model}")

        elif cmd == '/search':
            self.use_search = not self.use_search
            print(f"✓ Web search {'enabled' if self.use_search else 'disabled'}")

        elif cmd == '/functions':
            self.use_functions = not self.use_functions
            print(f"✓ Function calling {'enabled' if self.use_functions else 'disabled'}")

        elif cmd == '/reasoning':
            if len(parts) > 1:
                level = parts[1].lower()
                if level in ['minimal', 'low', 'medium', 'high']:
                    self.reasoning_effort = level
                    print(f"✓ Reasoning effort set to {level}")
                else:
                    print("❌ Invalid level. Use: minimal, low, medium, high")
            else:
                print(f"Current reasoning effort: {self.reasoning_effort}")

        elif cmd == '/verbosity':
            if len(parts) > 1:
                level = parts[1].lower()
                if level in ['low', 'medium', 'high']:
                    self.verbosity = level
                    print(f"✓ Verbosity set to {level}")
                else:
                    print("❌ Invalid level. Use: low, medium, high")
            else:
                print(f"Current verbosity: {self.verbosity}")

        elif cmd == '/clear':
            self.input_history = []
            self.previous_response_id = None
            print("✓ Conversation cleared")

        elif cmd == '/history':
            if not self.input_history:
                print("No conversation history")
            else:
                for i, item in enumerate(self.input_history):
                    role = item['role'].title()
                    content = str(item.get('content', ''))[:100]
                    print(f"{i+1}. {role}: {content}...")

        elif cmd == '/save':
            if len(parts) > 1:
                filename = parts[1]
                try:
                    with open(filename, 'w') as f:
                        json.dump(self.input_history, f, indent=2)
                    print(f"✓ Conversation saved to {filename}")
                except Exception as e:
                    print(f"❌ Error saving: {e}")
            else:
                print("Usage: /save <filename>")

        elif cmd == '/load':
            if len(parts) > 1:
                filename = parts[1]
                try:
                    with open(filename, 'r') as f:
                        self.input_history = json.load(f)
                    print(f"✓ Conversation loaded from {filename}")
                except Exception as e:
                    print(f"❌ Error loading: {e}")
            else:
                print("Usage: /load <filename>")

        elif cmd == '/export':
            if len(parts) > 1:
                filename = parts[1]
                try:
                    self.export_conversation_markdown(filename)
                    print(f"✓ Conversation exported to {filename}")
                except Exception as e:
                    print(f"❌ Error exporting: {e}")
            else:
                print("Usage: /export <filename.md>")

        elif cmd == '/paste':
            print("📝 Multi-line input mode. Type 'END' on a new line to finish:")
            lines = []
            while True:
                try:
                    line = input("... ")
                    if line.strip() == 'END':
                        break
                    lines.append(line)
                except (KeyboardInterrupt, EOFError):
                    print("\n❌ Multi-line input cancelled")
                    return True

            if lines:
                multi_input = '\n'.join(lines)
                print(f"\n📤 Processing {len(lines)} line(s) of input...\n")
                await self.chat(multi_input)
                print()

        else:
            print("❌ Unknown command. Type /help for available commands.")

        return True

    async def chat(self, user_input: str):
        """Handle chat interaction with streaming using Responses API"""
        try:
            # Build tools list
            tools = []

            # Add web search if enabled
            if self.use_search:
                tools.append({"type": "web_search"})

            # Add function tools if enabled
            if self.use_functions:
                tools.extend(self.function_tools)

            # Build request
            request_params = {
                "model": self.current_model,
                "stream": True,
                "reasoning": {"effort": self.reasoning_effort},
                "text": {"verbosity": self.verbosity}
            }

            # Add tools if any
            if tools:
                request_params["tools"] = tools

            # Use previous_response_id for context or build input
            if self.previous_response_id:
                request_params["previous_response_id"] = self.previous_response_id
                request_params["input"] = [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_input}]
                    }
                ]
            else:
                # First message in conversation
                request_params["input"] = [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_input}]
                    }
                ]

            # Add to history
            self.input_history.append({
                "role": "user",
                "content": user_input
            })

            print("🤖 GPT: ", end="", flush=True)

            # Stream the response
            full_text = ""
            response_id = None

            stream = self.client.responses.create(**request_params)

            for event in stream:
                # Store response ID
                if hasattr(event, 'id'):
                    response_id = event.id

                # Handle text delta events
                if event.type == "response.output_text.delta":
                    if hasattr(event, 'delta'):
                        print(event.delta, end="", flush=True)
                        full_text += event.delta

                # Handle completed events
                elif event.type == "response.completed":
                    pass

                # Handle errors
                elif event.type == "error":
                    print(f"\n❌ Error: {event}")
                    return

            print()  # Newline after response

            # Store response ID for next turn
            if response_id:
                self.previous_response_id = response_id

            # Add assistant response to history
            if full_text:
                self.input_history.append({
                    "role": "assistant",
                    "content": full_text
                })

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def export_conversation_markdown(self, filename: str):
        """Export conversation as markdown"""
        if not filename.endswith('.md'):
            filename += '.md'

        with open(filename, 'w') as f:
            f.write(f"# OpenAI Chat Session\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.current_model}\n")
            f.write(f"**Reasoning:** {self.reasoning_effort}\n")
            f.write(f"**Verbosity:** {self.verbosity}\n\n")
            f.write("---\n\n")

            for msg in self.input_history:
                role = msg['role']
                content = msg['content']

                if role == 'user':
                    f.write(f"## 👤 User\n\n{content}\n\n")
                elif role == 'assistant':
                    f.write(f"## 🤖 GPT\n\n{content}\n\n")

                f.write("---\n\n")

    async def get_input_async(self, prompt: str) -> str:
        """Get input with best available method"""
        if HAS_PROMPT_TOOLKIT:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.prompt_session.prompt(prompt)
                )
                return result
            except (KeyboardInterrupt, EOFError):
                raise
        else:
            return input(prompt)

    async def run(self):
        """Main chat loop"""
        print("🚀 OpenAI Interactive Chat (Responses API)")
        print("=" * 50)
        if HAS_PROMPT_TOOLKIT:
            print("✨ Enhanced mode: Multi-line paste | History | Auto-complete | Auto-suggest")
            print("💡 Tab for command completion | ↑/↓ for history | Type /help for commands")
        else:
            print("💡 Type /help for commands")
            print("⚠️  Install prompt_toolkit for better experience: pip install prompt_toolkit")
        self.print_status()

        try:
            while True:
                try:
                    user_input = await self.get_input_async("👤 You: ")
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    if user_input.startswith('/'):
                        should_continue = await self.handle_command(user_input)
                        if not should_continue:
                            break
                        continue

                    await self.chat(user_input)
                    print()

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break

        finally:
            pass

async def main():
    """Entry point"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
OpenAI Interactive Chat (Responses API)

Usage: python openai_chat.py [options]

Options:
  -h, --help       Show this help
  --model MODEL    Start with specific model
  --search         Enable web search by default
  --functions      Enable function calling by default
  --reasoning LEVEL    Set reasoning effort (minimal, low, medium, high)
  --verbosity LEVEL    Set text verbosity (low, medium, high)

Environment:
  OPENAI_API_KEY   Your OpenAI API key (required)
        """)
        return

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    # Initialize chat
    chat = OpenAIChat()

    # Parse command line options
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--model' and i + 1 < len(sys.argv):
            chat.current_model = sys.argv[i + 1]
        elif arg == '--search':
            chat.use_search = True
        elif arg == '--functions':
            chat.use_functions = True
        elif arg == '--reasoning' and i + 1 < len(sys.argv):
            level = sys.argv[i + 1].lower()
            if level in ['minimal', 'low', 'medium', 'high']:
                chat.reasoning_effort = level
        elif arg == '--verbosity' and i + 1 < len(sys.argv):
            level = sys.argv[i + 1].lower()
            if level in ['low', 'medium', 'high']:
                chat.verbosity = level

    # Start chat
    await chat.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
