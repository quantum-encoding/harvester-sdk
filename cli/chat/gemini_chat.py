#!/usr/bin/env python3
"""
Interactive Gemini Chat with Streaming and Function Calling
A terminal-based chat interface for Google Gemini models with real-time streaming
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
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Load environment variables
load_dotenv()

class GeminiChat:
    def __init__(self):
        if not HAS_GENAI:
            raise ImportError("Google GenAI not installed. Install with: pip install google-generativeai")

        # Get API key
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable")

        # Initialize client
        self.client = genai.Client(api_key=self.api_key)

        self.conversation = []
        self.current_model = "gemini-2.5-flash"
        self.use_functions = False
        self.use_search = False
        self.function_registry = {}
        self.setup_basic_tools()

        # Setup prompt session with history if prompt_toolkit is available
        if HAS_PROMPT_TOOLKIT:
            # Command completer for slash commands
            chat_commands = [
                '/help', '/quit', '/exit', '/model', '/functions', '/search',
                '/clear', '/history', '/save', '/load', '/export', '/paste'
            ]
            completer = WordCompleter(chat_commands, ignore_case=True, sentence=True)

            # Use file-based history for persistence across sessions
            history_file = os.path.expanduser('~/.gemini_chat_history')

            self.prompt_session = PromptSession(
                history=FileHistory(history_file),
                completer=completer,
                auto_suggest=AutoSuggestFromHistory()
            )
        else:
            self.prompt_session = None

    def setup_basic_tools(self):
        """Setup built-in function tools"""

        def calculate(expression: str) -> dict:
            """Perform mathematical calculations

            Args:
                expression: Mathematical expression to evaluate

            Returns:
                Calculation result or error
            """
            try:
                result = eval(expression, {"__builtins__": {}}, {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len, "pow": pow, "sqrt": lambda x: x**0.5
                })
                return {"expression": expression, "result": result}
            except Exception as e:
                return {"error": str(e)}

        def get_time(timezone: str = "UTC") -> dict:
            """Get current time in a timezone

            Args:
                timezone: Timezone name (e.g., 'UTC', 'America/New_York')

            Returns:
                Current time information
            """
            try:
                import pytz
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz)
                return {
                    "timezone": timezone,
                    "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": current_time.timestamp()
                }
            except Exception as e:
                return {"error": str(e)}

        def get_weather(location: str) -> dict:
            """Get weather information for a location (mock)

            Args:
                location: City name or location

            Returns:
                Weather information
            """
            # Mock implementation
            return {
                "location": location,
                "temperature": 22,
                "unit": "celsius",
                "conditions": "partly cloudy",
                "humidity": 65
            }

        # Register functions
        self.function_registry = {
            "calculate": calculate,
            "get_time": get_time,
            "get_weather": get_weather
        }

    def print_help(self):
        """Print help information"""
        print("""
Available Commands:
  /help          - Show this help
  /model <name>  - Switch model (gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash)
  /search        - Toggle Google Search grounding on/off
  /functions     - Toggle function calling on/off
  /clear         - Clear conversation history
  /history       - Show conversation history
  /save <file>   - Save conversation to file
  /load <file>   - Load conversation from file
  /export <file> - Export conversation as markdown
  /paste         - Enter multi-line input mode
  /quit, /exit   - Exit the chat

Current Settings:
  Model: {self.current_model}
  Google Search: {'ON' if self.use_search else 'OFF'}
  Functions: {'ON' if self.use_functions else 'OFF'}

Models Available:
  - gemini-2.5-pro       : Most capable model
  - gemini-2.5-flash     : Fast, balanced performance
  - gemini-2.5-flash-lite: Lightweight, efficient
  - gemini-2.0-flash     : Previous generation
        """.format(self=self))

    def print_status(self):
        """Print current status"""
        print(f"\n🤖 Gemini Chat - Model: {self.current_model} | Search: {'ON' if self.use_search else 'OFF'} | Functions: {'ON' if self.use_functions else 'OFF'}")
        print("Type /help for commands or start chatting...\n")

    async def handle_command(self, command: str) -> bool:
        """Handle user commands. Returns True if should continue, False to exit"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd in ['/quit', '/exit']:
            return False

        elif cmd == '/help':
            self.print_help()

        elif cmd == '/model':
            if len(parts) > 1:
                model = parts[1]
                valid_models = [
                    'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                    'gemini-2.0-flash'
                ]
                if model in valid_models:
                    self.current_model = model
                    print(f"✓ Switched to {model}")
                else:
                    print(f"❌ Invalid model. Use one of: {', '.join(valid_models)}")
            else:
                print(f"Current model: {self.current_model}")

        elif cmd == '/search':
            self.use_search = not self.use_search
            print(f"✓ Google Search grounding {'enabled' if self.use_search else 'disabled'}")

        elif cmd == '/functions':
            self.use_functions = not self.use_functions
            print(f"✓ Function calling {'enabled' if self.use_functions else 'disabled'}")

        elif cmd == '/clear':
            self.conversation = []
            print("✓ Conversation cleared")

        elif cmd == '/history':
            if not self.conversation:
                print("No conversation history")
            else:
                for i, msg in enumerate(self.conversation):
                    role = msg.get('role', 'unknown').title()
                    content = str(msg.get('parts', [''])[0])[:100]
                    print(f"{i+1}. {role}: {content}...")

        elif cmd == '/save':
            if len(parts) > 1:
                filename = parts[1]
                try:
                    with open(filename, 'w') as f:
                        json.dump(self.conversation, f, indent=2)
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
                        self.conversation = json.load(f)
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
                print("❌ No input provided")

        else:
            print("❌ Unknown command. Type /help for available commands.")

        return True

    async def chat(self, user_input: str):
        """Handle chat interaction"""
        try:
            # Prepare tools
            tools = []

            # Add Google Search if enabled
            if self.use_search:
                tools.append(types.Tool(google_search=types.GoogleSearch()))

            # Add functions if enabled
            if self.use_functions:
                # Create function declarations
                function_declarations = []
                for func_name, func in self.function_registry.items():
                    # Simple declaration
                    function_declarations.append({
                        "name": func_name,
                        "description": func.__doc__.split('\n')[0] if func.__doc__ else func_name,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"} if func_name == "calculate" else
                                "timezone": {"type": "string"} if func_name == "get_time" else
                                "location": {"type": "string"}
                            }
                        }
                    })

                tools.append(types.Tool(function_declarations=function_declarations))

            # Create config
            config = types.GenerateContentConfig(
                tools=tools if tools else None,
                temperature=0.7
            )

            # Add user message to history
            self.conversation.append({
                "role": "user",
                "parts": [{"text": user_input}]
            })

            print("🤖 Gemini: ", end="", flush=True)

            # Generate response with streaming
            response = await asyncio.to_thread(
                self.client.models.generate_content_stream,
                model=self.current_model,
                contents=user_input,
                config=config
            )

            full_text = ""
            async for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_text += chunk.text

                # Handle function calls
                if self.use_functions and hasattr(chunk, 'candidates'):
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call') and part.function_call:
                                    fc = part.function_call
                                    print(f"\n🔧 Calling function: {fc.name}")

                                    # Execute function
                                    if fc.name in self.function_registry:
                                        args = dict(fc.args) if hasattr(fc, 'args') else {}
                                        result = self.function_registry[fc.name](**args)
                                        print(f"   Result: {result}")
                                        full_text += f"\n[Function {fc.name} called with result: {result}]"

            print()  # Newline after response

            # Show grounding metadata if available (Google Search results)
            if self.use_search and hasattr(chunk, 'candidates'):
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        gm = candidate.grounding_metadata

                        # Show search queries
                        if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
                            print(f"\n🔍 Search queries: {', '.join(gm.web_search_queries)}")

                        # Show sources
                        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                            print(f"\n📚 Sources ({len(gm.grounding_chunks)}):")
                            for i, chunk_data in enumerate(gm.grounding_chunks[:5], 1):  # Show first 5
                                if hasattr(chunk_data, 'web'):
                                    print(f"  {i}. {chunk_data.web.title}: {chunk_data.web.uri}")

            # Add assistant response to history
            self.conversation.append({
                "role": "model",
                "parts": [{"text": full_text}]
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
            f.write(f"# Gemini Chat Session\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.current_model}\n")
            f.write(f"**Functions:** {'Enabled' if self.use_functions else 'Disabled'}\n\n")
            f.write("---\n\n")

            for msg in self.conversation:
                role = msg.get('role', 'unknown')
                parts = msg.get('parts', [])
                content = str(parts[0].get('text', '')) if parts else ''

                if role == 'user':
                    f.write(f"## 👤 User\n\n{content}\n\n")
                elif role == 'model':
                    f.write(f"## 🤖 Gemini\n\n{content}\n\n")

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
            return await self._get_input_fallback(prompt)

    async def _get_input_fallback(self, prompt: str) -> str:
        """Fallback input method using select for paste detection"""
        import select

        loop = asyncio.get_event_loop()

        def read_input():
            sys.stdout.write(prompt)
            sys.stdout.flush()

            first_line = sys.stdin.readline()
            if not first_line:
                return ""

            lines = [first_line.rstrip('\n')]

            if hasattr(select, 'select'):
                try:
                    while select.select([sys.stdin], [], [], 0.0)[0]:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        lines.append(line.rstrip('\n'))
                except (OSError, ValueError):
                    pass

            result = '\n'.join(lines) if len(lines) > 1 else lines[0]
            return result

        return await loop.run_in_executor(None, read_input)

    async def run(self):
        """Main chat loop"""
        print("🚀 Gemini Interactive Chat")
        print("=" * 50)
        if HAS_PROMPT_TOOLKIT:
            print("✨ Enhanced mode: Multi-line paste | History | Auto-complete | Auto-suggest")
            print("💡 Tab for command completion | ↑/↓ for history | Type /help for commands")
        else:
            print("💡 Paste multi-line text naturally, /help for commands")
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
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("""
Gemini Interactive Chat

Usage: python gemini_chat.py [options]

Options:
  -h, --help     Show this help
  --model MODEL  Start with specific model
  --functions    Enable function calling by default

Examples:
  python gemini_chat.py
  python gemini_chat.py --model gemini-2.5-pro
  python gemini_chat.py --functions

Environment:
  GEMINI_API_KEY or GOOGLE_GENAI_API_KEY   Your Google AI API key (required)
            """)
            return

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Google AI API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return

    # Initialize chat
    chat = GeminiChat()

    # Parse command line options
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--model' and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            if model in ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash']:
                chat.current_model = model
        elif arg == '--search':
            chat.use_search = True
        elif arg == '--functions':
            chat.use_functions = True

    # Start chat
    await chat.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
