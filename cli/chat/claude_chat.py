#!/usr/bin/env python3
"""
Interactive Claude Chat with Streaming
A terminal-based chat interface for Anthropic's Claude models with real-time streaming
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
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Load environment variables
load_dotenv()

class ClaudeChat:
    def __init__(self):
        if not HAS_ANTHROPIC:
            raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")

        # Get API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable")

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        self.conversation = []
        self.current_model = "claude-sonnet-4-20250514"
        self.system_prompt = "You are Claude, a helpful AI assistant."
        self.use_search = False
        self.use_fetch = False
        self.use_functions = False
        self.use_memory = False
        self.use_code_execution = False
        self.function_tools = []
        self.memory_dir = Path.home() / '.claude_memories'
        self.container_id = None  # For code execution container reuse
        self.setup_function_tools()
        self.setup_memory_dir()

        # Setup prompt session
        if HAS_PROMPT_TOOLKIT:
            chat_commands = [
                '/help', '/quit', '/exit', '/model', '/system', '/search', '/fetch', '/functions', '/memory', '/code',
                '/clear', '/history', '/save', '/load', '/export', '/paste'
            ]
            completer = WordCompleter(chat_commands, ignore_case=True, sentence=True)
            history_file = os.path.expanduser('~/.claude_chat_history')

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
                "name": "calculate",
                "description": "Perform mathematical calculations. Use this when the user asks you to calculate something or evaluate a mathematical expression.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'pow(2, 8)')"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_current_time",
                "description": "Get the current time in a specific timezone. Use this when the user asks about the current time in a location.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone name (e.g., 'America/New_York', 'Europe/London', 'UTC')"
                        }
                    },
                    "required": ["timezone"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get current weather information for a location. This is a mock function that returns sample data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name or location (e.g., 'San Francisco, CA', 'London, UK')"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]

    def setup_memory_dir(self):
        """Setup memory directory"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def execute_function(self, function_name: str, arguments: dict) -> str:
        """Execute a function call and return result"""
        if function_name == "calculate":
            try:
                expression = arguments.get("expression", "")
                # Safe eval with limited builtins
                result = eval(expression, {"__builtins__": {}}, {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len, "pow": pow, "sqrt": lambda x: x**0.5
                })
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"

        elif function_name == "get_current_time":
            try:
                import pytz
                tz = pytz.timezone(arguments.get("timezone", "UTC"))
                current_time = datetime.now(tz)
                return f"Current time in {arguments.get('timezone')}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            except Exception as e:
                return f"Error: {str(e)}"

        elif function_name == "get_weather":
            location = arguments.get("location", "Unknown")
            # Mock weather data
            return f"Weather in {location}: 22°C, partly cloudy, humidity 65%"

        return f"Error: Unknown function '{function_name}'"

    def handle_memory_command(self, command: str, **kwargs) -> str:
        """Handle memory tool commands"""
        path = kwargs.get('path', '')

        # Security: ensure path is within memory directory
        if path:
            full_path = (self.memory_dir / path.lstrip('/')).resolve()
            try:
                full_path.relative_to(self.memory_dir)
            except ValueError:
                return "Error: Access denied - path outside memory directory"

        if command == "view":
            if not path or path == "/memories":
                # List directory contents
                files = list(self.memory_dir.glob('*'))
                if not files:
                    return "Directory: /memories\n(empty)"
                file_list = "\n".join(f"- {f.name}" for f in files if f.is_file())
                return f"Directory: /memories\n{file_list}"
            else:
                # Read file contents
                file_path = self.memory_dir / path.lstrip('/').removeprefix('memories/')
                if not file_path.exists():
                    return f"Error: File not found: {path}"
                try:
                    content = file_path.read_text()
                    view_range = kwargs.get('view_range')
                    if view_range:
                        lines = content.split('\n')
                        start, end = view_range
                        content = '\n'.join(lines[start-1:end])
                    return content
                except Exception as e:
                    return f"Error reading file: {str(e)}"

        elif command == "create":
            file_path = self.memory_dir / path.lstrip('/').removeprefix('memories/')
            file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                file_path.write_text(kwargs.get('file_text', ''))
                return f"File created: {path}"
            except Exception as e:
                return f"Error creating file: {str(e)}"

        elif command == "str_replace":
            file_path = self.memory_dir / path.lstrip('/').removeprefix('memories/')
            if not file_path.exists():
                return f"Error: File not found: {path}"
            try:
                content = file_path.read_text()
                old_str = kwargs.get('old_str', '')
                new_str = kwargs.get('new_str', '')
                if old_str not in content:
                    return f"Error: String not found in file"
                content = content.replace(old_str, new_str)
                file_path.write_text(content)
                return f"File updated: {path}"
            except Exception as e:
                return f"Error updating file: {str(e)}"

        elif command == "insert":
            file_path = self.memory_dir / path.lstrip('/').removeprefix('memories/')
            if not file_path.exists():
                return f"Error: File not found: {path}"
            try:
                lines = file_path.read_text().split('\n')
                insert_line = kwargs.get('insert_line', 1)
                insert_text = kwargs.get('insert_text', '')
                lines.insert(insert_line - 1, insert_text.rstrip('\n'))
                file_path.write_text('\n'.join(lines))
                return f"Text inserted in {path}"
            except Exception as e:
                return f"Error inserting text: {str(e)}"

        elif command == "delete":
            file_path = self.memory_dir / path.lstrip('/').removeprefix('memories/')
            if not file_path.exists():
                return f"Error: File not found: {path}"
            try:
                if file_path.is_file():
                    file_path.unlink()
                else:
                    import shutil
                    shutil.rmtree(file_path)
                return f"Deleted: {path}"
            except Exception as e:
                return f"Error deleting: {str(e)}"

        elif command == "rename":
            old_path = self.memory_dir / kwargs.get('old_path', '').lstrip('/').removeprefix('memories/')
            new_path = self.memory_dir / kwargs.get('new_path', '').lstrip('/').removeprefix('memories/')
            if not old_path.exists():
                return f"Error: File not found: {kwargs.get('old_path')}"
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
                return f"Renamed: {kwargs.get('old_path')} -> {kwargs.get('new_path')}"
            except Exception as e:
                return f"Error renaming: {str(e)}"

        return f"Error: Unknown memory command '{command}'"

    def print_help(self):
        """Print help information"""
        print("""
Available Commands:
  /help          - Show this help
  /model <name>  - Switch model
  /search        - Toggle web search on/off
  /fetch         - Toggle web fetch on/off
  /functions     - Toggle function calling on/off
  /memory        - Toggle memory tool on/off
  /code          - Toggle code execution on/off
  /system <text> - Set system prompt
  /clear         - Clear conversation history
  /history       - Show conversation history
  /save <file>   - Save conversation to file
  /load <file>   - Load conversation from file
  /export <file> - Export conversation as markdown
  /paste         - Enter multi-line input mode
  /quit, /exit   - Exit the chat

Current Settings:
  Model: {self.current_model}
  Web Search: {'ON' if self.use_search else 'OFF'}
  Web Fetch: {'ON' if self.use_fetch else 'OFF'}
  Functions: {'ON' if self.use_functions else 'OFF'}
  Memory: {'ON' if self.use_memory else 'OFF'}
  Code Execution: {'ON' if self.use_code_execution else 'OFF'}
  Container ID: {self.container_id if self.container_id else 'None'}
  System: {self.system_prompt[:50]}...

Models Available:
  - claude-sonnet-4-20250514      : Latest Sonnet (best)
  - claude-opus-4-20250514        : Most capable
  - claude-3-7-sonnet-20250219    : Sonnet 3.7
  - claude-3-5-haiku-20241022     : Fast Haiku
        """.format(self=self))

    def print_status(self):
        """Print current status"""
        print(f"\n🤖 Claude Chat - Model: {self.current_model} | Search: {'ON' if self.use_search else 'OFF'} | Fetch: {'ON' if self.use_fetch else 'OFF'} | Functions: {'ON' if self.use_functions else 'OFF'} | Memory: {'ON' if self.use_memory else 'OFF'} | Code: {'ON' if self.use_code_execution else 'OFF'}")
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

        elif cmd == '/fetch':
            self.use_fetch = not self.use_fetch
            print(f"✓ Web fetch {'enabled' if self.use_fetch else 'disabled'}")

        elif cmd == '/functions':
            self.use_functions = not self.use_functions
            print(f"✓ Function calling {'enabled' if self.use_functions else 'disabled'}")

        elif cmd == '/memory':
            self.use_memory = not self.use_memory
            if self.use_memory:
                print(f"✓ Memory tool enabled (directory: {self.memory_dir})")
            else:
                print(f"✓ Memory tool disabled")

        elif cmd == '/code':
            self.use_code_execution = not self.use_code_execution
            if self.use_code_execution:
                print(f"✓ Code execution enabled")
                if self.container_id:
                    print(f"  Container: {self.container_id}")
            else:
                print(f"✓ Code execution disabled")

        elif cmd == '/system':
            if len(parts) > 1:
                self.system_prompt = parts[1]
                print(f"✓ System prompt updated")
            else:
                print(f"Current system: {self.system_prompt}")

        elif cmd == '/clear':
            self.conversation = []
            print("✓ Conversation cleared")

        elif cmd == '/history':
            if not self.conversation:
                print("No conversation history")
            else:
                for i, msg in enumerate(self.conversation):
                    role = msg['role'].title()
                    content = msg['content']
                    if isinstance(content, list):
                        content = content[0].get('text', '')[:100]
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
            print("❌ Unknown command. Type /help for available commands.")

        return True

    async def chat(self, user_input: str):
        """Handle chat interaction with streaming and tool use"""
        try:
            # Add user message to history
            self.conversation.append({
                "role": "user",
                "content": user_input
            })

            # Prepare tools
            tools = []
            betas = []

            # Add web search
            if self.use_search:
                tools.append({
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                })

            # Add web fetch
            if self.use_fetch:
                tools.append({
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_uses": 10,
                    "citations": {
                        "enabled": True
                    },
                    "max_content_tokens": 100000
                })
                if "web-fetch-2025-09-10" not in betas:
                    betas.append("web-fetch-2025-09-10")

            # Add function calling tools
            if self.use_functions:
                tools.extend(self.function_tools)

            # Add memory tool
            if self.use_memory:
                tools.append({
                    "type": "memory_20250818",
                    "name": "memory"
                })
                if "context-management-2025-06-27" not in betas:
                    betas.append("context-management-2025-06-27")

            # Add code execution tool
            if self.use_code_execution:
                tools.append({
                    "type": "code_execution_20250825",
                    "name": "code_execution"
                })
                if "code-execution-2025-08-25" not in betas:
                    betas.append("code-execution-2025-08-25")

            # Tool use loop - keep calling until we get a final response
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Prepare request
                request_kwargs = {
                    "model": self.current_model,
                    "max_tokens": 4096,
                    "system": self.system_prompt,
                    "messages": self.conversation
                }

                if tools:
                    request_kwargs["tools"] = tools

                # Add container for code execution reuse
                if self.container_id and self.use_code_execution:
                    request_kwargs["container"] = self.container_id

                if betas:
                    # Use beta client for memory/code execution
                    response = self.client.beta.messages.create(
                        **request_kwargs,
                        betas=betas
                    )
                else:
                    # Regular client
                    response = self.client.messages.create(**request_kwargs)

                # Extract and store container ID if code execution was used
                if self.use_code_execution and hasattr(response, 'container') and response.container:
                    self.container_id = response.container.id

                # Display assistant's text response
                if iteration == 1:
                    print("🤖 Claude: ", end="", flush=True)

                for block in response.content:
                    if hasattr(block, 'type') and block.type == 'text':
                        print(block.text, end="", flush=True)

                print()  # Newline

                # Check if we need to handle tool use
                tool_uses = [block for block in response.content if hasattr(block, 'type') and block.type == 'tool_use']

                if not tool_uses:
                    # No tool use, we're done
                    # Extract search and fetch metadata if available
                    search_queries = []
                    sources = []
                    fetched_urls = []

                    for block in response.content:
                        # Handle web search
                        if hasattr(block, 'type') and block.type == 'server_tool_use':
                            if hasattr(block, 'name') and block.name == 'web_search':
                                if hasattr(block, 'input') and 'query' in block.input:
                                    search_queries.append(block.input['query'])
                            elif hasattr(block, 'name') and block.name == 'web_fetch':
                                if hasattr(block, 'input') and 'url' in block.input:
                                    fetched_urls.append(block.input['url'])

                        # Handle web fetch results
                        if hasattr(block, 'type') and block.type == 'web_fetch_tool_result':
                            if hasattr(block, 'content') and hasattr(block.content, 'url'):
                                url = block.content.url
                                title = getattr(block.content.content, 'title', url) if hasattr(block.content, 'content') else url
                                sources.append({
                                    'url': url,
                                    'title': title,
                                    'type': 'fetch'
                                })

                        # Handle citations
                        if hasattr(block, 'citations'):
                            for citation in block.citations:
                                if hasattr(citation, 'type') and citation.type == 'web_search_result_location':
                                    sources.append({
                                        'url': citation.url,
                                        'title': citation.title,
                                        'type': 'search'
                                    })
                                elif hasattr(citation, 'type') and citation.type == 'char_location':
                                    # Fetch citation
                                    if hasattr(citation, 'document_title'):
                                        sources.append({
                                            'url': fetched_urls[citation.document_index] if hasattr(citation, 'document_index') and citation.document_index < len(fetched_urls) else 'unknown',
                                            'title': citation.document_title,
                                            'type': 'fetch'
                                        })

                    # Display search metadata
                    if search_queries:
                        print(f"\n🔍 Search queries: {', '.join(search_queries)}")

                    if fetched_urls:
                        print(f"\n📄 Fetched content from:")
                        for url in fetched_urls:
                            print(f"  • {url}")

                    if sources:
                        unique_sources = []
                        seen_urls = set()
                        for source in sources:
                            if source['url'] not in seen_urls:
                                unique_sources.append(source)
                                seen_urls.add(source['url'])

                        print(f"\n📚 Sources ({len(unique_sources)}):")
                        for i, source in enumerate(unique_sources[:10], 1):  # Show up to 10 sources
                            title = source['title'][:80] + "..." if len(source['title']) > 80 else source['title']
                            type_emoji = "🔍" if source.get('type') == 'search' else "📄"
                            print(f"  {i}. {type_emoji} {title}")
                            print(f"     {source['url']}")

                    # Add to conversation and break
                    self.conversation.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    break

                # Handle tool calls
                print()  # Extra newline before tool calls
                tool_results = []

                for tool_use in tool_uses:
                    tool_name = tool_use.name
                    tool_input = tool_use.input
                    tool_id = tool_use.id

                    print(f"🔧 Calling tool: {tool_name}")

                    # Execute the tool
                    if tool_name == "memory":
                        # Handle memory tool
                        command = tool_input.get('command', '')
                        result = self.handle_memory_command(command, **tool_input)
                    else:
                        # Handle function tools
                        result = self.execute_function(tool_name, tool_input)

                    print(f"   Result: {result[:100]}{'...' if len(result) > 100 else ''}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result
                    })

                # Add assistant message with tool uses
                self.conversation.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Add user message with tool results
                self.conversation.append({
                    "role": "user",
                    "content": tool_results
                })

                print()  # Newline before next iteration

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def export_conversation_markdown(self, filename: str):
        """Export conversation as markdown"""
        if not filename.endswith('.md'):
            filename += '.md'

        with open(filename, 'w') as f:
            f.write(f"# Claude Chat Session\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.current_model}\n")
            f.write(f"**System:** {self.system_prompt}\n\n")
            f.write("---\n\n")

            for msg in self.conversation:
                role = msg['role']
                content = msg['content']
                if isinstance(content, list):
                    content = content[0].get('text', '')

                if role == 'user':
                    f.write(f"## 👤 User\n\n{content}\n\n")
                elif role == 'assistant':
                    f.write(f"## 🤖 Claude\n\n{content}\n\n")

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
        print("🚀 Claude Interactive Chat")
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
Claude Interactive Chat

Usage: python claude_chat.py [options]

Options:
  -h, --help      Show this help
  --model MODEL   Start with specific model
  --search        Enable web search by default
  --fetch         Enable web fetch by default
  --functions     Enable function calling by default
  --memory        Enable memory tool by default
  --code          Enable code execution by default

Environment:
  ANTHROPIC_API_KEY   Your Anthropic API key (required)
        """)
        return

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    # Initialize chat
    chat = ClaudeChat()

    # Parse command line options
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--model' and i + 1 < len(sys.argv):
            chat.current_model = sys.argv[i + 1]
        elif arg == '--search':
            chat.use_search = True
        elif arg == '--fetch':
            chat.use_fetch = True
        elif arg == '--functions':
            chat.use_functions = True
        elif arg == '--memory':
            chat.use_memory = True
        elif arg == '--code':
            chat.use_code_execution = True

    # Start chat
    await chat.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
