#!/usr/bin/env python3
"""
Interactive Grok Chat with Streaming
A terminal-based chat interface for xAI's Grok models with real-time streaming
"""
import asyncio
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from providers.xai_streaming import XaiStreamingProvider
from providers.xai_functions import XaiFunctionCaller

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

# Load environment variables
load_dotenv()

class GrokChat:
    def __init__(self):
        self.streaming_provider = XaiStreamingProvider()
        self.function_caller = XaiFunctionCaller()
        self.conversation = []
        self.current_model = "grok-4"
        self.use_search = False
        self.use_functions = False
        self.setup_basic_tools()

        # Setup prompt session with history if prompt_toolkit is available
        if HAS_PROMPT_TOOLKIT:
            # Command completer for slash commands
            chat_commands = [
                '/help', '/quit', '/exit', '/model', '/search', '/functions',
                '/clear', '/history', '/save', '/load', '/export', '/paste'
            ]
            completer = WordCompleter(chat_commands, ignore_case=True, sentence=True)

            # Use file-based history for persistence across sessions
            history_file = os.path.expanduser('~/.grok_chat_history')

            self.prompt_session = PromptSession(
                history=FileHistory(history_file),
                completer=completer,
                auto_suggest=AutoSuggestFromHistory()
            )
        else:
            self.prompt_session = None
    
    def setup_basic_tools(self):
        """Setup basic utility tools"""
        def calculate(expression: str):
            try:
                result = eval(expression, {"__builtins__": {}}, {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len, "pow": pow, "sqrt": lambda x: x**0.5
                })
                return {"expression": expression, "result": result}
            except Exception as e:
                return {"error": str(e)}
        
        def get_time(timezone: str = "UTC"):
            from datetime import datetime
            import pytz
            try:
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz)
                return {
                    "timezone": timezone,
                    "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": current_time.timestamp()
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.function_caller.register_tool("calculate", "Perform mathematical calculations", calculate)
        self.function_caller.register_tool("get_time", "Get current time", get_time)
    
    def print_help(self):
        """Print help information"""
        print("""
Available Commands:
  /help          - Show this help
  /model <name>  - Switch model (grok-4, grok-3, grok-3-mini)
  /search        - Toggle Live Search on/off
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
  Live Search: {'ON' if self.use_search else 'OFF'}
  Functions: {'ON' if self.use_functions else 'OFF'}
  
Models Available:
  - grok-4      : Flagship reasoning model (256k context)
  - grok-3      : High-performance model (128k context)
  - grok-3-mini : Fast, efficient with reasoning trace
        """.format(self=self))
    
    def print_status(self):
        """Print current status"""
        print(f"\n🤖 Grok Chat - Model: {self.current_model} | Search: {'ON' if self.use_search else 'OFF'} | Functions: {'ON' if self.use_functions else 'OFF'}")
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
                if model in ['grok-4', 'grok-3', 'grok-3-mini']:
                    self.current_model = model
                    print(f"✓ Switched to {model}")
                else:
                    print("❌ Invalid model. Use: grok-4, grok-3, or grok-3-mini")
            else:
                print(f"Current model: {self.current_model}")
        
        elif cmd == '/search':
            self.use_search = not self.use_search
            print(f"✓ Live Search {'enabled' if self.use_search else 'disabled'}")
        
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
                    role = msg['role'].title()
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"{i+1}. {role}: {content}")
        
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
                await self.chat_with_streaming(multi_input)
                print()
            else:
                print("❌ No input provided")
        
        else:
            print("❌ Unknown command. Type /help for available commands.")
        
        return True
    
    async def chat_with_streaming(self, user_input: str):
        """Handle chat with streaming response"""
        # Add user message
        user_msg = {"role": "user", "content": user_input}
        self.conversation.append(user_msg)
        
        # Prepare search parameters if enabled
        search_params = None
        if self.use_search:
            search_params = {
                "mode": "auto",
                "return_citations": True,
                "sources": [
                    {"type": "web"},
                    {"type": "x"},
                    {"type": "news"}
                ]
            }
        
        try:
            print("🤖 Grok: ", end="", flush=True)
            
            if self.use_functions:
                # Use function calling (non-streaming for now)
                self.function_caller.conversation_messages = self.conversation.copy()
                result = await self.function_caller.process_conversation(
                    self.conversation,
                    model=self.current_model
                )
                response_text = result["response"]
                print(response_text)
                
                # Update conversation with full history
                self.conversation = result["messages"]
                
            else:
                # Use streaming
                response_text = ""
                async for response, chunk in self.streaming_provider.stream_completion(
                    messages=self.conversation,
                    model=self.current_model,
                    search_parameters=search_params
                ):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        response_text = response.content
                
                print()  # New line after streaming
                
                # Add assistant response to conversation
                assistant_msg = {"role": "assistant", "content": response_text}
                self.conversation.append(assistant_msg)
                
                # Show citations if available
                if hasattr(response, 'citations') and response.citations:
                    print("\n📚 Sources:")
                    for i, citation in enumerate(response.citations, 1):
                        print(f"  {i}. {citation}")
                
                # Show usage if available
                if hasattr(response, 'usage') and response.usage:
                    try:
                        if isinstance(response.usage, dict):
                            total_tokens = response.usage.get('total_tokens', 0)
                        else:
                            total_tokens = getattr(response.usage, 'total_tokens', 0)
                        if total_tokens > 0:
                            print(f"\n💰 Tokens used: {total_tokens}")
                    except Exception:
                        pass
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def export_conversation_markdown(self, filename: str):
        """Export conversation as markdown"""
        if not filename.endswith('.md'):
            filename += '.md'
        
        with open(filename, 'w') as f:
            f.write(f"# Grok Chat Session\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.current_model}\n")
            f.write(f"**Live Search:** {'Enabled' if self.use_search else 'Disabled'}\n")
            f.write(f"**Functions:** {'Enabled' if self.use_functions else 'Disabled'}\n\n")
            f.write("---\n\n")
            
            for i, msg in enumerate(self.conversation):
                role = msg['role']
                content = msg['content']
                
                if role == 'user':
                    f.write(f"## 👤 User\n\n{content}\n\n")
                elif role == 'assistant':
                    f.write(f"## 🤖 Grok\n\n{content}\n\n")
                elif role == 'system':
                    f.write(f"## ⚙️ System\n\n{content}\n\n")
                
                f.write("---\n\n")
    
    async def get_input_async(self, prompt: str) -> str:
        """Get input with best available method"""
        if HAS_PROMPT_TOOLKIT:
            # Use prompt_toolkit for best experience
            try:
                # Run prompt_toolkit in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.prompt_session.prompt(prompt)
                )
                return result
            except (KeyboardInterrupt, EOFError):
                raise
        else:
            # Fallback to select-based paste detection
            return await self._get_input_fallback(prompt)

    async def _get_input_fallback(self, prompt: str) -> str:
        """Fallback input method using select for paste detection"""
        import select

        loop = asyncio.get_event_loop()

        # Run blocking input in executor
        def read_input():
            sys.stdout.write(prompt)
            sys.stdout.flush()

            # Read first line
            first_line = sys.stdin.readline()
            if not first_line:
                return ""

            lines = [first_line.rstrip('\n')]

            # Check if there's more data waiting (pasted content)
            if hasattr(select, 'select'):
                try:
                    while select.select([sys.stdin], [], [], 0.0)[0]:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        lines.append(line.rstrip('\n'))
                except (OSError, ValueError):
                    pass

            # Combine all lines
            result = '\n'.join(lines) if len(lines) > 1 else lines[0]
            return result

        return await loop.run_in_executor(None, read_input)

    async def run(self):
        """Main chat loop"""
        print("🚀 Grok Interactive Chat")
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
                    # Get input with best available method
                    user_input = await self.get_input_async("👤 You: ")
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith('/'):
                        should_continue = await self.handle_command(user_input)
                        if not should_continue:
                            break
                        continue

                    # Process chat message
                    await self.chat_with_streaming(user_input)
                    print()  # Extra spacing

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break

        finally:
            pass  # Goodbye already printed

async def main():
    """Entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("""
Grok Interactive Chat

Usage: python grok_chat.py [options]

Options:
  -h, --help     Show this help
  --model MODEL  Start with specific model (grok-4, grok-3, grok-3-mini)
  --search       Enable Live Search by default
  --functions    Enable function calling by default

Examples:
  python grok_chat.py
  python grok_chat.py --model grok-3-mini --search
  python grok_chat.py --functions

Environment:
  XAI_API_KEY    Your xAI API key (required)
            """)
            return
    
    # Check API key
    if not os.getenv("XAI_API_KEY"):
        print("❌ Error: XAI_API_KEY environment variable not set")
        print("Please set your xAI API key:")
        print("export XAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize chat
    chat = GrokChat()
    
    # Parse command line options
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--model' and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            if model in ['grok-4', 'grok-3', 'grok-3-mini']:
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