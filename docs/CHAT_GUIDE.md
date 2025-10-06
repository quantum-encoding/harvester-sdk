# Harvester SDK - Interactive Chat Guide

Complete guide to using the streaming chat feature across all AI providers.

## Quick Start

```bash
# Default (Grok)
harvester chat

# Specific providers
harvester chat --provider gemini
harvester chat --provider openai
harvester chat --provider claude
harvester chat --provider deepseek
```

## Supported Providers

### 🚀 Grok (xAI) - Default
```bash
harvester chat
harvester chat --provider grok
harvester chat --provider grok --model grok-3-mini
harvester chat --search                    # Enable live web search
harvester chat --functions                 # Enable function calling
harvester chat --search --functions        # Enable both
```

**Features:**
- ✅ Streaming responses
- ✅ Live web search (X, news, web)
- ✅ Function calling (calculate, get_time)
- ✅ Multi-line paste
- ✅ Command history

**Models:**
- `grok-4` (default) - Flagship reasoning model
- `grok-3` - High-performance
- `grok-3-mini` - Fast, efficient

**History:** `~/.grok_chat_history`

---

### 💎 Gemini (Google AI)
```bash
harvester chat --provider gemini
harvester chat --provider gemini --model gemini-2.5-pro
harvester chat --provider gemini --functions
```

**Features:**
- ✅ Streaming responses
- ✅ Function calling (calculate, get_time, get_weather)
- ✅ Multi-line paste
- ✅ Command history

**Models:**
- `gemini-2.5-flash` (default) - Fast, balanced
- `gemini-2.5-pro` - Most capable
- `gemini-2.5-flash-lite` - Lightweight
- `gemini-2.0-flash` - Previous generation

**History:** `~/.gemini_chat_history`

---

### 🤖 OpenAI (GPT)
```bash
harvester chat --provider openai
harvester chat --provider gpt               # Alias
harvester chat --provider openai --model gpt-5
```

**Features:**
- ✅ Streaming responses
- ✅ Multi-line paste
- ✅ Command history
- ✅ System prompts

**Models:**
- `gpt-4o` (default) - Fast, multimodal
- `gpt-5` - Latest, most capable
- `gpt-4o-mini` - Efficient
- `gpt-4-turbo` - Turbo variant
- `o1` - Reasoning model
- `o1-mini` - Compact reasoning

**History:** `~/.openai_chat_history`

---

### 🎭 Claude (Anthropic)
```bash
harvester chat --provider claude
harvester chat --provider anthropic         # Alias
harvester chat --provider claude --model claude-opus-4-20250514
```

**Features:**
- ✅ Streaming responses
- ✅ Multi-line paste
- ✅ Command history
- ✅ System prompts

**Models:**
- `claude-sonnet-4-20250514` (default) - Latest Sonnet
- `claude-opus-4-20250514` - Most capable
- `claude-3-7-sonnet-20250219` - Sonnet 3.7
- `claude-3-5-haiku-20241022` - Fast Haiku

**History:** `~/.claude_chat_history`

---

### 💡 DeepSeek
```bash
harvester chat --provider deepseek
harvester chat --provider deepseek --model reasoner
```

**Features:**
- ✅ Streaming responses
- ✅ Multi-line paste
- ✅ Command history
- ✅ Cost-effective

**Models:**
- `chat` (default) - DeepSeek Chat
- `reasoner` - DeepSeek Reasoner (R1)

**History:** `~/.deepseek_chat_history`

---

## Universal Commands

All chat interfaces support these slash commands:

```bash
/help           # Show help and available commands
/model <name>   # Switch model mid-conversation
/system <text>  # Set system prompt (OpenAI, Claude)
/search         # Toggle search (Grok only)
/functions      # Toggle function calling (Grok, Gemini)
/clear          # Clear conversation history
/history        # Show conversation history
/save <file>    # Save conversation to JSON
/load <file>    # Load conversation from JSON
/export <file>  # Export as Markdown
/paste          # Multi-line input mode (type END to finish)
/quit, /exit    # Exit the chat
```

## Enhanced Features

All chats include `prompt_toolkit` integration:

- **Multi-line paste**: Just paste - handled automatically
- **Command history**: Use ↑/↓ arrows to navigate
- **Auto-completion**: Press Tab to complete `/` commands
- **Auto-suggestions**: Previous commands suggested as you type
- **Line editing**: Ctrl+A, Ctrl+E, Ctrl+K, and all readline shortcuts

## Examples

### Simple Chat
```bash
$ harvester chat
🚀 Grok Interactive Chat
==================================================
✨ Enhanced mode: Multi-line paste | History | Auto-complete | Auto-suggest

👤 You: hello!
🤖 Grok: Hello! How can I help you today?
```

### With Search
```bash
$ harvester chat --search
👤 You: what's the latest news about AI?
🤖 Grok: [searches web and provides results with citations]
📚 Sources:
  1. https://...
  2. https://...
```

### With Functions
```bash
$ harvester chat --functions
👤 You: calculate 15 * 23
🤖 Grok: [calls calculate function]
Result: 345
```

### Switch Providers
```bash
# Start with Grok
$ harvester chat
👤 You: /exit

# Switch to Gemini
$ harvester chat --provider gemini
👤 You: hello from gemini!

# Switch to Claude
$ harvester chat --provider claude
👤 You: hello from claude!
```

### Save Conversations
```bash
👤 You: /save my_conversation.json
✓ Conversation saved to my_conversation.json

👤 You: /export my_chat.md
✓ Conversation exported to my_chat.md
```

## Environment Variables

Set API keys before using each provider:

```bash
export XAI_API_KEY="your-grok-key"
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-claude-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

## Advanced Usage

### Model-Specific Configurations

**Grok with everything enabled:**
```bash
harvester chat --provider grok --model grok-4 --search --functions
```

**Gemini with functions:**
```bash
harvester chat --provider gemini --model gemini-2.5-pro --functions
```

**Claude with specific model:**
```bash
harvester chat --provider claude --model claude-opus-4-20250514
```

### Multi-line Input

Two ways to handle multi-line input:

1. **Natural paste** - Just paste and press Enter
2. **/paste command** - Type `/paste`, enter text, type `END`

### Keyboard Shortcuts

- `↑/↓` - Navigate command history
- `Tab` - Auto-complete commands
- `Ctrl+C` - Exit chat
- `Ctrl+A` - Move to start of line
- `Ctrl+E` - Move to end of line
- `Ctrl+K` - Delete to end of line

## Troubleshooting

**Missing API key:**
```
❌ Error: XAI_API_KEY environment variable not set
```
→ Set the required API key for your chosen provider

**prompt_toolkit not installed:**
```
⚠️  Install prompt_toolkit for better experience: pip install prompt_toolkit
```
→ Install for enhanced features: `pip install prompt_toolkit`

**Provider not recognized:**
```
❌ Unknown provider: xyz
Available providers: grok, gemini, openai, claude, deepseek
```
→ Use one of the supported providers

---

**© 2025 Quantum Encoding Ltd**
Open Source - MIT License
