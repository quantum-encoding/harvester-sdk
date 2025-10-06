# Harvester SDK - CLI Reference

**Open Source MIT License** - All features available to everyone

## Unified CLI: `harvester`

The Harvester SDK provides a single, unified command-line interface for all AI operations:

```bash
harvester [COMMAND] [OPTIONS]
```

### Core Commands

#### 📝 Text Processing
- **`batch`** - Process CSV batch with AI providers (text or image)
- **`process`** - Process directory with templates (formerly batch_code)
- **`message`** - Turn-based conversation with AI (non-streaming)
- **`json`** - Process single JSON request with AI

#### 💬 Interactive Chat
- **`chat`** - Start interactive streaming chat with AI provider
  - Enhanced with `prompt_toolkit` for better UX
  - Multi-line paste support
  - Command history (↑/↓ arrows)
  - Professional line editing (Ctrl+A, Ctrl+E, Ctrl+K, etc.)
  - Supports: Grok, DeepSeek, OpenAI, Anthropic, Google, and more

#### 🎨 Image Generation
- **`image`** - Generate images with AI models (DALL-E, Imagen, etc.)
  - Single image generation
  - Batch processing from CSV
  - Style templates support

#### 🔍 Search & Research
- **`search`** - Search the web with AI-enhanced results (Grok)

#### ⚙️ Configuration & Management
- **`config`** - Manage SDK configuration and API keys
- **`templates`** - Manage batch processing templates
- **`list-models`** - List available models and providers
- **`status`** - Check batch job status
- **`convert`** - Convert any file format to CSV for batch processing

#### 🔧 Advanced Features
- **`structured`** - Generate structured output with schema validation
- **`functions`** - Execute functions and tools

## Usage Examples

### Interactive Chat
```bash
# Start Grok chat (default)
harvester chat

# Chat with DeepSeek
harvester chat --provider deepseek

# Chat with specific model
harvester chat --provider openai --model gpt-4

# Enable live search (Grok only)
harvester chat --provider grok --search

# Chat Features:
# - Natural multi-line paste support
# - Command history with ↑/↓ arrows
# - /help for commands
# - /paste for explicit multi-line mode
# - /export to save as markdown
```

### Batch Processing
```bash
# Process CSV with text prompts
harvester batch data.csv --model gemini-2.5-flash --template default

# Process directory with templates
harvester process ~/my-code --template refactor --model gpt-5

# Batch image generation
harvester image --batch prompts.csv
```

### Single Operations
```bash
# Generate single image
harvester image "sunset over mountains" --model dalle-3

# Turn-based conversation
harvester message --model gemini-2.5-flash --system "You are a helpful assistant"

# Web search
harvester search "latest AI news" --provider grok
```

### Configuration
```bash
# Show current config
harvester config --show

# List available models
harvester list-models

# Manage templates
harvester templates --list
harvester templates --copy image_batch_blog
```

## Chat Command Reference

The `harvester chat` command provides an enhanced interactive experience:

**Enhanced Features (with `prompt_toolkit`):**
- **Multi-line paste**: Just paste - no special mode needed
- **Persistent history**: History saved to `~/.grok_chat_history` across sessions
- **Command auto-completion**: Press Tab to complete `/` commands
- **Auto-suggestions**: Previous commands suggested as you type
- **Line editing**: Ctrl+A (start), Ctrl+E (end), Ctrl+K (delete to end), and more
- **Commands**: Type `/help` to see all available commands

**Available Commands:**
- `/help` - Show help information
- `/model <name>` - Switch model (e.g., `/model grok-3-mini`)
- `/search` - Toggle live search on/off (Grok only)
- `/functions` - Toggle function calling on/off
- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/save <file>` - Save conversation to JSON
- `/load <file>` - Load conversation from JSON
- `/export <file>` - Export conversation as markdown
- `/paste` - Enter multi-line input mode (type END to finish)
- `/quit`, `/exit` - Exit the chat

**Supported Providers:**
- Grok (xAI) - `--provider grok`
- DeepSeek - `--provider deepseek`
- OpenAI - `--provider openai`
- Anthropic - `--provider anthropic`
- Google - Use via message command

## Installation

```bash
# Install the SDK
pip install -r requirements.txt

# Or install just the SDK dependencies
pip install harvester-sdk

# Verify installation
python -c "from harvester_sdk import HarvesterSDK; print('SDK imported successfully')"

# Test the CLI
harvester --help
```

## Requirements

**Core Dependencies:**
- Python 3.8+
- `prompt_toolkit` - Enhanced terminal input (installed automatically)
- API keys for desired providers (set via environment variables)

**Environment Variables:**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export XAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```