# 🚀 Harvester SDK - Complete AI Processing Platform

> **"The unified interface for all AI providers with enterprise-grade reliability."**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Providers: 7+](https://img.shields.io/badge/Providers-OpenAI%20%7C%20Anthropic%20%7C%20GenAI%20%7C%20Vertex%20%7C%20XAI%20%7C%20DeepSeek-brightgreen.svg)](#providers)

## 🌟 What is Harvester SDK?

Harvester SDK is a comprehensive AI processing platform that provides a **unified interface** to all major AI providers. Whether you need text generation, image creation, batch processing, agentic coding, or real-time conversations, Harvester SDK handles the complexity so you can focus on building.

### ⚡ Key Features

- **Multi-Provider Support** - OpenAI, Anthropic, Google AI Studio, Vertex AI, XAI, DeepSeek
- **Agentic Coding Assistants** - Grok Code Agent (fast) & Claude Code Agent (SDK-powered)
- **Enhanced Chat Experience** - `prompt_toolkit` integration with multi-line paste, command history, and professional line editing
- **Dual Authentication** - API keys (GenAI) and service accounts (Vertex AI)
- **Streaming & Turn-Based Chat** - Real-time streaming or non-streaming conversations
- **Batch Processing** - Cost-effective bulk operations with 50% savings
- **Template System** - 30+ Jinja2 templates for AI-powered transformations
- **Image Generation** - DALL-E, Imagen, GPT Image support
- **Enterprise Ready** - Rate limiting, retries, error handling

## 🚀 Quick Start

### Installation

```bash
# Standard installation (all providers, no browser automation)
pip install harvester-sdk

# With computer use / browser automation support (includes Playwright)
pip install harvester-sdk[computer]

# After installing with [computer], download browser binaries:
playwright install
```

**Note:** The `[computer]` extra is only needed if you plan to use the `harvester computer` command for AI-powered browser automation. It adds ~600MB of browser downloads.

### Basic Usage

```bash
# Main CLI conductor
harvester --help

# Turn-based conversation (non-streaming)
harvester message --model gemini-2.5-flash
harvester message --model sonnet-4-5 --system "You are a helpful assistant"

# Batch processing from CSV
harvester batch data.csv --model gpt-5 --template quick

# Process directory with templates
harvester process ./src --template refactor --model gemini-2.5-pro

# Generate images
harvester image "A beautiful sunset" --provider dalle3 --size 1024x1024
```

## 🔧 Provider Configuration

### Google AI Studio (GenAI) - API Key Authentication
```bash
export GEMINI_API_KEY=your_api_key
harvester message --model gemini-2.5-flash
```

### Google Vertex AI - Service Account Authentication
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
harvester message --model vtx-gemini-2.5-flash
```

### Other Providers
```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export XAI_API_KEY=your_xai_key
export DEEPSEEK_API_KEY=your_deepseek_key
```

## 📋 Available Commands

### Core Commands
- `harvester chat` - **Interactive streaming chat** with enhanced UX (multi-line paste, history, line editing)
- `harvester message` - Turn-based conversations (non-streaming)
- `harvester batch` - Batch process CSV files
- `harvester process` - Directory processing with templates
- `harvester image` - Image generation (single or batch)
- `harvester search` - AI-enhanced web search (Grok)

### Agentic Commands
- `harvester agent-grok` - **Grok Code Agent** - Fast / impressive agentic coding (grok-code-fast-1)
- `harvester agent-claude` - **Claude Code Agent** - Prone to hallucinations, be careful Claude will delete the Claude Agent SDK
- `harvester agent-openai` - **OpenAI Code Agent** - File operations with GPT-4o/o1/o3-mini
- `harvester agent-gpt5` - **GPT-5 Code Agent** - Advanced reasoning for complex coding tasks
- `harvester code-interpreter` - **Code Interpreter** - Python code execution in sandboxed containers
- `harvester image-gen` - **Image Generation** - AI-powered image creation and editing
- `harvester computer` - **GPT Computer Use** - AI agent that controls browser/computer

### Utility Commands
- `harvester list-models` - Show available models
- `harvester config --show` - Display configuration
- `harvester templates` - Manage batch processing templates
- `harvester status` - Check batch job status

### Chat Features
The `harvester chat` command provides a professional terminal experience:
- ✅ **Multi-line paste support** - Natural paste behavior, no special modes
- ✅ **Command history** - Use ↑/↓ arrows to recall previous messages
- ✅ **Line editing** - Ctrl+A, Ctrl+E, Ctrl+K, and other readline shortcuts
- ✅ **Slash commands** - `/help`, `/model`, `/search`, `/export`, and more
- ✅ **Export conversations** - Save to JSON or Markdown

## 🎯 Model Selection Guide

### Google AI Models

| **API Key (GenAI)** | **Service Account (Vertex)** | **Use Case** |
|---------------------|------------------------------|--------------|
| `gemini-2.5-flash` | `vtx-gemini-2.5-flash` | Fast, cost-effective |
| `gemini-2.5-pro` | `vtx-gemini-2.5-pro` | High-quality reasoning |
| `gemini-2.5-flash-lite` | `vtx-gemini-2.5-flash-lite` | low latency |

### Other Providers
- **OpenAI**: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- **Anthropic**: `claude-sonnet-4-5`,`claude-sonnet-4`, `claude-opus-4-1`
- **XAI**: `grok-code-fast-1`,`grok-4-fast-reasoning`,`grok-4-fast`,`grok-4-0709`, `grok-3`, `grok-3-mini`
- **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`

## 🤖 Agentic Coding Assistants

Harvester SDK includes multiple powerful agentic coding assistants that can autonomously handle complex multi-step coding tasks.

### Grok Code Agent (`agent-grok`)

**Powered by xAI's `grok-code-fast-1` model** - The fastest, most cost-effective agentic coding solution.

**Features:**
- ⚡ **4x faster** than claude-code agents
- 💰 **1/10th the cost** of comparable solutions
- 🧰 **11 tools**: file operations, JSON tools, command execution, directory management
- 🔁 **100 max iterations** with loop detection on file reads
- 🎯 **Streaming reasoning traces** - Read the Agents thoughts after
- 🛡️ **Safety first** - Dangerous commands (rm -rf /, dd, fork bombs) automatically blocked
- 🚀 **Ripgrep support** - 10x faster code search when available

**Example:**

```bash
🤖 agent-grok "build programs from this list /path/to/file.md" --show-reasoning
🎯 Type: general

============================================================
🔄 Iteration 1/100
============================================================

🔧 Tool Calls (1)
  → read_file({'file_path': '/home/user/linux_c_programs/linux_c_program_taxonomy.md'})
    ✓ # Comprehensive Taxonomy of C Programs for Linux
## ~300 Feasible Program Ideas by Category

---

##...

============================================================
🔄 Iteration 2/100
============================================================

🔧 Tool Calls (1)
  → list_files({'path': '/home/user/linux_c_programs'})
    ✓ Success

============================================================
🔄 Iteration 3/100
============================================================

🔧 Tool Calls (1)
  → list_files({'path': '/home/user/linux_c_programs/libs'})
    ✓ Success

============================================================
🔄 Iteration 4/100
============================================================

🔧 Tool Calls (1)
  → read_file({'file_path': '/home/user/linux_c_programs/libs/common.h'})
    ✓ #ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#inclu...

```

### Claude Code Agent (`agent-claude`)

**Built on Anthropic's official Claude Agent SDK** - infrastructure from the team behind Claude Code CLI.

```bash
# Execute a coding task
harvester agent-claude "Implement a REST API endpoint with validation"

# Complex debugging
harvester agent-claude "Debug the memory leak in the worker pool"
```

**Features:**
- 🏗️ **Production-tested agent loop** - Anthropic's own implementation
- 🎯 **Automatic context management** - Built-in compaction and caching
- 🔧 **Professional tooling** - Same tools as Claude Code CLI
- 🤝 **MCP protocol support** - External service integrations
- ⚙️ **Subagents** - Parallel task execution (be careful, faceless Claude agents do not respect the project)

**Example Output:**
```bash
🤖 agent-claude
📋 Task: Implement REST API...
🎯 Type: feature
🧠 Model: claude-sonnet-4-5

🔧 Using tool: Write
✓ Created api/endpoints.py

🔧 Using tool: Bash
✓ Tests passed

📊 Status: completed
💰 Cost: $0.097
```

### OpenAI Code Agent (`agent-openai`)

**Built on OpenAI's Agents SDK** - File operations and code execution with GPT-4o, o1, and o3-mini.

```bash
# Create a Python script
harvester agent-openai "Create a hello world Python script"

# Advanced reasoning with o1
harvester agent-openai "Refactor auth.py for better error handling" -m o1

# Complex task
harvester agent-openai "Add logging to all functions in utils.py"
```

**Features:**
- 📁 **File operations** - Read, write, edit files with intelligent parsing
- 🐚 **Shell execution** - Run commands and capture output
- 🧠 **Multi-step reasoning** - o1/o3-mini for complex tasks
- ⚡ **Fast iteration** - GPT-4o for quick tasks

### GPT-5 Code Agent (`agent-gpt5`)

**Powered by GPT-5** - Advanced reasoning for coding and agentic tasks with configurable effort and verbosity.

```bash
# Quick task with minimal reasoning
harvester agent-gpt5 "Create a hello world script" -r minimal -v low

# Standard coding task
harvester agent-gpt5 "Add error handling to auth.py"

# Complex refactoring with high reasoning
harvester agent-gpt5 "Refactor entire codebase for async/await" -r high -v high

# Cost-optimized with gpt-5-mini
harvester agent-gpt5 "Fix bugs in utils.py" -m gpt-5-mini

# High-throughput classification with nano
harvester agent-gpt5 "Classify all files by type" -m gpt-5-nano -r minimal
```

**Features:**
- 🧠 **Configurable reasoning** - minimal/low/medium/high effort levels
- 📝 **Verbosity control** - low/medium/high output length
- 🔧 **Custom tools** - Freeform text inputs with optional CFG constraints
- 💡 **Preambles** - Transparent tool-calling explanations
- 📁 **File operations** - Read, write, edit with intelligent parsing
- 🐚 **Shell execution** - Run commands with output capture

**Reasoning Effort Guide:**
- `minimal` - Fastest time-to-first-token, best for simple tasks
- `low` - Quick reasoning, good for straightforward coding
- `medium` - Balanced reasoning (default), good for most tasks
- `high` - Thorough reasoning, best for complex multi-step tasks

**Verbosity Guide:**
- `low` - Concise responses, minimal commentary
- `medium` - Balanced explanations (default)
- `high` - Detailed explanations and documentation

**Model Variants:**
- `gpt-5` - Complex reasoning, broad world knowledge, code-heavy tasks
- `gpt-5-mini` - Cost-optimized reasoning, balances speed/cost/capability
- `gpt-5-nano` - High-throughput, simple instruction-following

### Code Interpreter (`code-interpreter`)

**Python code execution in sandboxed containers** - Write and run Python to solve complex problems.

```bash
# Solve a math problem
harvester code-interpreter "Solve the equation 3x + 11 = 14"

# Data analysis with file upload
harvester code-interpreter "Analyze data.csv and create a histogram" -u data.csv

# Image processing with file download
harvester code-interpreter "Resize image.png to 800x600" -u image.png -d

# Generate visualization
harvester code-interpreter "Create a sine wave plot from 0 to 2π" -d

# Use GPT-5 for complex problems
harvester code-interpreter "Calculate fibonacci(100) using memoization" -m gpt-5
```

**Features:**
- 🐍 **Sandboxed Python execution** - Safe, isolated environment
- 📁 **File upload/download** - Work with data files and images
- 📊 **Data analysis** - Process CSV, JSON, Excel files
- 📈 **Visualization** - Create plots, charts, graphs
- 🔄 **Iterative solving** - Model debugs and retries code automatically
- 🖼️ **Image processing** - Crop, resize, transform images (with reasoning models)

**Use Cases:**
- Mathematical computations and equation solving
- Data analysis and statistical processing
- File format conversions
- Image manipulation and computer vision
- Chart and graph generation
- Algorithm prototyping

**Container Management:**
- Containers auto-expire after 20 minutes of inactivity
- Use `--container-id` to reuse an existing container
- Files are automatically uploaded to the container
- Use `-d` flag to download all generated files

### Image Generation (`image-gen`)

**AI-powered image creation and editing** - Generate and edit images using GPT Image model.

```bash
# Generate an image
harvester image-gen "A gray tabby cat hugging an otter with an orange scarf"

# High quality with custom size
harvester image-gen "Sunset over mountains" -o sunset.png -q high -s 1024x1536

# Edit an existing image
harvester image-gen "Make it more colorful" -i input.jpg -o edited.png

# Multi-turn editing
harvester image-gen "Draw a futuristic cityscape" -o city.png
harvester image-gen "Make it realistic" --edit -o city_realistic.png

# Use GPT-5 for best results
harvester image-gen "Abstract art with vibrant colors" -m gpt-5 -q high
```

**Features:**
- 🎨 **Text-to-image generation** - Create images from descriptions
- ✏️ **Image editing** - Modify existing images with text prompts
- 🔄 **Multi-turn editing** - Iteratively refine images
- ✨ **Automatic prompt optimization** - AI improves your prompts
- 📐 **Configurable output** - Size, quality, format, background
- 🖼️ **Input image support** - Edit or enhance existing images

**Configuration Options:**
- **Size**: 1024x1024, 1024x1536, 1536x1024, or auto
- **Quality**: low, medium, high, or auto (model chooses best)
- **Format**: png, jpeg, webp
- **Background**: transparent, opaque, or auto

**Use Cases:**
- Art and illustration creation
- Photo editing and enhancement
- Concept visualization
- Marketing material generation
- Storyboarding and mockups
- Style transfer and artistic effects

**Prompting Tips:**
- Use terms like "draw" or "edit" in your prompts
- For combining images: "edit the first image by adding..." not "merge"
- Be specific about style, colors, and composition
- The model will automatically optimize your prompt for better results

### Agent Comparison

| Feature | Grok | Claude | OpenAI | GPT-5 | Code Interpreter |
|---------|------|--------|--------|-------|------------------|
| **Speed** | ⚡⚡⚡⚡ Very Fast | ⚡⚡ Thorough | ⚡⚡⚡ Fast | ⚡⚡⚡⚡ Configurable | ⚡⚡⚡ Fast |
| **Cost** | 💰 ~$0.002 | 💰💰 ~$0.10 | 💰💰 ~$0.05 | 💰-💰💰💰 Configurable | 💰💰 ~$0.05 |
| **Reasoning** | Fast | Deep | Multi-step | Configurable | Iterative |
| **Best For** | Quick tasks | Claude fans | o1 reasoning | Complex coding | Data/Math |
| **Capabilities** | File ops | File ops | File ops | File ops | Python execution |
| **Environment** | Local files | Local files | Local files | Local files | Sandboxed container |

**When to use which:**
- **Grok Agent**: General use, fast iteration, prototyping, cost-effective
- **Claude Agent**: If you like Claude (prone to hallucinations, use with caution)
- **OpenAI Agent**: Advanced reasoning with o1/o3-mini, multi-step tasks
- **GPT-5 Agent**: Complex coding tasks requiring configurable reasoning depth
- **Code Interpreter**: Data analysis, math problems, visualizations, Python execution

### Examples

See practical examples in:
- `/example/agent-grok/` - Output from Grok Code Agent
- `/example/agent-claude/` - Output from Claude Code Agent
- `/example/batch-results...` - Batch processing files
## 💼 Programming Interface

### Python SDK Usage

```python
from harvester_sdk import HarvesterSDK

# Initialize SDK
sdk = HarvesterSDK()

# Quick processing
result = await sdk.quick_process(
    prompt="Explain quantum computing",
    model="gemini-2.5-pro"
)

# Batch processing
results = await sdk.process_batch(
    requests=["What is AI?", "Explain ML", "Define neural networks"],
    model="claude-sonnet-4-20250514"
)

# Multi-provider council (get consensus)
consensus = await sdk.quick_council(
    prompt="What is consciousness?",
    models=["gemini-2.5-pro", "claude-sonnet-4-20250514", "gpt-4o"]
)
```

### Provider Factory

```python
from providers.provider_factory import ProviderFactory

# Create provider factory
factory = ProviderFactory()

# Get provider for specific model
provider = factory.get_provider("gemini-2.5-flash")  # -> GenAI provider
provider = factory.get_provider("vtx-gemini-2.5-flash")  # -> Vertex AI provider

# Generate completion
response = await provider.complete("Hello, world!", "gemini-2.5-flash")
```

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        HARVESTER SDK                              │
├───────────────────────────────────────────────────────────────────┤
│                    Main CLI Conductor                             │
│                   (harvester command)                             │
├────────┬────────┬────────┬────────┬────────┬────────┬────────────┤
│Message │ Batch  │Process │ Image  │ Search │ Grok   │  Claude    │
│(Chat)  │  CSV   │  Dir   │  Gen   │Enhanced│ Agent  │  Agent     │
├────────┴────────┴────────┴────────┴────────┴────────┴────────────┤
│                     Provider Factory                              │
├────────┬────────┬────────┬────────┬────────┬──────────────────────┤
│ GenAI  │Vertex  │ OpenAI │Anthropic│  XAI   │     DeepSeek         │
│(APIKey)│(SA)    │        │         │ (Grok) │                      │
└────────┴────────┴────────┴────────┴────────┴──────────────────────┘
         └──────────────────────────────────────┘
                    Agentic Tools Layer
         ┌─────────────────┬────────────────────┐
         │  Grok Agent     │  Claude Agent      │
         │  (Custom Loop)  │  (Official SDK)    │
         │  - 9 tools      │  - Full SDK tools  │
         │  - Challenger   │  - previous champ  │
         └─────────────────┴────────────────────┘
```

## 🔒 Authentication Methods

### Clear Separation for Google Services

**Google AI Studio (GenAI)**:
- ✅ Simple API key: `GEMINI_API_KEY`
- ✅ Models: `gemini-2.5-flash`, `gemini-2.5-pro`
- ✅ Best for: Personal use, quick setup

**Google Vertex AI**:
- ✅ Service account: `GOOGLE_APPLICATION_CREDENTIALS`
- ✅ Models: `vtx-gemini-2.5-flash`, `vtx-gemini-2.5-pro`
- ✅ Best for: Enterprise, GCP integration

## 🌟 Open Source & Free

All features are **completely free and open source** under the MIT License. No tiers, no paywalls, no restrictions.

- ✅ **Unlimited workers** - Scale as much as you need
- ✅ **All providers** - Full access to every AI provider
- ✅ **Advanced features** - Structured output, function calling, multi-provider parallelism
- ✅ **Enterprise ready** - Production-grade reliability built-in

## 📖 Examples

### Turn-Based Conversation

```bash
# Start a conversation with Gemini
harvester message --model gemini-2.5-flash

# Chat with Claude
harvester message --model claude-sonnet-4

# System prompt example
harvester message --model grok-4-0709 --system "You are an expert programmer"
```

### Batch Processing

```bash
# Process CSV with AI
harvester batch questions.csv --model gemini-2.5-pro --template analysis

# Directory transformation
harvester process ./legacy_code --template modernize.j2 --model claude-sonnet-4-5
```

### Image Generation

```bash
# DALL-E 3
harvester image "A futuristic city" --provider dalle-3 --quality hd

# Imagen 4
harvester image "Abstract art" --provider vertex_image --model imagen-4
```

## 🤝 Support & Contributing

- **Documentation**: Full guides in `/docs`
- **Issues**: Report bugs via GitHub issues
- **Enterprise**: Contact info@quantumencoding.io
- **License**: MIT - see LICENSE file

## 🌟 Why Harvester SDK?

1. **Unified Interface** - One API for all providers
2. **Authentication Clarity** - Clear separation of auth methods
3. **Production Ready** - Error handling, retries, rate limiting
4. **Flexible Deployment** - CLI tools + Python SDK
5. **Cost Optimization** - Batch processing with 50% savings
6. **Multi-Modal** - Text, images, and more
7. **Enterprise Grade** - Open source, well-documented, production-ready

---

**© 2025 QUANTUM ENCODING LTD**  
📧 Contact: info@quantumencoding.io  
🌐 Website: https://quantumencoding.io  

*The complete AI processing platform for modern applications.*
