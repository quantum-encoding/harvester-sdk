# 🚀 Harvester SDK - Complete AI Processing Platform

> **"The unified interface for all AI providers with enterprise-grade reliability."**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Providers: 7+](https://img.shields.io/badge/Providers-OpenAI%20%7C%20Anthropic%20%7C%20GenAI%20%7C%20Vertex%20%7C%20XAI%20%7C%20DeepSeek-brightgreen.svg)](#providers)

## 🌟 What is Harvester SDK?

Harvester SDK is a comprehensive AI processing platform that provides a **unified interface** to all major AI providers. Whether you need text generation, image creation, batch processing, or real-time conversations, Harvester SDK handles the complexity so you can focus on building.

### ⚡ Key Features

- **Multi-Provider Support** - OpenAI, Anthropic, Google AI Studio, Vertex AI, XAI, DeepSeek
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
# Install the SDK
pip install harvester-sdk

# Install with all providers
pip install harvester-sdk[all]

# Install specific providers
pip install harvester-sdk[openai,anthropic,genai]
```

### Basic Usage

```bash
# Main CLI conductor
harvester --help

# Turn-based conversation (non-streaming)
harvester message --model gemini-2.5-flash
harvester message --model claude-sonnet-4-20250514 --system "You are a helpful assistant"

# Batch processing from CSV
harvester batch data.csv --model gpt-4o --template quick

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
| `gemini-1.5-flash` | `vtx-gemini-1.5-flash` | Legacy support |

### Other Providers
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Anthropic**: `claude-sonnet-4-20250514`, `claude-opus-4-1-20250805`
- **XAI**: `grok-4-0709`, `grok-3`, `grok-3-mini`
- **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`

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
┌─────────────────────────────────────────────────────────┐
│                    HARVESTER SDK                        │
├─────────────────────────────────────────────────────────┤
│                 Main CLI Conductor                      │
│              (harvester command)                        │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│ Message  │  Batch   │ Process  │  Image   │   Search   │
│(Non-str) │   CSV    │   Dir    │   Gen    │ Enhanced   │
├──────────┴──────────┴──────────┴──────────┴────────────┤
│                  Provider Factory                       │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│  GenAI   │ Vertex   │  OpenAI  │Anthropic │    XAI     │
│(API Key) │(Service) │          │          │ DeepSeek   │
└──────────┴──────────┴──────────┴──────────┴────────────┘
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
harvester message --model gemini-2.5-flash --save

# Chat with Claude
harvester message --model claude-sonnet-4-20250514 --temperature 0.3

# System prompt example
harvester message --model grok-4-0709 --system "You are an expert programmer"
```

### Batch Processing

```bash
# Process CSV with AI
harvester batch questions.csv --model gemini-2.5-pro --template analysis

# Directory transformation
harvester process ./legacy_code --template modernize --model claude-sonnet-4-20250514
```

### Image Generation

```bash
# DALL-E 3
harvester image "A futuristic city" --provider dalle3 --quality hd

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