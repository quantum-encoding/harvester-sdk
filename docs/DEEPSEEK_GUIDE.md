# DeepSeek Integration Guide

## Overview

The Harvester SDK now supports **DeepSeek V3.2-Exp** with 50%+ cost reduction and dual API format support!

## What's New in V3.2-Exp

✨ **DeepSeek Sparse Attention (DSA)** - Faster, more efficient training & inference on long context
💰 **50%+ Price Reduction** - API costs cut by more than half
⚡ **Performance** - Benchmarks show V3.2-Exp performs on par with V3.1-Terminus

## Available Models

| Model | Alias | Description | Max Output | Cost (Input/Output per 1M tokens) |
|-------|-------|-------------|------------|----------------------------------|
| `deepseek-chat` | `ds-1` | V3.2-Exp with DSA | 8K | $0.14 / $0.55 |
| `deepseek-reasoner` | `ds-2` | DeepSeek R1 Reasoner | 64K | $0.28 / $1.10 |
| `deepseek-v3.1-terminus` | `ds-1-legacy` | Legacy comparison model (until Oct 15, 2025) | 8K | $0.14 / $0.55 |

## API Formats Supported

### 1. OpenAI Format (Default)
```python
from providers.deepseek_provider import DeepseekProvider

provider = DeepseekProvider({
    'api_key': 'your-api-key',
    'api_format': 'openai'  # Default
})

result = await provider.complete("Hello", "deepseek-chat")
```

### 2. Anthropic Format
```python
provider = DeepseekProvider({
    'api_key': 'your-api-key',
    'api_format': 'anthropic'
})

result = await provider.complete("Hello", "deepseek-chat")
```

## CLI Usage

### Basic Chat
```bash
# V3.2-Exp (default)
harvester message --model deepseek-chat
harvester message --model ds-1

# Reasoner model
harvester message --model deepseek-reasoner
harvester message --model ds-2
```

### Comparison Testing (V3.1-Terminus)
```bash
# Use legacy model for comparison until Oct 15, 2025
harvester message --model deepseek-v3.1-terminus
harvester message --model ds-1-legacy
```

## Environment Variables

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

## Comparison Testing

To compare V3.2-Exp vs V3.1-Terminus:

```python
import asyncio
from providers.deepseek_provider import DeepseekProvider

async def compare_models():
    provider = DeepseekProvider({'api_key': 'your-key'})
    
    prompt = "Explain quantum computing in 2 sentences"
    
    # V3.2-Exp (default)
    v32_result = await provider.complete(prompt, "deepseek-chat")
    print(f"V3.2-Exp: {v32_result}")
    
    # V3.1-Terminus (for comparison)
    v31_result = await provider.complete(prompt, "deepseek-v3.1-terminus")
    print(f"V3.1-Terminus: {v31_result}")

asyncio.run(compare_models())
```

**Provide feedback**: https://feedback.deepseek.com/dsa

## Cost Comparison

| Operation | V3.1-Terminus | V3.2-Exp | Savings |
|-----------|---------------|----------|---------|
| 1M input tokens (cache miss) | $0.27 | $0.14 | 48% |
| 1M output tokens | $1.10 | $0.55 | 50% |
| **Total for typical request** | **~50% cheaper** | | |

## Features by API Format

### OpenAI Format
✅ Standard chat completions
✅ Streaming support
✅ Function calling
✅ System messages
✅ Tool use

### Anthropic Format
✅ Messages API compatible
✅ System prompts
✅ Tool use (basic support)
⚠️ Some fields ignored (see compatibility table below)

## Anthropic API Compatibility

| Feature | Support Status |
|---------|---------------|
| Basic messages | ✅ Fully Supported |
| System prompts | ✅ Fully Supported |
| Tool use | ✅ Fully Supported |
| Temperature (0.0-2.0) | ✅ Fully Supported |
| max_tokens | ✅ Fully Supported |
| stop_sequences | ✅ Fully Supported |
| Image input | ❌ Not Supported |
| Document input | ❌ Not Supported |
| Cache control | ⚠️ Ignored |

## Advanced Usage

### Using Anthropic Format in Code

```python
# Use DeepSeek with Anthropic SDK format
from providers.deepseek_provider import DeepseekProvider

provider = DeepseekProvider({
    'api_format': 'anthropic',
    'api_key': 'your-key'
})

# The provider automatically formats requests in Anthropic style
response = await provider.complete(
    "Write a Python function to calculate fibonacci",
    "deepseek-chat"
)
```

### Environment Setup for Claude Code Compatibility

```bash
# Use DeepSeek with Claude Code
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_AUTH_TOKEN=your_deepseek_api_key
export ANTHROPIC_MODEL=deepseek-chat
export ANTHROPIC_SMALL_FAST_MODEL=deepseek-chat
export API_TIMEOUT_MS=600000
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
```

## Migration from V3.1-Terminus

Existing code will automatically use V3.2-Exp:

```python
# This now uses V3.2-Exp by default
provider = DeepseekProvider({'api_key': 'key'})
result = await provider.complete("Hello", "deepseek-chat")
```

To explicitly use V3.1-Terminus for comparison:

```python
result = await provider.complete("Hello", "deepseek-v3.1-terminus")
```

**Note**: V3.1-Terminus comparison endpoint expires **October 15, 2025, 15:59 UTC**

---

© 2025 Quantum Encoding Ltd
