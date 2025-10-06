# Harvester SDK Installation Guide

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/quantumencoding/harvester-sdk.git
cd harvester-sdk

# 2. Run the setup script
./setup.sh

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Test the installation
python -c "from harvester_sdk import HarvesterSDK; print('SDK imported successfully')"
```

## Manual Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .  # Install package in development mode
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
# OpenAI (GPT-5, DALL-E 3)
OPENAI_API_KEY=sk-...

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# Google Cloud (Gemini, Imagen)
GOOGLE_CLOUD_PROJECT=your-project-id

# DeepSeek
DEEPSEEK_API_KEY=sk-...

# xAI (Grok)
XAI_API_KEY=xai-...
```

Or export them in your shell:

```bash
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'
export GOOGLE_CLOUD_PROJECT='your-project-id'
export DEEPSEEK_API_KEY='your-key-here'
export XAI_API_KEY='your-key-here'
```

### 4. Google Cloud Authentication (for Gemini/Imagen)

```bash
# Install gcloud CLI if needed
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Available Models

### Text Generation
- **GPT-5 Series**: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- **Gemini 2.5**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- **Claude**: `claude-opus-4`, `claude-sonnet-4`, `claude-3-5-haiku`
- **Grok**: `grok-4`, `grok-3`, `grok-fast`
- **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`

### Image Generation
- **OpenAI**: `dalle-3`, `gpt-image-1`
- **Google**: `imagen-4`, `imagen-4-ultra`, `imagen-4-fast`

### Multimodal/Vision
- **Grok Vision**: `grok-image` (text model with image understanding)

## Testing Your Installation

### Test Text Generation

```python
from harvester_sdk import HarvesterSDK
import asyncio

sdk = HarvesterSDK()

# Test GPT-5
async def test():
    result = await sdk.async_generate_text(
        prompt="What is the meaning of life?",
        model="gpt-5-nano"
    )
    print(result)

asyncio.run(test())
```

### Test Image Generation

```bash
# Using the image CLI
python cli/image/image_cli.py -p "modern minimalist living room" -m dalle-3

# Or using Gemini Flash Image (fastest)
python cli/image/image_cli.py -p "sunset over mountains" -m goo-2-img
```

### Test CLI Commands

```bash
# Test main harvester CLI
python harvester.py --help

# Test batch submission
python cli/batch/batch_submit.py --help

# Test template processing
python parallel_template_cli.py --help
```

## Common Issues

### Missing API Key
```
ValueError: OpenAI API key required
```
**Solution**: Set the appropriate environment variable or add to `.env` file

### Google Cloud Auth Error
```
Failed to authenticate with Google Cloud
```
**Solution**: Run `gcloud auth login` and ensure project is set

### Module Import Error
```
ModuleNotFoundError: No module named 'harvester_sdk'
```
**Solution**: Install package with `pip install -e .` in the project root

## Directory Structure

After installation:
```
harvester-sdk/
├── cli/              # CLI entry points
├── config/           # Configuration files
├── core/             # Core SDK modules
├── docs/             # User documentation
├── providers/        # Provider implementations
├── templates/        # Jinja2 templates
├── harvester.py      # Main CLI
└── README.md         # Documentation
```

Outputs will be saved to sovereign directories based on CLI used.

## License

This project is licensed under the MIT License - all features are open source and free to use.

## Support

For issues or questions:
- Check the [User Documentation](docs/)
- Review the [providers configuration](config/providers.yaml)
- See the [CLI Reference](docs/CLI_REFERENCE.md)
- Read the [DeepSeek Guide](docs/DEEPSEEK_GUIDE.md) for cost-effective processing

---

**© 2025 Quantum Encoding Ltd**
Open Source - MIT License