#!/bin/bash
# Harvester SDK - Development Setup Script
# © 2025 QUANTUM ENCODING LTD

set -e  # Exit on any error

echo "🚀 Harvester SDK - Development Setup"
echo "© 2025 QUANTUM ENCODING LTD"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📍 Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install based on user choice
echo ""
echo "Choose installation type:"
echo "1) Core only (minimal dependencies)"
echo "2) Full (all AI providers)"
echo "3) Development (full + dev tools)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "📥 Installing core dependencies..."
        pip install -r requirements.txt
        ;;
    2)
        echo "📥 Installing full dependencies (all providers)..."
        pip install -r requirements-full.txt
        ;;
    3)
        echo "📥 Installing development dependencies..."
        pip install -r requirements-full.txt
        pip install -e .
        ;;
    *)
        echo "❌ Invalid choice. Installing core dependencies..."
        pip install -r requirements.txt
        ;;
esac

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p outputs/{conversations,batches,images}

# Check for API keys
echo ""
echo "🔑 API Key Status:"
echo "------------------"

check_api_key() {
    if [ -z "${!1}" ]; then
        echo "❌ $1 not set"
        return 1
    else
        echo "✅ $1 is configured"
        return 0
    fi
}

# Check each provider's API key
check_api_key "GEMINI_API_KEY"
check_api_key "OPENAI_API_KEY" 
check_api_key "ANTHROPIC_API_KEY"
check_api_key "XAI_API_KEY"
check_api_key "DEEPSEEK_API_KEY"
check_api_key "GOOGLE_APPLICATION_CREDENTIALS"

# Show installation summary
echo ""
echo "✅ Installation complete!"
echo ""
echo "🔑 Set up your API keys:"
echo "   export GEMINI_API_KEY=your_key      # Google AI Studio"
echo "   export OPENAI_API_KEY=your_key      # OpenAI" 
echo "   export ANTHROPIC_API_KEY=your_key   # Anthropic"
echo "   export XAI_API_KEY=your_key         # XAI"
echo "   export DEEPSEEK_API_KEY=your_key    # DeepSeek"
echo ""
echo "🎯 For Vertex AI (enterprise):"
echo "   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json"
echo ""
echo "2. Test the installation:"
echo "   source venv/bin/activate"
echo "   python harvester.py --help"
echo "   python harvester.py tier"
echo ""
echo "3. Start using Harvester SDK:"
echo "   harvester message --model gemini-2.5-flash"
echo "   harvester message --model vtx-gemini-2.5-flash  # Vertex AI"
echo ""
echo "📧 Support: info@quantumencoding.io"
echo "🌐 Website: https://quantumencoding.io"