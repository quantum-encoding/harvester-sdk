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

# Harvester SDK is now FREEMIUM - Full power, no restrictions!
echo ""
echo "🌟 FREEMIUM BUILD - MAX POWER! 🌟"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ All providers, all features, unlimited power"
echo "✨ Agentic coding assistants (Grok + Claude)"
echo "✨ No tiers, no paywalls, no restrictions"
echo ""
echo "📥 Installing FULL dependencies (all providers + agents)..."
pip install -r requirements.txt

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
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ FREEMIUM INSTALLATION COMPLETE - MAX POWER UNLOCKED! ✅"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 What you get (100% FREE):"
echo "   ✨ ALL AI providers (OpenAI, Anthropic, Google, XAI, DeepSeek)"
echo "   ✨ BOTH agentic coding assistants (Grok + Claude)"
echo "   ✨ UNLIMITED workers and batch processing"
echo "   ✨ ALL advanced features (structured output, function calling)"
echo "   ✨ Production-grade reliability built-in"
echo ""
echo "🔑 Set up your API keys:"
echo "   export GEMINI_API_KEY=your_key      # Google AI Studio"
echo "   export OPENAI_API_KEY=your_key      # OpenAI"
echo "   export ANTHROPIC_API_KEY=your_key   # Anthropic (for Claude Agent)"
echo "   export XAI_API_KEY=your_key         # XAI (for Grok Agent)"
echo "   export DEEPSEEK_API_KEY=your_key    # DeepSeek"
echo ""
echo "🎯 For Vertex AI (enterprise):"
echo "   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json"
echo ""
echo "🚀 Quick Start:"
echo "   source venv/bin/activate"
echo "   python harvester.py --help"
echo ""
echo "💬 Chat with AI:"
echo "   harvester chat --model gemini-2.5-flash"
echo "   harvester message --model claude-sonnet-4-20250514"
echo ""
echo "🤖 Agentic Coding (NEW!):"
echo "   harvester agent-grok \"Create a Python script for data validation\""
echo "   harvester agent-claude \"Implement a REST API endpoint\""
echo ""
echo "📦 Batch Processing:"
echo "   harvester batch data.csv --model gpt-4o --template analysis"
echo ""
echo "🖼️  Image Generation:"
echo "   harvester image \"A beautiful sunset\" --provider dalle3"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📧 Support: info@quantumencoding.io"
echo "🌐 Website: https://quantumencoding.io"
echo "🎉 Enjoy unlimited AI processing power - it's FREE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"