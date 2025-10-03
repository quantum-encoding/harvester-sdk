"""
Harvester SDK - Version Information
© 2025 QUANTUM ENCODING LTD
"""

__version__ = "2.0.0"
__author__ = "Quantum Encoding Ltd"
__email__ = "info@quantumencoding.io"
__website__ = "https://quantumencoding.io"
__description__ = "Complete AI Processing Platform - Unified interface for all AI providers"
__license__ = "MIT"

# Version components
VERSION_MAJOR = 2
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_BUILD = "stable"

# Feature flags
FEATURES = {
    "multi_provider_support": True,
    "genai_provider": True,
    "vertex_ai_provider": True,
    "turn_based_chat": True,
    "batch_processing": True,
    "image_generation": True,
    "template_system": True,
    "open_source": True,
    "all_features_unlocked": True,
}

# Provider support matrix
SUPPORTED_PROVIDERS = {
    "genai": {
        "name": "Google AI Studio",
        "auth": "api_key",
        "models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-lite"
        ]
    },
    "google": {
        "name": "Google Vertex AI",
        "auth": "service_account",
        "models": [
            "vtx-gemini-2.5-pro",
            "vtx-gemini-2.5-flash",
            "vtx-gemini-2.5-flash-lite",
            "imagen-4",
            "imagen-4-ultra",
            "imagen-4-fast"
        ]
    },
    "openai": {
        "name": "OpenAI",
        "auth": "api_key",
        "models": [
            "gpt-5",
            "gpt-5-nano",
            "gpt-5-mini",
            "dall-e-3",
            "gpt-image-1"
        ]
    },
    "anthropic": {
        "name": "Anthropic",
        "auth": "api_key",
        "models": [
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022"
        ]
    },
    "xai": {
        "name": "xAI",
        "auth": "api_key",
        "models": [
            "grok-4-0709",
            "grok-3",
            "grok-3-mini",
            "grok-2-image-1212"
        ]
    },
    "deepseek": {
        "name": "DeepSeek",
        "auth": "api_key",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-v3.1-terminus"
        ],
        "api_formats": ["openai", "anthropic"]
    }
}

def get_version_info():
    """Get complete version information"""
    return {
        "version": __version__,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "build": VERSION_BUILD,
        "author": __author__,
        "email": __email__,
        "website": __website__,
        "description": __description__,
        "license": __license__,
        "providers": len(SUPPORTED_PROVIDERS),
        "features": list(FEATURES.keys())
    }

def print_banner():
    """Print the SDK banner"""
    print(f"""
🚀 Harvester SDK v{__version__}
© 2025 {__author__}
📧 {__email__}
🌐 {__website__}

{__description__}
""")

if __name__ == "__main__":
    print_banner()
    info = get_version_info()
    print("Version Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")