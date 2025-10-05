"""
LiteLLM Integration - Use 100+ models with unified interface

Provides easy access to models from OpenAI, Anthropic, Google, Cohere,
Mistral, and 100+ other providers through LiteLLM.
"""

from typing import Optional, Dict, Any
import os


def create_litellm_agent(
    name: str,
    model: str,
    api_key: Optional[str] = None,
    instructions: str = "You are a helpful assistant.",
    tools: Optional[list] = None,
    temperature: Optional[float] = None,
    track_usage: bool = True,
    **kwargs
):
    """
    Create an agent using any LiteLLM-supported model.

    Supports 100+ models from different providers:
    - OpenAI: openai/gpt-4.1, openai/gpt-5
    - Anthropic: anthropic/claude-3-5-sonnet-20240620
    - Google: gemini/gemini-2.5-flash
    - Cohere: cohere/command-r-plus
    - And many more!

    Args:
        name: Agent name
        model: Model name (with provider prefix, e.g., "anthropic/claude-3-5-sonnet")
        api_key: API key for the model provider
        instructions: Agent instructions
        tools: List of tools
        temperature: Sampling temperature
        track_usage: Whether to track token usage
        **kwargs: Additional arguments for Agent

    Returns:
        OpenAIAgent instance configured with LiteLLM model

    Example:
        # Anthropic Claude
        agent = create_litellm_agent(
            name="Claude Assistant",
            model="anthropic/claude-3-5-sonnet-20240620",
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

        # Google Gemini
        agent = create_litellm_agent(
            name="Gemini Assistant",
            model="gemini/gemini-2.5-flash",
            api_key=os.environ["GEMINI_API_KEY"]
        )

    Note:
        Install LiteLLM support: pip install harvester-sdk[computer]
        (The computer extra includes openai-agents which includes litellm)
    """
    try:
        from agents.extensions.models.litellm_model import LitellmModel
        from agents import Agent, ModelSettings
    except ImportError:
        raise ImportError(
            "LiteLLM integration not available. Install with: "
            "pip install harvester-sdk[computer]"
        )

    from ..openai_agent import OpenAIAgent

    # Build model settings
    model_settings_kwargs = {}
    if temperature is not None:
        model_settings_kwargs["temperature"] = temperature
    if track_usage:
        model_settings_kwargs["include_usage"] = True

    model_settings = (
        ModelSettings(**model_settings_kwargs) if model_settings_kwargs else None
    )

    # Create LiteLLM model
    litellm_model = LitellmModel(model=model, api_key=api_key)

    # Create raw Agent
    raw_agent = Agent(
        name=name,
        instructions=instructions,
        model=litellm_model,
        tools=tools or [],
        model_settings=model_settings,
        **kwargs
    )

    # Wrap in OpenAIAgent
    agent_wrapper = OpenAIAgent.__new__(OpenAIAgent)
    agent_wrapper.agent = raw_agent
    return agent_wrapper


# Common model configurations
LITELLM_MODELS = {
    # OpenAI
    "gpt-5": "openai/gpt-5",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3-5-sonnet-20240620",
    "claude-3.5-haiku": "anthropic/claude-3-5-haiku-20241022",
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
    # Google
    "gemini-2.5-flash": "gemini/gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro-preview-04-17",
    "gemini-1.5-flash": "gemini/gemini-1.5-flash",
    "gemini-1.5-pro": "gemini/gemini-1.5-pro",
    # Cohere
    "command-r-plus": "cohere/command-r-plus",
    "command-r": "cohere/command-r",
    # Mistral
    "mistral-large": "mistral/mistral-large-latest",
    "mistral-medium": "mistral/mistral-medium-latest",
    "mistral-small": "mistral/mistral-small-latest",
    # Groq
    "llama-3.3-70b": "groq/llama-3.3-70b-versatile",
    "llama-3.1-8b": "groq/llama-3.1-8b-instant",
    # DeepSeek
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-coder": "deepseek/deepseek-coder",
}


def get_litellm_model_id(alias: str) -> str:
    """
    Get full LiteLLM model ID from a friendly alias.

    Args:
        alias: Friendly model name (e.g., "claude-3.5-sonnet")

    Returns:
        Full LiteLLM model ID (e.g., "anthropic/claude-3-5-sonnet-20240620")

    Example:
        model_id = get_litellm_model_id("claude-3.5-sonnet")
        # Returns: "anthropic/claude-3-5-sonnet-20240620"
    """
    return LITELLM_MODELS.get(alias, alias)


class LiteLLMConfig:
    """
    Configuration helper for LiteLLM agents.

    Provides common settings for different model providers.
    """

    @staticmethod
    def anthropic(
        model: str = "claude-3.5-sonnet",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Get configuration for Anthropic Claude models.

        Args:
            model: Model alias or full ID
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict of configuration parameters

        Example:
            config = LiteLLMConfig.anthropic("claude-3.5-sonnet")
            agent = create_litellm_agent("Claude", **config)
        """
        return {
            "model": get_litellm_model_id(model),
            "api_key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
            "temperature": 1.0,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def google(
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get configuration for Google Gemini models.

        Args:
            model: Model alias or full ID
            api_key: Google API key (default: from GEMINI_API_KEY env)

        Returns:
            Dict of configuration parameters
        """
        return {
            "model": get_litellm_model_id(model),
            "api_key": api_key or os.environ.get("GEMINI_API_KEY"),
            "temperature": 0.7,
        }

    @staticmethod
    def cohere(
        model: str = "command-r-plus",
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get configuration for Cohere models.

        Args:
            model: Model alias or full ID
            api_key: Cohere API key (default: from COHERE_API_KEY env)

        Returns:
            Dict of configuration parameters
        """
        return {
            "model": get_litellm_model_id(model),
            "api_key": api_key or os.environ.get("COHERE_API_KEY"),
            "temperature": 0.7,
        }

    @staticmethod
    def groq(
        model: str = "llama-3.3-70b",
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get configuration for Groq models.

        Args:
            model: Model alias or full ID
            api_key: Groq API key (default: from GROQ_API_KEY env)

        Returns:
            Dict of configuration parameters
        """
        return {
            "model": get_litellm_model_id(model),
            "api_key": api_key or os.environ.get("GROQ_API_KEY"),
            "temperature": 0.7,
        }


def disable_tracing_for_non_openai():
    """
    Disable tracing when using non-OpenAI models.

    Call this to avoid 401 errors when using models that
    don't have OpenAI API keys for trace upload.

    Example:
        disable_tracing_for_non_openai()
        agent = create_litellm_agent("Claude", model="claude-3.5-sonnet")
    """
    try:
        from agents import set_tracing_disabled
        set_tracing_disabled(True)
    except ImportError:
        pass


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        print("LiteLLM Integration Examples\n")

        # Example 1: Anthropic Claude
        print("Example 1: Anthropic Claude agent")
        config = LiteLLMConfig.anthropic("claude-3.5-sonnet")
        print(f"Config: {config['model']}")

        # Example 2: Google Gemini
        print("\nExample 2: Google Gemini agent")
        config = LiteLLMConfig.google("gemini-2.5-flash")
        print(f"Config: {config['model']}")

        # Example 3: Groq Llama
        print("\nExample 3: Groq Llama agent")
        config = LiteLLMConfig.groq("llama-3.3-70b")
        print(f"Config: {config['model']}")

        # Example 4: Model aliases
        print("\nExample 4: Model aliases")
        for alias in ["claude-3.5-sonnet", "gemini-2.5-flash", "gpt-5"]:
            full_id = get_litellm_model_id(alias)
            print(f"{alias} -> {full_id}")

        print("\n✓ LiteLLM examples created successfully!")
        print("\nSupported models:")
        for alias in list(LITELLM_MODELS.keys())[:5]:
            print(f"  - {alias}")
        print(f"  ... and {len(LITELLM_MODELS) - 5} more!")

    asyncio.run(main())
