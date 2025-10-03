"""
xAI Provider for Grok Models - Advanced Edition
Supports: grok-4, grok-3, grok-3-mini with Live Search and Reasoning
"""
import logging
from typing import Any, Dict, List, Union, Optional
import os
import aiohttp
from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class XaiAdvancedProvider(BaseProvider):
    """
    Advanced xAI provider with Live Search and Reasoning support.
    Uses direct API calls to support all features.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key: str = config.get("api_key") or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key missing. Set XAI_API_KEY environment variable or add to config")
        
        # xAI endpoint
        self.base_url: str = config.get("endpoint", "https://api.x.ai/v1/chat/completions")
        
        # Model settings with xAI pricing (per million tokens)
        self.model_settings: Dict[str, Dict[str, Any]] = {
            "grok-4": {
                "max_tokens": 131072,
                "cost_per_million_input": 2.00,  # $2/M input tokens
                "cost_per_million_output": 10.00,  # $10/M output tokens (includes reasoning tokens)
                "context_window": 256000,
                "supports_vision": True,
                "is_reasoning": True,
                "returns_reasoning_content": False,  # grok-4 doesn't return reasoning_content
                "supports_reasoning_effort": False  # grok-4 doesn't support reasoning_effort
            },
            "grok-4-0709": {
                "max_tokens": 131072,
                "cost_per_million_input": 2.00,
                "cost_per_million_output": 10.00,
                "context_window": 256000,
                "supports_vision": True,
                "is_reasoning": True,
                "returns_reasoning_content": False,
                "supports_reasoning_effort": False
            },
            "grok-3": {
                "max_tokens": 131072,
                "cost_per_million_input": 15.00,  # $15/M input
                "cost_per_million_output": 60.00,  # $60/M output
                "context_window": 131072,
                "supports_vision": True,
                "is_reasoning": False
            },
            "grok-3-mini": {
                "max_tokens": 131072,
                "cost_per_million_input": 1.00,  # $1/M input
                "cost_per_million_output": 4.00,  # $4/M output (includes reasoning tokens)
                "context_window": 131072,
                "supports_vision": True,
                "is_reasoning": True,
                "returns_reasoning_content": True,  # Returns reasoning trace
                "supports_reasoning_effort": True,
                "default_reasoning_effort": "low"  # Can be "low" or "high"
            },
            "grok-2-image-1212": {
                "max_tokens": 0,  # Image generation model
                "cost_per_image": 0.10,  # $0.10 per image
                "context_window": 0,
                "supports_vision": False,
                "is_reasoning": False,
                "is_image_generation": True
            }
        }
        
        # Model aliases for convenience
        self.aliases: Dict[str, str] = config.get("aliases", {
            "grok": "grok-4",
            "grok-mini": "grok-3-mini",
            "grok-vision": "grok-4",
            "grok-reasoning": "grok-4",
            "grok-image": "grok-2-image-1212"
        })
        
        self.default_model = config.get("default_model", "grok-3-mini")
        logger.info(f"xAI Advanced provider initialized with {len(self.model_settings)} models")

    def resolve_model_alias(self, alias: str) -> str:
        """Resolve model alias to actual model name"""
        return self.aliases.get(alias, alias)

    async def complete(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str,
        **kwargs
    ) -> str:
        """
        Complete a prompt using xAI's Grok models with Live Search support.
        
        Args:
            prompt: The prompt to complete (string or message list)
            model: The model to use
            **kwargs: Additional parameters including:
                - search_parameters: Dict with Live Search configuration
                - temperature: Generation temperature
                - max_tokens: Maximum tokens to generate
                - reasoning_effort: For reasoning models (low/high)
                - return_reasoning: Include reasoning trace in output
            
        Returns:
            The completion text (and reasoning_content if available)
        """
        actual_model = self.resolve_model_alias(model)
        
        if actual_model not in self.model_settings:
            logger.warning(f"Unknown model '{actual_model}', falling back to {self.default_model}")
            actual_model = self.default_model
        
        settings = self.model_settings[actual_model]
        
        # Handle image generation models differently
        if settings.get("is_image_generation"):
            raise NotImplementedError("Image generation not yet implemented")
        
        # Prepare messages
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are Grok, a highly intelligent, helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        
        # Build request parameters
        params = {
            "model": actual_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", settings["max_tokens"]),
            "stream": False
        }
        
        # Add Live Search parameters if provided
        if "search_parameters" in kwargs:
            search_params = kwargs["search_parameters"]
            
            # Build search parameters dict
            search_config = {}
            
            # Mode: auto, on, off
            if "mode" in search_params:
                search_config["mode"] = search_params["mode"]
            
            # Return citations
            if "return_citations" in search_params:
                search_config["return_citations"] = search_params["return_citations"]
            
            # Date range
            if "from_date" in search_params:
                search_config["from_date"] = search_params["from_date"]
            if "to_date" in search_params:
                search_config["to_date"] = search_params["to_date"]
            
            # Max search results
            if "max_search_results" in search_params:
                search_config["max_search_results"] = search_params["max_search_results"]
            
            # Data sources
            if "sources" in search_params:
                sources = []
                for source in search_params["sources"]:
                    source_config = {"type": source.get("type", "web")}
                    
                    # Add source-specific parameters
                    if "country" in source:
                        source_config["country"] = source["country"]
                    if "excluded_websites" in source:
                        source_config["excluded_websites"] = source["excluded_websites"]
                    if "allowed_websites" in source:
                        source_config["allowed_websites"] = source["allowed_websites"]
                    if "safe_search" in source:
                        source_config["safe_search"] = source["safe_search"]
                    if "included_x_handles" in source:
                        source_config["included_x_handles"] = source["included_x_handles"]
                    if "excluded_x_handles" in source:
                        source_config["excluded_x_handles"] = source["excluded_x_handles"]
                    if "post_favorite_count" in source:
                        source_config["post_favorite_count"] = source["post_favorite_count"]
                    if "post_view_count" in source:
                        source_config["post_view_count"] = source["post_view_count"]
                    if "links" in source:
                        source_config["links"] = source["links"]
                    
                    sources.append(source_config)
                
                search_config["sources"] = sources
            
            # Add search parameters to request
            params["search_parameters"] = search_config
            logger.info(f"Live Search enabled with config: {search_config}")
        
        # Add reasoning_effort for supported models
        if settings.get("is_reasoning") and settings.get("supports_reasoning_effort"):
            reasoning_effort = kwargs.get("reasoning_effort", settings.get("default_reasoning_effort", "low"))
            params["reasoning_effort"] = reasoning_effort
            logger.info(f"Using reasoning_effort: {reasoning_effort}")
        
        # Initialize session if needed
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            # Make the API call
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"xAI API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract the response
                if 'choices' not in data or not data['choices']:
                    raise Exception("No response content from xAI Grok")
                
                choice = data['choices'][0]
                content = choice['message']['content']
                
                # Handle citations if Live Search was used
                citations_info = ""
                if 'citations' in data and data['citations']:
                    citations_info = f"\n\n**Sources:**\n"
                    for i, citation in enumerate(data['citations'], 1):
                        citations_info += f"{i}. {citation}\n"
                    logger.info(f"Live Search returned {len(data['citations'])} citations")
                
                # Handle reasoning content if available
                reasoning_trace = ""
                if 'reasoning_content' in choice.get('message', {}):
                    reasoning_chars = len(choice['message']['reasoning_content'])
                    logger.info(f"Reasoning trace available: {reasoning_chars} chars")
                    if kwargs.get("return_reasoning", False):
                        reasoning_trace = f"\n\n**Reasoning Process:**\n{choice['message']['reasoning_content']}\n"
                
                # Log token usage
                if 'usage' in data:
                    usage = data['usage']
                    logger.debug(f"Tokens used - Input: {usage.get('prompt_tokens', 0)}, "
                               f"Output: {usage.get('completion_tokens', 0)}, "
                               f"Total: {usage.get('total_tokens', 0)}")
                    
                    # Log Live Search sources used
                    if 'num_sources_used' in usage:
                        sources_used = usage['num_sources_used']
                        search_cost = sources_used * 0.025  # $0.025 per source
                        logger.info(f"Live Search sources used: {sources_used} (Cost: ${search_cost:.3f})")
                    
                    # Log reasoning tokens if available
                    if 'reasoning_tokens' in usage:
                        reasoning_tokens = usage['reasoning_tokens']
                        reasoning_cost = (reasoning_tokens / 1_000_000) * settings["cost_per_million_output"]
                        logger.info(f"Reasoning tokens: {reasoning_tokens} (Cost: ${reasoning_cost:.4f})")
                        logger.info(f"Total completion tokens (including reasoning): {usage['completion_tokens']}")
                
                # Return content with optional citations and reasoning
                return content + citations_info + reasoning_trace
                
        except Exception as e:
            logger.error(f"xAI API error: {e}")
            raise

    def estimate_tokens(self, prompt: str, response: str = "", model: str = None) -> int:
        """Estimate token count for prompt and response"""
        # Simple estimation: ~4 chars per token (same as OpenAI)
        total_chars = len(prompt) + len(response)
        return int(total_chars / 4)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate estimated cost for token usage"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings[self.default_model])
        
        if settings.get("is_image_generation"):
            # For image generation, cost is per image
            return settings.get("cost_per_image", 0.10)
        
        input_cost = (prompt_tokens / 1_000_000) * settings["cost_per_million_input"]
        output_cost = (completion_tokens / 1_000_000) * settings["cost_per_million_output"]
        
        return input_cost + output_cost

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        actual_model = self.resolve_model_alias(model)
        return self.model_settings.get(actual_model, self.model_settings[self.default_model])

    def list_available_models(self) -> List[str]:
        """List all available models"""
        return list(self.model_settings.keys())

    async def close(self):
        """Cleanup resources"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("xAI session closed")
            except Exception as e:
                logger.warning(f"Error closing xAI session: {e}")
        
        await super().close()

    def __repr__(self) -> str:
        return f"XaiAdvancedProvider(models={len(self.model_settings)}, default={self.default_model})"