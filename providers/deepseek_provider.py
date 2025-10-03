"""
DeepSeek provider implementation

Supports both OpenAI and Anthropic API formats:
- OpenAI format: https://api.deepseek.com/v1/chat/completions
- Anthropic format: https://api.deepseek.com/anthropic
- V3.1-Terminus: https://api.deepseek.com/v3.1_terminus_expires_on_20251015/chat/completions
"""
import os
import aiohttp
import json
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class DeepseekProvider(BaseProvider):
    """Provider for DeepSeek models with OpenAI and Anthropic API support"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('DEEPSEEK_API_KEY')

        # Default endpoints
        self.openai_endpoint = config.get('endpoint', 'https://api.deepseek.com/v1/chat/completions')
        self.anthropic_endpoint = 'https://api.deepseek.com/anthropic'
        self.terminus_endpoint = config.get('terminus_endpoint',
            'https://api.deepseek.com/v3.1_terminus_expires_on_20251015/chat/completions')

        # API format preference (can be overridden per request)
        self.api_format = config.get('api_format', 'openai')  # 'openai' or 'anthropic'

        # Updated model settings per API documentation (V3.2-Exp - 50% cheaper!)
        self.model_settings = {
            'deepseek-chat': {  # V3.2-Exp with DeepSeek Sparse Attention (DSA)
                'max_tokens': 8000,
                'temperature': 0.7,
                'cost_per_million_input_cache_hit': 0.04,    # 50% reduction
                'cost_per_million_input_cache_miss': 0.14,   # 50% reduction
                'cost_per_million_output': 0.55              # 50% reduction
            },
            'deepseek-reasoner': {  # DeepSeek R1 Reasoner
                'max_tokens': 64000,
                'temperature': 0.7,
                'cost_per_million_input_cache_hit': 0.07,    # 50% reduction
                'cost_per_million_input_cache_miss': 0.28,   # 50% reduction
                'cost_per_million_output': 1.10              # 50% reduction
            },
            'deepseek-v3.1-terminus': {  # Legacy for comparison (expires Oct 15, 2025)
                'max_tokens': 8000,
                'temperature': 0.7,
                'cost_per_million_input_cache_hit': 0.04,
                'cost_per_million_input_cache_miss': 0.14,
                'cost_per_million_output': 0.55,
                'endpoint_override': self.terminus_endpoint
            }
        }
    
    async def complete(self, prompt: str, model: str, api_format: Optional[str] = None) -> str:
        """Send completion request to DeepSeek (supports OpenAI and Anthropic formats)"""
        # Resolve model alias
        actual_model = self.resolve_model_alias(model)

        # Verify model is supported
        if actual_model not in self.model_settings:
            raise ValueError(f"Unsupported model: {actual_model}. Valid models: {list(self.model_settings.keys())}")

        # Apply rate limiting
        estimated_tokens = self.estimate_tokens(prompt, "")
        await self._apply_rate_limit(estimated_tokens)

        # Get model settings
        settings = self.model_settings[actual_model]

        # Determine endpoint (check for model-specific override)
        endpoint = settings.get('endpoint_override', self.openai_endpoint)

        # Determine API format
        format_to_use = api_format or self.api_format

        if format_to_use == 'anthropic':
            return await self._complete_anthropic(prompt, actual_model, settings)
        else:
            return await self._complete_openai(prompt, actual_model, settings, endpoint)

    async def _complete_openai(self, prompt: str, model: str, settings: Dict, endpoint: str) -> str:
        """Complete using OpenAI-compatible API format"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': model,
            'messages': [{
                'role': 'user',
                'content': prompt
            }],
            'max_tokens': settings['max_tokens'],
            'temperature': settings['temperature'],
            'stream': False
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(endpoint, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error: {response.status} - {error_text}")

                data = await response.json()

                # Log the actual model used (for V3.1-Terminus verification)
                response_model = data.get('model', 'unknown')
                if 'terminus' in response_model.lower():
                    logger.info(f"✓ Using DeepSeek V3.1-Terminus (comparison mode)")

                if 'choices' in data and data['choices']:
                    return data['choices'][0]['message']['content']

                raise Exception("No response content from DeepSeek")

        except Exception as e:
            logger.error(f"DeepSeek OpenAI API error: {e}")
            raise

    async def _complete_anthropic(self, prompt: str, model: str, settings: Dict) -> str:
        """Complete using Anthropic-compatible API format"""
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': model,
            'max_tokens': settings['max_tokens'],
            'temperature': settings['temperature'],
            'system': 'You are a helpful assistant.',
            'messages': [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt
                }]
            }]
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(
                self.anthropic_endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek Anthropic API error: {response.status} - {error_text}")

                data = await response.json()

                # Extract text from Anthropic format response
                if 'content' in data and data['content']:
                    return data['content'][0]['text']

                raise Exception("No response content from DeepSeek")

        except Exception as e:
            logger.error(f"DeepSeek Anthropic API error: {e}")
            raise
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {
            # V3.2-Exp (default)
            'ds-1': 'deepseek-chat',
            'ds-2': 'deepseek-reasoner',
            'deepseek-chat': 'deepseek-chat',
            'deepseek-reasoner': 'deepseek-reasoner',
            # Legacy V3.1-Terminus
            'ds-1-legacy': 'deepseek-v3.1-terminus',
            'deepseek-v3.1-terminus': 'deepseek-v3.1-terminus',
            # Backward compatibility
            'deepseek-v3': 'deepseek-chat',
            'deepseek-r1': 'deepseek-reasoner'
        })
        return aliases.get(alias, alias)
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for DeepSeek models"""
        # DeepSeek uses similar tokenization to GPT
        # Approximately 1 token per 4 characters
        total_chars = len(prompt) + len(response)
        return int(total_chars / 4)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for token usage (using cache miss pricing)"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model)
        
        if not settings:
            return 0.0
        
        # Using cache miss pricing by default (worst-case scenario)
        input_cost = (input_tokens / 1_000_000) * settings['cost_per_million_input_cache_miss']
        output_cost = (output_tokens / 1_000_000) * settings['cost_per_million_output']
        
        return input_cost + output_cost
    
    def generate_text(self, prompt: str, model: str = 'deepseek-chat', **kwargs) -> str:
        """Synchronous wrapper for complete() - for compatibility"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.complete(prompt, model))
            return result
        finally:
            loop.close()