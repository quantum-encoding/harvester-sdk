"""
Anthropic (Claude) provider implementation
"""
import aiohttp
import json
from typing import Dict, Any
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('endpoint', 'https://api.anthropic.com/v1/messages')
        self.anthropic_version = config.get('headers', {}).get('anthropic-version', '2023-06-01')
        
        # Model-specific settings
        self.model_settings = {
            'claude-3-opus-20240229': {
                'max_tokens': 4096,
                'cost_per_million_input': 15.00,
                'cost_per_million_output': 75.00
            },
            'claude-3-sonnet-20240229': {
                'max_tokens': 4096,
                'cost_per_million_input': 3.00,
                'cost_per_million_output': 15.00
            },
            'claude-2.1': {
                'max_tokens': 4096,
                'cost_per_million_input': 8.00,
                'cost_per_million_output': 24.00
            }
        }
    
    async def complete(self, prompt: str, model: str) -> str:
        """Send completion request to Anthropic"""
        # Resolve model alias
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        estimated_tokens = self.estimate_tokens(prompt, "")
        await self._apply_rate_limit(estimated_tokens)
        
        # Prepare request
        settings = self.model_settings.get(actual_model, self.model_settings['claude-2.1'])
        
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': self.anthropic_version,
            'content-type': 'application/json'
        }
        
        payload = {
            'model': actual_model,
            'messages': [{
                'role': 'user',
                'content': prompt
            }],
            'max_tokens': settings['max_tokens'],
            'temperature': 0.7
        }
        
        # Make request
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract response text
                if 'content' in data and data['content']:
                    return data['content'][0].get('text', '')
                
                raise Exception("No response content from Anthropic")
                
        except Exception as e:
            logger.error(f"Anthropic completion error: {e}")
            raise
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        return aliases.get(alias, alias)
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for Claude models"""
        # Claude uses a similar tokenization to GPT models
        # Approximately 1 token per 4 characters
        total_chars = len(prompt) + len(response)
        return int(total_chars / 4)
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings['claude-2.1'])
        
        # Assume 70% input, 30% output for cost calculation
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        input_cost = (input_tokens / 1_000_000) * settings['cost_per_million_input']
        output_cost = (output_tokens / 1_000_000) * settings['cost_per_million_output']
        
        return input_cost + output_cost