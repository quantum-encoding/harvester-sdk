"""
Generic provider for custom or unsupported providers
"""
import aiohttp
import json
from typing import Dict, Any
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class GenericProvider(BaseProvider):
    """Generic provider that can be configured for any API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.endpoint = config.get('endpoint')
        self.headers = config.get('headers', {})
        self.request_format = config.get('request_format', 'openai')  # openai, anthropic, custom
    
    async def complete(self, prompt: str, model: str) -> str:
        """Send completion request using configured format"""
        # Resolve model alias
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        estimated_tokens = self.estimate_tokens(prompt, "")
        await self._apply_rate_limit(estimated_tokens)
        
        # Prepare request based on format
        if self.request_format == 'openai':
            payload = self._prepare_openai_format(prompt, actual_model)
        elif self.request_format == 'anthropic':
            payload = self._prepare_anthropic_format(prompt, actual_model)
        else:
            payload = self._prepare_custom_format(prompt, actual_model)
        
        # Add API key to headers if needed
        headers = self.headers.copy()
        if self.api_key and 'Authorization' not in headers:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Make request
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(
                self.endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract response based on format
                return self._extract_response(data)
                
        except Exception as e:
            logger.error(f"Generic provider completion error: {e}")
            raise
    
    def _prepare_openai_format(self, prompt: str, model: str) -> Dict:
        """Prepare request in OpenAI format"""
        return {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 4096,
            'temperature': 0.7
        }
    
    def _prepare_anthropic_format(self, prompt: str, model: str) -> Dict:
        """Prepare request in Anthropic format"""
        return {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 4096
        }
    
    def _prepare_custom_format(self, prompt: str, model: str) -> Dict:
        """Prepare request in custom format"""
        # Override this in config
        return {
            'prompt': prompt,
            'model': model
        }
    
    def _extract_response(self, data: Dict) -> str:
        """Extract response from API response"""
        # Try common response formats
        if 'choices' in data and data['choices']:
            return data['choices'][0].get('message', {}).get('content', '')
        
        if 'content' in data and data['content']:
            return data['content'][0].get('text', '')
        
        if 'response' in data:
            return data['response']
        
        if 'text' in data:
            return data['text']
        
        raise Exception(f"Unable to extract response from: {data}")
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        return aliases.get(alias, alias)
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count"""
        total_chars = len(prompt) + len(response)
        return int(total_chars / 4)
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        # Use configured cost or default
        cost_per_million = self.config.get('cost_per_million_tokens', 1.0)
        return (tokens / 1_000_000) * cost_per_million
