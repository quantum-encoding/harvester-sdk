"""
xAI (Grok) provider implementation
"""
import aiohttp
import os
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class XAIProvider(BaseProvider):
    """Provider for xAI Grok models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Smart authentication: env vars → config → error
        self.api_key = (
            os.environ.get('XAI_API_KEY') or 
            config.get('api_key')
        )
        if not self.api_key:
            raise ValueError(
                "xAI API key required. Set environment variable:\n"
                "  export XAI_API_KEY=your-api-key\n"
                "Or add to config:\n"
                "  api_key: your-api-key"
            )
        
        self.base_url = config.get('endpoint', 'https://api.x.ai/v1/chat/completions')
        
        # Grok model settings
        self.model_settings = {
            # Grok 4 - Latest flagship model (July 2024)
            'grok-4-0709': {
                'max_tokens': 128000,  # 128K output
                'temperature': 0.7,
                'cost_per_million_input': 5.00,
                'cost_per_million_output': 15.00,
                'context_window': 131072  # 128K context
            },
            # Grok 3 - Previous generation
            'grok-3': {
                'max_tokens': 65536,   # 64K output
                'temperature': 0.7,
                'cost_per_million_input': 2.00,
                'cost_per_million_output': 10.00,
                'context_window': 65536  # 64K context
            },
            # Grok 3 Mini - Fast, lightweight
            'grok-3-mini': {
                'max_tokens': 32768,   # 32K output
                'temperature': 0.7,
                'cost_per_million_input': 0.50,
                'cost_per_million_output': 2.00,
                'context_window': 32768  # 32K context
            },
            # Grok 2 Image - Multimodal vision model (December 2023)
            'grok-2-image-1212': {
                'max_tokens': 8192,    # 8K output
                'temperature': 0.7,
                'cost_per_million_input': 1.00,
                'cost_per_million_output': 3.00,
                'context_window': 32768,  # 32K context
                'supports_vision': True
            }
        }
    
    async def complete(self, prompt: str, model: str) -> str:
        """Send completion request to xAI Grok"""
        # Resolve model alias
        actual_model = self.resolve_model_alias(model)
        
        # Verify model is supported
        if actual_model not in self.model_settings:
            logger.warning(f"Unknown model {actual_model}, defaulting to grok-3")
            actual_model = 'grok-3'
        
        # Apply rate limiting
        estimated_tokens = self.estimate_tokens(prompt, "")
        await self._apply_rate_limit(estimated_tokens)
        
        # Prepare request
        settings = self.model_settings[actual_model]
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': actual_model,
            'messages': [{
                'role': 'user',
                'content': prompt
            }],
            'max_tokens': settings['max_tokens'],
            'temperature': settings['temperature'],
            'stream': False
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
                    raise Exception(f"xAI API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract response text
                if 'choices' in data and data['choices']:
                    return data['choices'][0]['message']['content']
                
                raise Exception("No response content from xAI Grok")
                
        except Exception as e:
            logger.error(f"xAI Grok completion error: {e}")
            raise
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        resolved = aliases.get(alias, alias)
        
        # Map common aliases to Grok models
        alias_mapping = {
            'grok-4': 'grok-4-0709',
            'grok-fast': 'grok-3-mini',
            'grok-image': 'grok-2-image-1212',
            'grok-vision': 'grok-2-image-1212',
            'grok': 'grok-3'  # Default to Grok 3
        }
        
        if resolved in alias_mapping:
            resolved = alias_mapping[resolved]
        
        logger.debug(f"Resolved model alias {alias} -> {resolved}")
        return resolved
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for Grok models"""
        total_chars = len(prompt) + len(response)
        # Grok uses similar tokenization to GPT models
        # Roughly 1 token per 4 characters for English text
        estimated_tokens = int(total_chars / 4)
        logger.debug(f"Estimated {estimated_tokens} tokens for {total_chars} characters")
        return estimated_tokens
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage based on Grok pricing"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings['grok-3'])
        
        # For cost estimation, assume 70% input tokens, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * settings['cost_per_million_input']
        output_cost = (output_tokens / 1_000_000) * settings['cost_per_million_output']
        
        total_cost = input_cost + output_cost
        logger.debug(f"Estimated cost for {tokens} tokens: ${total_cost:.6f}")
        
        return total_cost
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("xAI session closed")
            except Exception as e:
                logger.warning(f"Error closing xAI session: {e}")
        
        await super().close()
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return list(self.model_settings.keys())
    
    def validate_configuration(self) -> dict:
        """Validate provider configuration"""
        status = {
            'provider': 'xai',
            'authentication': bool(self.api_key),
            'models_available': len(self.model_settings),
            'issues': []
        }
        
        if not self.api_key:
            status['issues'].append("XAI_API_KEY not set")
        
        return status