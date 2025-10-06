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
        self.beta_url = 'https://api.x.ai/v1/beta/chat/completions'  # For structured outputs
        
        # Grok model settings (from xAI pricing table)
        self.model_settings = {
            # Grok 4 - Latest flagship model (July 2024)
            'grok-4-0709': {
                'max_tokens': 2000000,  # 2M output
                'temperature': 0.7,
                'cost_per_million_input': 5.00,
                'cost_per_million_output': 15.00,
                'context_window': 256000,  # 256K context
                'rate_limit': 480  # TPM
            },
            # Grok 4 Fast Reasoning - 2M context, 4M output
            'grok-4-fast-reasoning': {
                'max_tokens': 4000000,  # 4M output
                'temperature': 0.7,
                'cost_per_million_input': 5.00,
                'cost_per_million_output': 15.00,
                'context_window': 2000000,  # 2M context
                'rate_limit': 480  # TPM
            },
            # Grok 4 Fast Non-Reasoning - 2M context, 4M output
            'grok-4-fast-non-reasoning': {
                'max_tokens': 4000000,  # 4M output
                'temperature': 0.7,
                'cost_per_million_input': 5.00,
                'cost_per_million_output': 15.00,
                'context_window': 2000000,  # 2M context
                'rate_limit': 480  # TPM
            },
            # Grok Code Fast - 256K context, 2M output
            'grok-code-fast-1': {
                'max_tokens': 2000000,  # 2M output
                'temperature': 0.7,
                'cost_per_million_input': 2.00,
                'cost_per_million_output': 10.00,
                'context_window': 256000,  # 256K context
                'rate_limit': 480  # TPM
            },
            # Grok 3 - Previous generation
            'grok-3': {
                'max_tokens': 131072,  # 131K output
                'temperature': 0.7,
                'cost_per_million_input': 2.00,
                'cost_per_million_output': 10.00,
                'context_window': 131072,  # 131K context
                'rate_limit': 600  # TPM
            },
            # Grok 3 Mini - Fast, lightweight
            'grok-3-mini': {
                'max_tokens': 131072,  # 131K output
                'temperature': 0.7,
                'cost_per_million_input': 0.50,
                'cost_per_million_output': 2.00,
                'context_window': 131072,  # 131K context
                'rate_limit': 480  # TPM
            },
            # Grok 2 Vision EU (December 2023) - Vision model with structured output
            'grok-2-vision-1212-eu-west-1': {
                'max_tokens': 32768,   # Output matches context
                'temperature': 0.7,
                'cost_per_million_input': 1.00,
                'cost_per_million_output': 3.00,
                'context_window': 32768,  # 32K context
                'supports_vision': True,
                'supports_structured': True,  # Supports structured outputs
                'rate_limit': 50  # TPM (EU region)
            },
            # Grok 2 Vision US (December 2023) - Vision model with structured output
            'grok-2-vision-1212-us-east-1': {
                'max_tokens': 32768,   # Output matches context
                'temperature': 0.7,
                'cost_per_million_input': 1.00,
                'cost_per_million_output': 3.00,
                'context_window': 32768,  # 32K context
                'supports_vision': True,
                'supports_structured': True,  # Supports structured outputs
                'rate_limit': 600  # TPM (US region - higher limit)
            },
            # Grok 2 Image - Image generation model (December 2023)
            'grok-2-image-1212': {
                'type': 'image_generation',  # Not a language model
                'cost_per_image': 1.00,  # Per image pricing
                'rate_limit': 300  # images per month
                # Note: This is an image generation model, not text completion
            }
        }
    
    async def complete(self, prompt: str, model: str, **kwargs) -> str:
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
            'temperature': settings.get('temperature', 0.7),
            'stream': False
        }

        # Handle structured output (OpenAI-compatible response_format)
        use_beta_endpoint = False
        if 'json_schema' in kwargs:
            schema = kwargs['json_schema']
            # xAI uses OpenAI-compatible structured output via beta endpoint
            # Convert Pydantic schema to OpenAI's response_format
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': schema.get('name', 'response'),
                    'strict': True,
                    'schema': schema.get('schema', schema)
                }
            }
            use_beta_endpoint = True

        # Make request (use beta endpoint for structured outputs)
        endpoint = self.beta_url if use_beta_endpoint else self.base_url

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(
                endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"xAI API error: {response.status} - {error_text}")
                
                data = await response.json()

                # Extract response text
                if 'choices' in data and data['choices']:
                    content = data['choices'][0]['message']['content']

                    # For structured outputs, check for 'parsed' field
                    if 'parsed' in data['choices'][0]['message']:
                        import json
                        return json.dumps(data['choices'][0]['message']['parsed'])

                    if not content:
                        logger.warning(f"Empty content from xAI. Full response: {data}")
                    return content

                logger.error(f"Unexpected xAI response structure: {data}")
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