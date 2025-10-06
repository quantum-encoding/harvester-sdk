"""
OpenAI DALL-E 3 image generation provider implementation
Clean implementation for DALL-E 3 only (DALL-E 2 deprecated)
"""
import asyncio
import aiohttp
import os
import json
import time
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class DallE3Provider(BaseProvider):
    """Provider for OpenAI DALL-E 3 image generation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Smart authentication: env vars → config → error
        self.api_key = (
            os.environ.get('OPENAI_API_KEY') or 
            config.get('api_key')
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set environment variable:\n"
                "  export OPENAI_API_KEY=your-api-key\n"
                "Or add to config:\n"
                "  api_key: your-api-key"
            )
        
        # DALL-E 3 specific settings
        self.model_settings = {
            'dall-e-3': {
                'max_prompts': 1,  # DALL-E 3 only supports 1 image per request
                'sizes': ['1024x1024', '1024x1792', '1792x1024'],
                'quality': ['standard', 'hd'],
                'style': ['vivid', 'natural'],
                'cost_per_image_standard': 0.040,  # $0.040 for 1024x1024 standard
                'cost_per_image_hd': 0.080,       # $0.080 for 1024x1024 HD
                'cost_per_image_hd_large': 0.120  # $0.120 for 1792x sizes HD
            }
        }
        
        # Default generation parameters
        self.default_params = {
            'size': '1024x1024',
            'quality': 'standard',
            'style': 'vivid',
            'response_format': 'b64_json',  # Return base64 for file saving
            'n': 1  # DALL-E 3 only supports 1
        }
        
        self.endpoint = 'https://api.openai.com/v1/images/generations'
    
    async def complete(self, prompt: str, model: str) -> str:
        """Generate image using DALL-E 3 and return image data"""
        return await self.generate_image(prompt, model)
    
    async def generate_image(
        self, 
        prompt: str, 
        model: str,
        size: str = None,
        quality: str = None,
        style: str = None,
        response_format: str = None,
        n: int = 1,
        **kwargs
    ) -> str:
        """
        Generate image using OpenAI DALL-E 3
        
        Args:
            prompt: The image generation prompt
            model: Model alias (dall-e-3) or direct model name
            size: Image size ('1024x1024', '1024x1792', '1792x1024')
            quality: Image quality ('standard' or 'hd')
            style: Image style ('vivid' or 'natural')
            response_format: Response format ('url' or 'b64_json')
            n: Number of images (always 1 for DALL-E 3)
            
        Returns:
            JSON string with image data and metadata
        """
        # Resolve model alias to actual model name
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        await self._apply_rate_limit(1)  # 1 request
        
        # Get model settings
        settings = self.model_settings['dall-e-3']
        
        # Build parameters
        params = {
            'prompt': prompt,
            'model': actual_model,
            'response_format': response_format or self.default_params['response_format'],
            'n': 1  # DALL-E 3 always generates 1 image
        }
        
        # Validate and set size
        if size and size in settings['sizes']:
            params['size'] = size
        else:
            if size:
                logger.warning(f"Size {size} not supported for DALL-E 3, using default")
            params['size'] = self.default_params['size']
        
        # Validate and set quality
        if quality and quality in settings['quality']:
            params['quality'] = quality
        else:
            if quality:
                logger.warning(f"Quality {quality} not supported for DALL-E 3, using default")
            params['quality'] = self.default_params['quality']
            
        # Validate and set style
        if style and style in settings['style']:
            params['style'] = style
        else:
            if style:
                logger.warning(f"Style {style} not supported for DALL-E 3, using default")
            params['style'] = self.default_params['style']
        
        # Make API request
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(self.endpoint, headers=headers, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract image data
                if 'data' in data and data['data']:
                    images = []
                    for img_data in data['data']:
                        image_info = {
                            'revised_prompt': img_data.get('revised_prompt', prompt)
                        }
                        
                        if params['response_format'] == 'url':
                            image_info['url'] = img_data.get('url')
                        elif params['response_format'] == 'b64_json':
                            image_info['b64_json'] = img_data.get('b64_json')
                        
                        images.append(image_info)
                    
                    # Return standardized format
                    return json.dumps({
                        'model': actual_model,
                        'prompt': prompt,
                        'parameters': params,
                        'images': images,
                        'count': len(images),
                        'provider': 'dalle3'
                    })
                else:
                    raise Exception("No image data in OpenAI response")
                    
        except Exception as e:
            logger.error(f"DALL-E 3 image generation error: {e}")
            raise Exception(f"DALL-E 3 error: {e}")
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        resolved = aliases.get(alias, alias)
        
        # Map aliases to DALL-E 3
        alias_mapping = {
            'dall-e-3': 'dall-e-3',
            'dalle3': 'dall-e-3', 
            'dalle-3': 'dall-e-3'
        }
        
        if resolved in alias_mapping:
            resolved = alias_mapping[resolved]
        
        # Only DALL-E 3 is supported
        if resolved != 'dall-e-3':
            logger.warning(f"Model {resolved} not supported, using dall-e-3")
            return 'dall-e-3'
        
        logger.debug(f"Resolved model alias {alias} -> {resolved}")
        return resolved
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for image generation (prompt tokens only)"""
        prompt_tokens = len(prompt) // 4
        logger.debug(f"Estimated {prompt_tokens} tokens for prompt")
        return prompt_tokens
    
    def estimate_cost(self, tokens: int, model: str, **kwargs) -> float:
        """Estimate cost for DALL-E 3 image generation"""
        settings = self.model_settings['dall-e-3']
        
        # Get parameters
        size = kwargs.get('size', '1024x1024')
        quality = kwargs.get('quality', 'standard')
        n = kwargs.get('n', 1)  # Always 1 for DALL-E 3
        
        # Calculate cost based on quality and size
        if quality == 'hd':
            if size in ['1792x1024', '1024x1792']:
                cost_per_image = settings['cost_per_image_hd_large']
            else:
                cost_per_image = settings['cost_per_image_hd']
        else:
            cost_per_image = settings['cost_per_image_standard']
        
        total_cost = cost_per_image * n
        logger.debug(f"Estimated cost for DALL-E 3: ${total_cost:.4f}")
        
        return total_cost
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("DALL-E 3 session closed")
            except Exception as e:
                logger.warning(f"Error closing DALL-E 3 session: {e}")
        
        await super().close()
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return ['dall-e-3']
    
    def validate_configuration(self) -> dict:
        """Validate provider configuration"""
        status = {
            'provider': 'dalle3',
            'authentication': bool(self.api_key),
            'models_available': 1,
            'issues': []
        }
        
        if not self.api_key:
            status['issues'].append("OPENAI_API_KEY not set")
        
        return status