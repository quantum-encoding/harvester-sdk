"""
OpenAI DALL-E image generation provider implementation
"""
import asyncio
import aiohttp
import os
import base64
import time
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class OpenAIImageProvider(BaseProvider):
    """Provider for OpenAI DALL-E image generation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required for image generation")
        
        # Model-specific settings
        self.model_settings = {
            'dall-e-3': {
                'max_prompts': 1,  # DALL-E 3 only supports 1 image per request
                'sizes': ['1024x1024', '1024x1792', '1792x1024'],
                'quality': ['standard', 'hd'],
                'style': ['vivid', 'natural'],
                'cost_per_image_standard': 0.040,  # $0.040 for 1024x1024 standard
                'cost_per_image_hd': 0.080,       # $0.080 for 1024x1024 HD
                'cost_per_image_hd_large': 0.120  # $0.120 for 1792x sizes HD
            },
            'dall-e-2': {
                'max_prompts': 10,  # DALL-E 2 supports up to 10 images per request
                'sizes': ['256x256', '512x512', '1024x1024'],
                'cost_per_image_256': 0.016,   # $0.016 for 256x256
                'cost_per_image_512': 0.018,   # $0.018 for 512x512
                'cost_per_image_1024': 0.020   # $0.020 for 1024x1024
            }
        }
        
        # Default generation parameters
        self.default_params = {
            'size': '1024x1024',
            'quality': 'standard',
            'style': 'vivid',
            'response_format': 'url',  # or 'b64_json'
            'n': 1  # number of images to generate
        }
    
    async def complete(self, prompt: str, model: str) -> str:
        """Generate image using DALL-E and return image URL or base64 data"""
        # This is actually generate_image for image providers
        return await self.generate_image(prompt, model)
    
    async def generate_image(
        self, 
        prompt: str, 
        model: str,
        size: str = None,
        quality: str = None,
        style: str = None,
        response_format: str = 'url',
        n: int = 1
    ) -> str:
        """
        Generate image using OpenAI DALL-E
        
        Args:
            prompt: The image generation prompt
            model: Model alias (gpt-1, gpt-2) or direct model name
            size: Image size (e.g., '1024x1024')
            quality: Image quality ('standard' or 'hd')
            style: Image style ('vivid' or 'natural')
            response_format: Response format ('url' or 'b64_json')
            n: Number of images to generate
            
        Returns:
            JSON string with image URLs or base64 data
        """
        # Resolve model alias to actual model name
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        await self._apply_rate_limit(1)  # 1 request
        
        # Prepare request parameters
        params = {
            'prompt': prompt,
            'model': actual_model,
            'response_format': response_format,
            'n': min(n, self.model_settings[actual_model]['max_prompts'])
        }
        
        # Add model-specific parameters
        settings = self.model_settings.get(actual_model, self.model_settings['dall-e-3'])
        
        # Set size
        if size:
            if size in settings.get('sizes', []):
                params['size'] = size
            else:
                logger.warning(f"Size {size} not supported for {actual_model}, using default")
                params['size'] = settings['sizes'][0]
        else:
            params['size'] = self.default_params['size']
        
        # DALL-E 3 specific parameters
        if actual_model == 'dall-e-3':
            if quality and quality in settings['quality']:
                params['quality'] = quality
            else:
                params['quality'] = self.default_params['quality']
                
            if style and style in settings['style']:
                params['style'] = style
            else:
                params['style'] = self.default_params['style']
        
        # Make API request
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        url = 'https://api.openai.com/v1/images/generations'
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(url, headers=headers, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract image data
                if 'data' in data and data['data']:
                    images = []
                    for img_data in data['data']:
                        if response_format == 'url':
                            images.append({
                                'url': img_data.get('url'),
                                'revised_prompt': img_data.get('revised_prompt', prompt)
                            })
                        elif response_format == 'b64_json':
                            images.append({
                                'b64_json': img_data.get('b64_json'),
                                'revised_prompt': img_data.get('revised_prompt', prompt)
                            })
                    
                    # Return as JSON string for consistency with text providers
                    import json
                    return json.dumps({
                        'model': actual_model,
                        'prompt': prompt,
                        'parameters': params,
                        'images': images,
                        'count': len(images)
                    })
                else:
                    raise Exception("No image data in OpenAI response")
                    
        except Exception as e:
            logger.error(f"OpenAI image generation error: {e}")
            raise Exception(f"OpenAI DALL-E error: {e}")
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        resolved = aliases.get(alias, alias)
        
        # Map common aliases to DALL-E models
        if resolved in ['dall-e-3', 'dalle3', 'dall-e-3']:
            return 'dall-e-3'
        elif resolved in ['dall-e-2', 'dalle2', 'dall-e-2']:
            return 'dall-e-2'
        
        # Default to DALL-E 3 if unknown
        if resolved not in self.model_settings:
            logger.warning(f"Unknown model {resolved}, defaulting to dall-e-3")
            return 'dall-e-3'
        
        logger.debug(f"Resolved model alias {alias} -> {resolved}")
        return resolved
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count for image generation
        
        For image generation, we count the prompt tokens only
        The response is image data, not text tokens
        """
        # Rough estimation: 1 token per 4 characters for prompt
        prompt_tokens = len(prompt) // 4
        logger.debug(f"Estimated {prompt_tokens} tokens for prompt")
        return prompt_tokens
    
    def estimate_cost(self, tokens: int, model: str, **kwargs) -> float:
        """Estimate cost for image generation based on model and parameters"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings['dall-e-3'])
        
        # Get parameters from kwargs
        size = kwargs.get('size', '1024x1024')
        quality = kwargs.get('quality', 'standard')
        n = kwargs.get('n', 1)
        
        # Calculate cost based on model and parameters
        if actual_model == 'dall-e-3':
            if quality == 'hd':
                if size in ['1792x1024', '1024x1792']:
                    cost_per_image = settings['cost_per_image_hd_large']
                else:
                    cost_per_image = settings['cost_per_image_hd']
            else:
                cost_per_image = settings['cost_per_image_standard']
        elif actual_model == 'dall-e-2':
            if size == '256x256':
                cost_per_image = settings['cost_per_image_256']
            elif size == '512x512':
                cost_per_image = settings['cost_per_image_512']
            else:
                cost_per_image = settings['cost_per_image_1024']
        else:
            cost_per_image = 0.040  # Fallback
        
        total_cost = cost_per_image * n
        logger.debug(f"Estimated cost for {n} images: ${total_cost:.4f}")
        
        return total_cost
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("OpenAI session closed")
            except Exception as e:
                logger.warning(f"Error closing OpenAI session: {e}")
        
        await super().close()
    
    def get_available_models(self) -> list:
        """Get list of available image models"""
        return list(self.model_settings.keys())
    
    def validate_configuration(self) -> dict:
        """Validate provider configuration"""
        status = {
            'provider': 'openai_image',
            'authentication': bool(self.api_key),
            'models_available': len(self.model_settings),
            'issues': []
        }
        
        if not self.api_key:
            status['issues'].append("OPENAI_API_KEY not set")
        
        return status