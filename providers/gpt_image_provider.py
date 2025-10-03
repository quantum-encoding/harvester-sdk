"""
OpenAI GPT Image 1 generation provider implementation
Separate provider for GPT Image 1 format (different from DALL-E 3)
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

class GPTImageProvider(BaseProvider):
    """Provider for OpenAI GPT Image 1 generation"""
    
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
        
        # GPT Image 1 specific settings (different format than DALL-E)
        self.model_settings = {
            'gpt-image-1': {
                'max_prompts': 4,  # GPT Image supports multiple images
                'sizes': ['1024x1024', '1024x1792', '1792x1024', '512x512'],
                'quality': ['standard', 'enhanced'],
                'format': ['png', 'webp', 'jpeg'],
                'cost_per_image': 0.035,  # Different pricing from DALL-E 3
                'supports_batch': True,
                'max_batch_size': 4
            }
        }
        
        # Default generation parameters (different structure)
        self.default_params = {
            'size': '1024x1024',
            'quality': 'standard',
            'format': 'png',
            'response_format': 'b64_json',
            'n': 1,
            'guidance_scale': 7.5,  # GPT Image specific parameter
            'inference_steps': 30   # GPT Image specific parameter
        }
        
        # Different endpoint for GPT Image
        self.endpoint = 'https://api.openai.com/v1/images/gpt-generate'
    
    async def complete(self, prompt: str, model: str) -> str:
        """Generate image using GPT Image 1 and return image data"""
        return await self.generate_image(prompt, model)
    
    async def generate_image(
        self, 
        prompt: str, 
        model: str,
        size: str = None,
        quality: str = None,
        format: str = None,
        response_format: str = None,
        n: int = 1,
        guidance_scale: float = None,
        inference_steps: int = None,
        **kwargs
    ) -> str:
        """
        Generate image using OpenAI GPT Image 1
        
        Args:
            prompt: The image generation prompt
            model: Model alias (gpt-image-1) or direct model name
            size: Image size ('1024x1024', '1024x1792', '1792x1024', '512x512')
            quality: Image quality ('standard' or 'enhanced')
            format: Output format ('png', 'webp', 'jpeg')
            response_format: Response format ('url' or 'b64_json')
            n: Number of images (1-4)
            guidance_scale: Guidance scale (1.0-20.0, default 7.5)
            inference_steps: Inference steps (10-50, default 30)
            
        Returns:
            JSON string with image data and metadata
        """
        # Resolve model alias to actual model name
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        await self._apply_rate_limit(1)  # 1 request
        
        # Get model settings
        settings = self.model_settings['gpt-image-1']
        
        # Build parameters (different structure than DALL-E)
        params = {
            'prompt': prompt,
            'model': actual_model,
            'response_format': response_format or self.default_params['response_format'],
            'n': min(n, settings['max_batch_size'])
        }
        
        # Validate and set size
        if size and size in settings['sizes']:
            params['size'] = size
        else:
            if size:
                logger.warning(f"Size {size} not supported for GPT Image 1, using default")
            params['size'] = self.default_params['size']
        
        # Validate and set quality
        if quality and quality in settings['quality']:
            params['quality'] = quality
        else:
            if quality:
                logger.warning(f"Quality {quality} not supported for GPT Image 1, using default")
            params['quality'] = self.default_params['quality']
            
        # Validate and set format
        if format and format in settings['format']:
            params['format'] = format
        else:
            if format:
                logger.warning(f"Format {format} not supported for GPT Image 1, using default")
            params['format'] = self.default_params['format']
        
        # GPT Image specific parameters
        if guidance_scale is not None:
            params['guidance_scale'] = max(1.0, min(20.0, guidance_scale))
        else:
            params['guidance_scale'] = self.default_params['guidance_scale']
            
        if inference_steps is not None:
            params['inference_steps'] = max(10, min(50, inference_steps))
        else:
            params['inference_steps'] = self.default_params['inference_steps']
        
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
                    raise Exception(f"OpenAI GPT Image API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract image data (different response format)
                if 'generated_images' in data and data['generated_images']:
                    images = []
                    for img_data in data['generated_images']:
                        image_info = {
                            'seed': img_data.get('seed'),
                            'guidance_scale': img_data.get('guidance_scale'),
                            'inference_steps': img_data.get('inference_steps')
                        }
                        
                        if params['response_format'] == 'url':
                            image_info['url'] = img_data.get('image_url')
                        elif params['response_format'] == 'b64_json':
                            image_info['b64_json'] = img_data.get('image_data')
                        
                        images.append(image_info)
                    
                    # Return standardized format
                    return json.dumps({
                        'model': actual_model,
                        'prompt': prompt,
                        'parameters': params,
                        'images': images,
                        'count': len(images),
                        'provider': 'gpt_image',
                        'generation_info': {
                            'guidance_scale': params['guidance_scale'],
                            'inference_steps': params['inference_steps']
                        }
                    })
                else:
                    raise Exception("No generated_images in OpenAI GPT Image response")
                    
        except Exception as e:
            logger.error(f"GPT Image 1 generation error: {e}")
            raise Exception(f"GPT Image 1 error: {e}")
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        resolved = aliases.get(alias, alias)
        
        # Map aliases to GPT Image 1
        alias_mapping = {
            'gpt-image-1': 'gpt-image-1',
            'gpt-img-1': 'gpt-image-1'
        }
        
        if resolved in alias_mapping:
            resolved = alias_mapping[resolved]
        
        # Only GPT Image 1 is supported
        if resolved != 'gpt-image-1':
            logger.warning(f"Model {resolved} not supported, using gpt-image-1")
            return 'gpt-image-1'
        
        logger.debug(f"Resolved model alias {alias} -> {resolved}")
        return resolved
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for image generation (prompt tokens only)"""
        prompt_tokens = len(prompt) // 4
        logger.debug(f"Estimated {prompt_tokens} tokens for prompt")
        return prompt_tokens
    
    def estimate_cost(self, tokens: int, model: str, **kwargs) -> float:
        """Estimate cost for GPT Image 1 generation"""
        settings = self.model_settings['gpt-image-1']
        
        # Get number of images
        n = kwargs.get('n', 1)
        
        # GPT Image has flat pricing per image
        total_cost = settings['cost_per_image'] * n
        logger.debug(f"Estimated cost for GPT Image 1 ({n} images): ${total_cost:.4f}")
        
        return total_cost
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("GPT Image session closed")
            except Exception as e:
                logger.warning(f"Error closing GPT Image session: {e}")
        
        await super().close()
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return ['gpt-image-1']
    
    def validate_configuration(self) -> dict:
        """Validate provider configuration"""
        status = {
            'provider': 'gpt_image',
            'authentication': bool(self.api_key),
            'models_available': 1,
            'issues': []
        }
        
        if not self.api_key:
            status['issues'].append("OPENAI_API_KEY not set")
        
        return status