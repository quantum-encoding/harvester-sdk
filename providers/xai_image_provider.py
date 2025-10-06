"""
xAI Grok Image Generation Provider
Uses xAI SDK for image generation
"""
import os
import json
import base64
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class XaiImageProvider(BaseProvider):
    """Provider for xAI Grok image generation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.get('api_key') or os.environ.get('XAI_API_KEY')
        if not self.api_key:
            raise ValueError("XAI_API_KEY required for Grok image generation")

        # Import xAI SDK
        try:
            from xai_sdk import Client
            self.client = Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("xai_sdk not installed. Run: pip install xai-sdk")

    async def complete(self, prompt: str, model: str) -> str:
        """Generate image using xAI Grok"""
        return await self.generate_image(prompt, model)

    async def generate_image(
        self,
        prompt: str,
        model: str,
        n: int = 1,
        response_format: str = "base64",
        **kwargs
    ) -> str:
        """
        Generate image using xAI Grok Image

        Args:
            prompt: Image generation prompt
            model: Model name (grok-2-image or grok-2-image-1212)
            n: Number of images (1-10, default 1)
            response_format: "url" or "base64" (default base64 for consistency)

        Returns:
            JSON string with image data
        """
        # Apply rate limiting
        await self._apply_rate_limit(1)

        # Map image_format for SDK
        image_format = "base64" if response_format == "b64_json" else response_format

        try:
            if n == 1:
                # Single image
                response = self.client.image.sample(
                    model="grok-2-image",
                    prompt=prompt,
                    image_format=image_format
                )

                # Convert to standard format
                if image_format == "base64":
                    # response.image returns raw bytes
                    b64_data = base64.b64encode(response.image).decode('utf-8')
                    images = [{
                        'b64_json': b64_data,
                        'revised_prompt': response.prompt
                    }]
                else:
                    images = [{
                        'url': response.url,
                        'revised_prompt': response.prompt
                    }]
            else:
                # Multiple images
                responses = self.client.image.sample_batch(
                    model="grok-2-image",
                    prompt=prompt,
                    n=n,
                    image_format=image_format
                )

                images = []
                for resp in responses:
                    if image_format == "base64":
                        b64_data = base64.b64encode(resp.image).decode('utf-8')
                        images.append({
                            'b64_json': b64_data,
                            'revised_prompt': resp.prompt
                        })
                    else:
                        images.append({
                            'url': resp.url,
                            'revised_prompt': resp.prompt
                        })

            # Return in standard format
            return json.dumps({
                'model': 'grok-2-image',
                'prompt': prompt,
                'revised_prompt': images[0].get('revised_prompt', prompt) if images else prompt,
                'images': images,
                'count': len(images),
                'provider': 'xai_grok_image'
            })

        except Exception as e:
            logger.error(f"xAI Grok image generation error: {e}")
            raise Exception(f"xAI Grok image error: {e}")

    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for image generation"""
        # Rough estimation: 1 token per 4 characters for prompt
        prompt_tokens = len(prompt) // 4
        return prompt_tokens

    def estimate_cost(self, tokens: int, model: str, **kwargs) -> float:
        """Estimate cost for xAI Grok image generation"""
        # Get number of images
        n = kwargs.get('n', 1)
        # Cost per image (estimate - check xAI pricing)
        cost_per_image = 0.05  # Placeholder
        return cost_per_image * n

    def resolve_model_alias(self, alias: str) -> str:
        """Resolve model alias to actual model name"""
        # xAI only has one image model
        alias_mapping = {
            'grok-image': 'grok-2-image',
            'grok-img': 'grok-2-image',
            'grok-2-image-1212': 'grok-2-image',
        }
        return alias_mapping.get(alias, 'grok-2-image')

    async def close(self):
        """Clean up resources"""
        # xAI SDK client doesn't need explicit cleanup
        await super().close()
