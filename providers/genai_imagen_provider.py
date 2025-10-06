"""
Google AI Studio (GenAI) Imagen Provider
Alternative to Vertex AI - uses API key authentication

Models available:
- imagen-4.0-generate-001 (Standard)
- imagen-4.0-ultra-generate-001 (Ultra quality)
- imagen-4.0-fast-generate-001 (Fast generation)
"""

import logging
import base64
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path
import os

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    from io import BytesIO
except ImportError:
    genai = None
    Image = None
    
from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class GenAIImagenProvider(BaseProvider):
    """
    Google AI Studio Imagen Provider
    
    This provider uses Google's AI Studio API for Imagen models
    with simple API key authentication, as an alternative to
    Vertex AI which requires service account authentication.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        if genai is None:
            raise ImportError(
                "Google GenAI not installed. Install with: pip install google-generativeai"
            )
        
        # API key from environment or config
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Google GenAI API key required. Set GOOGLE_GENAI_API_KEY environment variable")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        # Model configurations
        self.model_configs = {
            "imagen-4.0-generate-001": {
                "name": "Standard Imagen 4.0",
                "max_images": 4,
                "sizes": ["1K", "2K"],
                "aspect_ratios": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                "cost_tier": "standard"
            },
            "imagen-4.0-ultra-generate-001": {
                "name": "Ultra Quality Imagen 4.0",
                "max_images": 4,
                "sizes": ["1K", "2K"],
                "aspect_ratios": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                "cost_tier": "premium"
            },
            "imagen-4.0-fast-generate-001": {
                "name": "Fast Imagen 4.0",
                "max_images": 4,
                "sizes": ["1K", "2K"],
                "aspect_ratios": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                "cost_tier": "economy"
            }
        }
        
        self.default_model = config.get("default_model", "imagen-4.0-generate-001")
        
        logger.info(f"GenAI Imagen provider initialized with {len(self.model_configs)} models")
    
    async def generate_image(
        self,
        prompt: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images using Google AI Studio Imagen
        
        Args:
            prompt: Text prompt for image generation
            model: Model to use (imagen-4.0-generate-001, ultra, or fast)
            **kwargs: Additional parameters:
                - number_of_images: 1-4 images
                - aspect_ratio: "1:1", "3:4", "4:3", "9:16", "16:9"
                - sample_image_size: "1K" or "2K"
                - person_generation: "dont_allow", "allow_adult", "allow_all"
                - negative_prompt: What to avoid in the image
        
        Returns:
            Dict with generated images in base64 format
        """
        model = model or self.default_model
        
        if model not in self.model_configs:
            logger.warning(f"Unknown model {model}, using default")
            model = self.default_model
        
        # Build generation config
        config_params = {
            "number_of_images": kwargs.get("number_of_images", 1),
        }
        
        # Add optional parameters
        if "aspect_ratio" in kwargs:
            config_params["aspect_ratio"] = kwargs["aspect_ratio"]
        
        if "sample_image_size" in kwargs:
            config_params["sample_image_size"] = kwargs["sample_image_size"]
        
        if "person_generation" in kwargs:
            config_params["person_generation"] = kwargs["person_generation"]
        
        try:
            # Generate images
            logger.info(f"Generating {config_params['number_of_images']} images with {model}")
            
            response = await asyncio.to_thread(
                self.client.models.generate_images,
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(**config_params)
            )
            
            # Convert images to base64
            images = []
            for generated_image in response.generated_images:
                # Convert PIL image to base64
                buffered = BytesIO()
                generated_image.image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images.append(img_base64)
            
            logger.info(f"Successfully generated {len(images)} images")
            
            return {
                "success": True,
                "images": images,
                "model": model,
                "prompt": prompt,
                "config": config_params
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "prompt": prompt
            }
    
    async def generate_batch(
        self,
        prompts: List[str],
        model: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple images from a list of prompts
        
        Note: GenAI doesn't have native batch support,
        so we process sequentially with rate limiting
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            result = await self.generate_image(prompt, model, **kwargs)
            results.append(result)
            
            # Rate limiting to avoid hitting quotas
            if i < len(prompts) - 1:
                await asyncio.sleep(1)  # 1 second between requests
        
        return results
    
    async def edit_image(
        self,
        image_path: str,
        prompt: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Edit an existing image (not directly supported by GenAI Imagen)
        
        GenAI Imagen doesn't support direct image editing.
        This method generates a new image based on the prompt.
        For true image editing, use Vertex AI or other providers.
        """
        logger.warning("GenAI Imagen doesn't support image editing. Generating new image from prompt.")
        return await self.generate_image(prompt, model, **kwargs)
    
    def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        model = model or self.default_model
        
        if model in self.model_configs:
            return self.model_configs[model]
        else:
            return {
                "error": f"Unknown model: {model}",
                "available_models": list(self.model_configs.keys())
            }
    
    def estimate_cost(self, model: str, num_images: int = 1) -> float:
        """
        Estimate cost for image generation
        
        Note: Pricing varies by model and usage tier.
        Check Google AI Studio pricing for current rates.
        """
        # Placeholder - check current pricing
        cost_per_image = {
            "imagen-4.0-generate-001": 0.020,  # Standard
            "imagen-4.0-ultra-generate-001": 0.040,  # Ultra
            "imagen-4.0-fast-generate-001": 0.010  # Fast
        }
        return cost_per_image.get(model, 0.020) * num_images
    
    async def complete(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Compatibility method for BaseProvider interface
        Generates an image and returns the base64 string
        """
        result = await self.generate_image(prompt, model, **kwargs)
        
        if result["success"] and result["images"]:
            return result["images"][0]  # Return first image as base64
        else:
            raise Exception(f"Image generation failed: {result.get('error', 'Unknown error')}")


# Example usage
if __name__ == "__main__":
    async def test_genai_imagen():
        config = {
            "api_key": "YOUR_API_KEY"  # Or set GOOGLE_GENAI_API_KEY env var
        }
        
        provider = GenAIImagenProvider(config)
        
        # Test standard generation
        result = await provider.generate_image(
            prompt="A beautiful sunset over mountains",
            model="imagen-4.0-generate-001",
            number_of_images=2,
            aspect_ratio="16:9"
        )
        
        if result["success"]:
            print(f"Generated {len(result['images'])} images")
            # Save first image
            with open("test_genai_imagen.png", "wb") as f:
                f.write(base64.b64decode(result["images"][0]))
        else:
            print(f"Generation failed: {result['error']}")
    
    # Run test
    import asyncio
    asyncio.run(test_genai_imagen())