"""
Google AI Studio (GenAI) Veo 3 Video Provider
Video generation with native audio using Google AI Studio API

Models available:
- veo-3.0-generate-preview (8-second 720p videos with audio)
- veo-2.0-generate (5-8 second 720p silent videos)
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os
import base64
from io import BytesIO

try:
    from google import genai
    from google.genai import types
    from PIL import Image
except ImportError:
    genai = None
    Image = None
    types = None
    
from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class GenAIVeo3Provider(BaseProvider):
    """
    Google AI Studio Veo 3 Video Provider
    
    This provider uses Google's AI Studio API for Veo video models,
    featuring state-of-the-art video generation with native audio support.
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
            "veo-3.0-generate-preview": {
                "name": "Veo 3 Preview",
                "duration": 8,  # seconds
                "resolution": "720p",
                "fps": 24,
                "has_audio": True,
                "aspect_ratios": ["16:9"],
                "max_text_tokens": 1024,
                "latency_min": 11,  # seconds
                "latency_max": 360,  # 6 minutes during peak
                "retention_days": 2
            },
            "veo-3.0-fast-generate-preview": {
                "name": "Veo 3 Fast Preview",
                "duration": 8,
                "resolution": "720p", 
                "fps": 24,
                "has_audio": True,
                "aspect_ratios": ["16:9"],
                "max_text_tokens": 1024,
                "latency_min": 8,
                "latency_max": 180,
                "retention_days": 2
            },
            "veo-2.0-generate-001": {
                "name": "Veo 2 Stable",
                "duration": "5-8",  # variable
                "resolution": "720p",
                "fps": 24,
                "has_audio": False,  # Silent only
                "aspect_ratios": ["16:9", "9:16"],
                "max_text_tokens": 1024,
                "max_image_size_mb": 20,  # Supports images up to 20MB
                "max_videos": 2,  # Can generate up to 2 videos
                "latency_min": 10,
                "latency_max": 240,
                "retention_days": 2
            }
        }
        
        self.default_model = config.get("default_model", "veo-3.0-generate-preview")
        
        logger.info(f"GenAI Veo 3 provider initialized with {len(self.model_configs)} models")
    
    async def generate_video(
        self,
        prompt: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Google AI Studio Veo
        
        Args:
            prompt: Text prompt for video generation (supports audio cues in Veo 3)
            model: Model to use (veo-3.0-generate-preview or veo-2.0-generate)
            **kwargs: Additional parameters:
                - negative_prompt: What not to include in the video
                - image: Initial image to animate (PIL Image or base64)
                - aspect_ratio: "16:9" or "9:16" (Veo 2 only)
                - person_generation: "allow_all", "allow_adult", "dont_allow"
                - output_path: Where to save the video file
                - poll_interval: Seconds between status checks (default 10)
        
        Returns:
            Dict with video file path and metadata
        """
        model = model or self.default_model
        
        if model not in self.model_configs:
            logger.warning(f"Unknown model {model}, using default")
            model = self.default_model
        
        config_info = self.model_configs[model]
        
        # Build generation config
        config_params = {}
        
        if "negative_prompt" in kwargs:
            config_params["negative_prompt"] = kwargs["negative_prompt"]
        
        if "aspect_ratio" in kwargs:
            # Validate aspect ratio for model
            if kwargs["aspect_ratio"] in config_info["aspect_ratios"]:
                config_params["aspect_ratio"] = kwargs["aspect_ratio"]
            else:
                logger.warning(f"Aspect ratio {kwargs['aspect_ratio']} not supported by {model}")
        
        if "person_generation" in kwargs:
            config_params["person_generation"] = kwargs["person_generation"]
        
        try:
            # Handle optional image input
            generation_kwargs = {
                "model": model,
                "prompt": prompt
            }
            
            if config_params:
                generation_kwargs["config"] = types.GenerateVideosConfig(**config_params)
            
            # Add image if provided
            if "image" in kwargs:
                image = kwargs["image"]
                
                # Convert to PIL Image if needed
                if isinstance(image, str):
                    if Path(image).exists():
                        # File path
                        image = Image.open(image)
                    else:
                        # Assume base64
                        image_bytes = base64.b64decode(image)
                        image = Image.open(BytesIO(image_bytes))
                
                generation_kwargs["image"] = image
            
            # Start video generation (returns operation)
            logger.info(f"Starting video generation with {model}")
            logger.info(f"Expected latency: {config_info['latency_min']}-{config_info['latency_max']} seconds")
            
            operation = await asyncio.to_thread(
                self.client.models.generate_videos,
                **generation_kwargs
            )
            
            # Poll for completion
            poll_interval = kwargs.get("poll_interval", 10)
            start_time = time.time()
            
            while not operation.done:
                elapsed = int(time.time() - start_time)
                logger.info(f"Video generation in progress... ({elapsed}s elapsed)")
                
                await asyncio.sleep(poll_interval)
                operation = await asyncio.to_thread(
                    self.client.operations.get,
                    operation
                )
            
            # Get the generated video
            generated_video = operation.response.generated_videos[0]
            
            # Save video file
            output_path = kwargs.get("output_path", f"veo_{model.split('-')[1]}_{int(time.time())}.mp4")
            
            logger.info(f"Downloading video to {output_path}")
            await asyncio.to_thread(
                self.client.files.download,
                file=generated_video.video
            )
            
            # Save to file
            generated_video.video.save(output_path)
            
            total_time = int(time.time() - start_time)
            logger.info(f"Video generated successfully in {total_time} seconds")
            
            return {
                "success": True,
                "video_path": output_path,
                "model": model,
                "prompt": prompt,
                "config": config_params,
                "duration": config_info["duration"],
                "resolution": config_info["resolution"],
                "has_audio": config_info["has_audio"],
                "generation_time": total_time,
                "retention_note": f"Video stored on server for {config_info['retention_days']} days"
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
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
        Generate multiple videos from a list of prompts
        
        Note: Videos are generated sequentially to avoid overwhelming the API
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing video {i+1}/{len(prompts)}")
            
            # Add index to output filename
            if "output_path" in kwargs:
                base_path = Path(kwargs["output_path"])
                kwargs["output_path"] = str(base_path.parent / f"{base_path.stem}_{i+1}{base_path.suffix}")
            
            result = await self.generate_video(prompt, model, **kwargs)
            results.append(result)
            
            # Add delay between generations
            if i < len(prompts) - 1 and result["success"]:
                logger.info("Waiting before next video generation...")
                await asyncio.sleep(5)
        
        return results
    
    async def generate_with_audio_cues(
        self,
        visual_prompt: str,
        dialogue: Optional[List[Dict[str, str]]] = None,
        sound_effects: Optional[List[str]] = None,
        ambient_noise: Optional[str] = None,
        model: str = "veo-3.0-generate-preview",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video with specific audio cues (Veo 3 only)
        
        Args:
            visual_prompt: Visual description of the scene
            dialogue: List of dialogue entries [{"speaker": "name", "text": "dialogue"}]
            sound_effects: List of sound effect descriptions
            ambient_noise: Description of background sounds
            model: Must be Veo 3 model for audio support
            **kwargs: Additional generation parameters
        
        Returns:
            Dict with video file path and metadata
        """
        # Check if model supports audio
        if not self.model_configs.get(model, {}).get("has_audio", False):
            logger.warning(f"Model {model} doesn't support audio. Use veo-3.0-generate-preview")
            model = "veo-3.0-generate-preview"
        
        # Build comprehensive prompt with audio cues
        prompt_parts = [visual_prompt]
        
        # Add dialogue with quotes
        if dialogue:
            for entry in dialogue:
                speaker = entry.get("speaker", "")
                text = entry["text"]
                if speaker:
                    prompt_parts.append(f'{speaker}: "{text}"')
                else:
                    prompt_parts.append(f'"{text}"')
        
        # Add sound effects
        if sound_effects:
            for sfx in sound_effects:
                prompt_parts.append(f"SFX: {sfx}")
        
        # Add ambient noise
        if ambient_noise:
            prompt_parts.append(f"Ambient: {ambient_noise}")
        
        # Combine all parts
        full_prompt = " ".join(prompt_parts)
        
        logger.info(f"Generating video with audio cues: {len(dialogue or [])} dialogue, "
                   f"{len(sound_effects or [])} SFX, ambient: {bool(ambient_noise)}")
        
        return await self.generate_video(full_prompt, model, **kwargs)
    
    async def animate_image(
        self,
        image_path: str,
        prompt: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Animate a static image to create a video
        
        Args:
            image_path: Path to the image file or base64 string
            prompt: Description of how to animate the image
            model: Model to use for generation
            **kwargs: Additional generation parameters
        
        Returns:
            Dict with video file path and metadata
        """
        kwargs["image"] = image_path
        return await self.generate_video(prompt, model, **kwargs)
    
    async def generate_from_imagen(
        self,
        prompt: str,
        imagen_model: str = "imagen-3.0-generate-002",
        veo_model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an image with Imagen, then animate it with Veo
        
        Args:
            prompt: Prompt for both image and video generation
            imagen_model: Imagen model for initial image
            veo_model: Veo model for video generation
            **kwargs: Additional generation parameters
        
        Returns:
            Dict with video file path and metadata
        """
        try:
            # Step 1: Generate image with Imagen
            logger.info(f"Generating initial image with {imagen_model}")
            
            imagen_response = await asyncio.to_thread(
                self.client.models.generate_images,
                model=imagen_model,
                prompt=prompt
            )
            
            if not imagen_response.generated_images:
                raise Exception("Failed to generate initial image")
            
            generated_image = imagen_response.generated_images[0].image
            
            # Step 2: Generate video from image
            logger.info("Animating generated image with Veo")
            
            kwargs["image"] = generated_image
            return await self.generate_video(prompt, veo_model, **kwargs)
            
        except Exception as e:
            logger.error(f"Image-to-video generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }
    
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
    
    def estimate_cost(self, model: str, num_videos: int = 1) -> float:
        """
        Estimate cost for video generation
        
        Note: Check Google AI Studio pricing for current rates
        """
        # Placeholder pricing - update with actual rates
        cost_per_video = {
            "veo-3.0-generate-preview": 0.10,  # Veo 3 with audio
            "veo-3.0-fast-generate-preview": 0.08,  # Veo 3 Fast
            "veo-2.0-generate": 0.05  # Veo 2 (no audio)
        }
        return cost_per_video.get(model, 0.10) * num_videos
    
    async def complete(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Compatibility method for BaseProvider interface
        Generates a video and returns the file path
        """
        result = await self.generate_video(prompt, model, **kwargs)
        
        if result["success"]:
            return result["video_path"]
        else:
            raise Exception(f"Video generation failed: {result.get('error', 'Unknown error')}")


# Example usage
if __name__ == "__main__":
    async def test_veo3():
        config = {
            "api_key": "YOUR_API_KEY"  # Or set GOOGLE_GENAI_API_KEY env var
        }
        
        provider = GenAIVeo3Provider(config)
        
        # Test 1: Basic video generation
        result = await provider.generate_video(
            prompt="A majestic eagle soaring through mountain peaks at sunset",
            model="veo-3.0-generate-preview",
            output_path="eagle_sunset.mp4"
        )
        
        if result["success"]:
            print(f"✅ Generated video: {result['video_path']}")
            print(f"   Duration: {result['duration']} seconds")
            print(f"   Has audio: {result['has_audio']}")
        
        # Test 2: Video with audio cues
        result2 = await provider.generate_with_audio_cues(
            visual_prompt="Two hikers discovering a hidden waterfall",
            dialogue=[
                {"speaker": "Hiker 1", "text": "Look at that! It's incredible!"},
                {"speaker": "Hiker 2", "text": "I've never seen anything like it"}
            ],
            sound_effects=["waterfall roaring", "birds chirping"],
            ambient_noise="Forest sounds, gentle breeze",
            output_path="hikers_waterfall.mp4"
        )
        
        if result2["success"]:
            print(f"✅ Generated video with audio: {result2['video_path']}")
        
        # Test 3: Animate an image
        if Path("test_image.jpg").exists():
            result3 = await provider.animate_image(
                image_path="test_image.jpg",
                prompt="Camera slowly zooms in while the subject turns their head",
                output_path="animated_portrait.mp4"
            )
            
            if result3["success"]:
                print(f"✅ Animated image: {result3['video_path']}")
    
    # Run test
    import asyncio
    asyncio.run(test_veo3())