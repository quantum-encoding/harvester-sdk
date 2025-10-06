"""
Google Vertex AI Imagen 4 image generation provider implementation
Updated for latest API (January 2025)
"""
import asyncio
import aiohttp
import subprocess
import time
import json
import base64
import os
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class VertexImageProvider(BaseProvider):
    """Provider for Google Vertex AI Imagen 4 image generation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Smart authentication: env vars → config file → error
        self.project_id = (
            os.environ.get('GOOGLE_CLOUD_PROJECT') or 
            os.environ.get('GCP_PROJECT') or 
            config.get('project_id')
        )
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID required. Set environment variable:\n"
                "  export GOOGLE_CLOUD_PROJECT=your-project-id\n"
                "Or add to config:\n"
                "  project_id: your-project-id"
            )
        
        self.location = config.get('location', 'us-central1')
        
        # Build endpoint
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models"
        
        # Model-specific settings (Imagen 4 only - 3.0 deprecated)
        self.model_settings = {
            'imagen-4.0-generate-001': {
                'max_images': 4,
                'aspect_ratios': ['1:1', '9:16', '16:9', '3:4', '4:3'],
                'quality_options': ['1K', '2K'],
                'supports_enhance_prompt': True,
                'supports_negative_prompt': False,
                'default_watermark': True,
                'cost_per_image': 0.060
            },
            'imagen-4.0-ultra-generate-001': {
                'max_images': 4,
                'aspect_ratios': ['1:1', '9:16', '16:9', '3:4', '4:3'],
                'quality_options': ['1K', '2K'],
                'supports_enhance_prompt': True,
                'supports_negative_prompt': False,
                'default_watermark': True,
                'ultra_model': True,
                'cost_per_image': 0.080
            },
            'imagen-4.0-fast-generate-001': {
                'max_images': 4,
                'aspect_ratios': ['1:1', '9:16', '16:9', '3:4', '4:3'],
                'quality_options': ['1K', '2K'],
                'supports_enhance_prompt': True,
                'supports_negative_prompt': False,
                'default_watermark': True,
                'fast_model': True,
                'cost_per_image': 0.030
            }
        }
        
        # Default generation parameters (new API format)
        self.default_params = {
            'aspectRatio': '16:9',  # Default for blog images
            'sampleCount': 1,
            'sampleImageSize': '1K',
            'addWatermark': False,  # No watermark
            'safetySetting': 'block_only_high',  # Less restrictive
            'personGeneration': 'allow_all',
            'enhancePrompt': True,
            'language': 'en',
            'outputOptions': {
                'mimeType': 'image/png',
                'compressionQuality': 95
            }
        }
        
        # Use gcloud auth (proven working approach from json_processor)
        self.access_token = None
        self.token_expiry = None
    
    def _get_access_token(self) -> str:
        """Get Google Cloud access token using gcloud CLI (working approach from json_processor)"""
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            token = result.stdout.strip()
            
            # Cache token for 50 minutes (tokens expire in 60 minutes)
            self.access_token = token
            self.token_expiry = time.time() + 3000
            
            return token
        except subprocess.TimeoutExpired:
            raise Exception("gcloud auth timeout - ensure you're logged in with 'gcloud auth login'")
        except subprocess.CalledProcessError as e:
            raise Exception(f"gcloud auth failed: {e.stderr}")
        except FileNotFoundError:
            raise Exception("gcloud CLI not found - please install Google Cloud SDK")
        except Exception as e:
            raise Exception(f"Failed to authenticate with Google Cloud: {e}")
    
    async def get_access_token(self) -> str:
        """Get or refresh access token"""
        if self.access_token and self.token_expiry and time.time() < self.token_expiry:
            return self.access_token
        
        # Run token refresh in thread pool since gcloud is synchronous
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_access_token)
    
    async def complete(self, prompt: str, model: str) -> str:
        """Generate image using Vertex AI Imagen and return image data"""
        # This is actually generate_image for image providers
        return await self.generate_image(prompt, model)
    
    async def generate_image(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = None,
        sample_count: int = None,
        sample_image_size: str = None,
        add_watermark: bool = None,
        safety_setting: str = None,
        person_generation: str = None,
        enhance_prompt: bool = None,
        negative_prompt: str = None,
        seed: int = None,
        language: str = None,
        output_format: str = None,
        **kwargs
    ) -> str:
        """
        Generate image using Vertex AI Imagen - Updated for stable models
        
        Args:
            prompt: The image generation prompt
            model: Model alias (goo-3-img, goo-4-img) or direct model name
            aspect_ratio: Image aspect ratio (1:1, 9:16, 16:9, 3:4, 4:3)
            sample_count: Number of images to generate (default: 1)
            sample_image_size: Image quality '1K' or '2K' (default: '1K')
            add_watermark: Whether to add watermark (default: False)
            safety_setting: Safety setting (block_only_high, block_some, block_few, block_most)
            person_generation: Person generation policy (dont_allow, allow_adult, allow_all)
            enhance_prompt: Whether to enhance prompt (default: True)
            negative_prompt: What to avoid in the image
            seed: Random seed for reproducible results
            language: Language code (default: 'en')
            output_format: Output format 'webp', 'png', 'jpg' (default: 'webp')
            
        Returns:
            JSON string with image data and metadata
        """
        # Resolve model alias to actual model name
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        await self._apply_rate_limit(1)  # 1 request
        
        # Get access token
        access_token = await self.get_access_token()
        
        # Build parameters based on the new Imagen API specification
        settings = self.model_settings.get(actual_model, self.model_settings['imagen-4.0-generate-001'])
        
        # Build parameters for the new API structure
        params = {
            'prompt': prompt,
            'aspect_ratio': aspect_ratio or self.default_params['aspect_ratio'],
            'sample_count': sample_count or self.default_params['sample_count'],
            'sample_image_size': sample_image_size or self.default_params['sample_image_size'],
            'add_watermark': add_watermark if add_watermark is not None else self.default_params['add_watermark'],
            'safety_setting': safety_setting or self.default_params['safety_setting'],
            'person_generation': person_generation or self.default_params['person_generation'],
            'enhance_prompt': enhance_prompt if enhance_prompt is not None else self.default_params['enhance_prompt'],
            'language': language or self.default_params['language']
        }
        
        # Add optional parameters
        if negative_prompt:
            params['negative_prompt'] = negative_prompt
        if seed is not None:
            params['seed'] = seed
        
        # Validate parameters against model capabilities
        if params['aspect_ratio'] not in settings['aspect_ratios']:
            logger.warning(f"Aspect ratio {params['aspect_ratio']} not supported by {actual_model}, using default")
            params['aspect_ratio'] = '16:9'
        
        # All Imagen 4 models support the same safety and person generation settings
        if params['safety_setting'] not in ['block_only_high', 'block_some', 'block_few', 'block_most']:
            logger.warning(f"Safety setting {params['safety_setting']} not supported, using default")
            params['safety_setting'] = 'block_only_high'
        
        if params['person_generation'] not in ['dont_allow', 'allow_adult', 'allow_all']:
            logger.warning(f"Person generation {params['person_generation']} not supported, using default")
            params['person_generation'] = 'allow_all'
        
        if params['sample_image_size'] not in settings.get('quality_options', ['1K']):
            logger.warning(f"Sample image size {params['sample_image_size']} not supported by {actual_model}, using 1K")
            params['sample_image_size'] = '1K'
        
        # Build request payload in new Imagen API format
        instances = [params]  # Use the params directly as the instance
        
        payload = {
            "instances": instances
        }
        
        logger.debug(f"Vertex AI request payload: {json.dumps(payload, indent=2)}")
        
        # Make API request
        url = f"{self.base_url}/{actual_model}:predict"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Vertex AI error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract image data from new Imagen API response format
                if 'predictions' in data and data['predictions']:
                    images = []
                    enhanced_prompt = prompt  # Default to original
                    
                    for prediction in data['predictions']:
                        # Handle both response formats for stability
                        image_data = {}
                        
                        # New API format - images array
                        if 'images' in prediction:
                            for img in prediction['images']:
                                if 'bytesBase64Encoded' in img:
                                    image_data = {
                                        'b64_json': img['bytesBase64Encoded'],
                                        'safety_rating': img.get('safetyRating', 'unknown'),
                                        'enhanced_prompt': img.get('enhanced_prompt', prompt)
                                    }
                                    if 'enhanced_prompt' in img:
                                        enhanced_prompt = img['enhanced_prompt']
                                    images.append(image_data)
                        
                        # Legacy format fallback
                        elif 'bytesBase64Encoded' in prediction:
                            image_data = {
                                'b64_json': prediction['bytesBase64Encoded'],
                                'safety_rating': prediction.get('safetyRating', 'unknown'),
                                'enhanced_prompt': prediction.get('enhanced_prompt', prompt)
                            }
                            if 'enhanced_prompt' in prediction:
                                enhanced_prompt = prediction['enhanced_prompt']
                            images.append(image_data)
                        elif 'generated_image' in prediction:
                            image_data = {
                                'b64_json': prediction['generated_image'],
                                'safety_rating': prediction.get('safetyRating', 'unknown'),
                                'enhanced_prompt': prediction.get('enhanced_prompt', prompt)
                            }
                            images.append(image_data)
                    
                    # Return as JSON string with enhanced prompt info
                    return json.dumps({
                        'model': actual_model,
                        'prompt': prompt,
                        'enhanced_prompt': enhanced_prompt,
                        'parameters': params,
                        'images': images,
                        'count': len(images),
                        'provider': 'vertex_ai_imagen'
                    })
                else:
                    raise Exception("No predictions in Vertex AI response")
                    
        except Exception as e:
            logger.error(f"Vertex AI image generation error: {e}")
            raise Exception(f"Vertex AI Imagen error: {e}")
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        resolved = aliases.get(alias, alias)
        
        # Map simplified aliases to Imagen 4 models only
        alias_mapping = {
            # Simplified aliases for CLI usage
            'imagen-4': 'imagen-4.0-generate-001',
            'imagen-4-ultra': 'imagen-4.0-ultra-generate-001', 
            'imagen-4-fast': 'imagen-4.0-fast-generate-001',
            'imagen': 'imagen-4.0-generate-001'  # Default to standard 4.0
        }
        
        if resolved in alias_mapping:
            resolved = alias_mapping[resolved]
        
        # Default to latest Imagen if unknown
        if resolved not in self.model_settings:
            logger.warning(f"Unknown model {resolved}, defaulting to imagen-4.0-generate-001")
            return 'imagen-4.0-generate-001'
        
        logger.debug(f"Resolved model alias {alias} -> {resolved}")
        return resolved
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count for image generation
        
        For image generation, we count the prompt tokens only
        """
        # Rough estimation: 1 token per 4 characters for prompt
        prompt_tokens = len(prompt) // 4
        logger.debug(f"Estimated {prompt_tokens} tokens for prompt")
        return prompt_tokens
    
    def estimate_cost(self, tokens: int, model: str, **kwargs) -> float:
        """Estimate cost for image generation"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings['imagen-4.0-generate-001'])
        
        # Number of images (default 1, check both 'n' and 'number_of_images')
        n = kwargs.get('n', kwargs.get('number_of_images', 1))
        
        # Simple cost calculation - cost per image
        total_cost = settings['cost_per_image'] * n
        logger.debug(f"Estimated cost for {n} images using {actual_model}: ${total_cost:.4f}")
        
        return total_cost
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("Vertex AI session closed")
            except Exception as e:
                logger.warning(f"Error closing Vertex AI session: {e}")
        
        await super().close()
    
    def get_available_models(self) -> list:
        """Get list of available image models"""
        return list(self.model_settings.keys())
    
    def validate_configuration(self) -> dict:
        """Validate provider configuration"""
        status = {
            'provider': 'vertex_image',
            'authentication': False,
            'models_available': len(self.model_settings),
            'issues': []
        }
        
        # Check if gcloud is available and authenticated
        try:
            # Test gcloud auth
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            if result.stdout.strip():
                status['authentication'] = True
            else:
                status['issues'].append("gcloud auth returned empty token")
        except subprocess.TimeoutExpired:
            status['issues'].append("gcloud auth timeout")
        except subprocess.CalledProcessError as e:
            status['issues'].append(f"gcloud auth failed: {e.stderr}")
        except FileNotFoundError:
            status['issues'].append("gcloud CLI not installed")
        except Exception as e:
            status['issues'].append(f"gcloud auth error: {e}")
        
        return status