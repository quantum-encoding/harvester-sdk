"""
Google Gemini Experimental Image Generation Provider
Supports Gemini 2.0 Flash Preview - generates BOTH text and images
"""
import asyncio
import aiohttp
import subprocess
import os
import json
import base64
import time
from typing import Dict, Any, Optional, List
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class GeminiExpProvider(BaseProvider):
    """Provider for Gemini 2.0 Flash Preview with image generation capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Smart authentication: env vars → config → error
        self.project_id = (
            os.environ.get('GOOGLE_CLOUD_PROJECT') or 
            os.environ.get('GCP_PROJECT') or 
            config.get('project_id', 'quantum-encoding-web-prod')
        )
        
        self.location = config.get('location', 'global')
        
        # Build Vertex AI endpoint
        self.vertex_endpoint = f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models"
        
        # Model configuration for Gemini 2.0 Flash Preview Image Generation
        self.model_settings = {
            'gemini-2.0-flash-exp-image': {
                'model_id': 'gemini-2.0-flash-preview-image-generation',
                'max_tokens': 8192,
                'temperature': 1.0,
                'cost_per_million_input': 0.50,  # Estimated
                'cost_per_million_output': 2.00,  # Estimated
                'context_window': 32768,
                'supports_image_generation': True,
                'response_modalities': ['TEXT', 'IMAGE']
            }
        }
        
        # Default generation config
        self.default_config = {
            'temperature': 1.0,
            'top_p': 0.95,
            'max_output_tokens': 8192,
            'response_modalities': ['TEXT', 'IMAGE'],
            'safety_settings': [
                {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_IMAGE_HATE', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_IMAGE_HARASSMENT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT', 'threshold': 'OFF'}
            ]
        }
        
        # Authentication method
        self.auth_method = config.get('authentication_method', 'gcloud_cli')
        self.api_key = os.environ.get('GEMINI_API_KEY') or config.get('api_key')
        
        # Cache for access token
        self.access_token = None
        self.token_expiry = None
        
    def _get_access_token(self) -> str:
        """Get Google Cloud access token using gcloud CLI"""
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            token = result.stdout.strip()
            
            # Cache token for 50 minutes
            self.access_token = token
            self.token_expiry = time.time() + 3000
            
            return token
        except subprocess.TimeoutExpired:
            raise Exception("gcloud auth timeout - ensure you're logged in with 'gcloud auth login'")
        except subprocess.CalledProcessError as e:
            raise Exception(f"gcloud auth failed: {e.stderr}")
        except FileNotFoundError:
            raise Exception("gcloud CLI not found - please install Google Cloud SDK")
    
    async def get_access_token(self) -> str:
        """Get or refresh access token"""
        if self.access_token and self.token_expiry and time.time() < self.token_expiry:
            return self.access_token
        
        # Run token refresh in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_access_token)
    
    async def complete(self, prompt: str, model: str = 'gemini-2.0-flash-exp-image') -> str:
        """Generate text and/or images using Gemini 2.0 Flash Preview"""
        return await self.generate_multimodal(prompt, model)
    
    async def generate_multimodal(
        self,
        prompt: str,
        model: str = 'gemini-2.0-flash-exp-image',
        response_modalities: List[str] = None,
        temperature: float = None,
        max_output_tokens: int = None,
        top_p: float = None,
        **kwargs
    ) -> str:
        """
        Generate multimodal content (text + images) using Gemini 2.0 Flash Preview
        
        Args:
            prompt: The generation prompt
            model: Model alias (always maps to gemini-2.0-flash-preview-image-generation)
            response_modalities: List of ['TEXT', 'IMAGE'] or subset
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
            top_p: Top-p sampling parameter
            
        Returns:
            JSON string with text and/or image data
        """
        # Resolve model settings
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings['gemini-2.0-flash-exp-image'])
        
        # Apply rate limiting
        await self._apply_rate_limit(1)
        
        # Build generation config
        generation_config = {
            'temperature': temperature or self.default_config['temperature'],
            'maxOutputTokens': max_output_tokens or self.default_config['max_output_tokens'],
            'topP': top_p or self.default_config['top_p'],
            'responseModalities': response_modalities or self.default_config['response_modalities']
        }
        
        # Build request payload
        payload = {
            'contents': [
                {
                    'role': 'user',
                    'parts': [
                        {'text': prompt}
                    ]
                }
            ],
            'generationConfig': generation_config,
            'safetySettings': self.default_config['safety_settings']
        }
        
        logger.debug(f"Gemini Exp request: {json.dumps(payload, indent=2)}")
        
        # Choose endpoint based on auth method
        if self.auth_method == 'api_key' and self.api_key:
            # Use Google AI API with key
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings['model_id']}:generateContent"
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': self.api_key
            }
        else:
            # Use Vertex AI with gcloud auth
            access_token = await self.get_access_token()
            url = f"{self.vertex_endpoint}/{settings['model_id']}:streamGenerateContent"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini Exp API error: {response.status} - {error_text}")
                
                # Handle streaming response
                full_text = []
                images = []
                
                async for line in response.content:
                    if line:
                        try:
                            # Parse streaming JSON response
                            data = json.loads(line.decode('utf-8').strip())
                            
                            if 'candidates' in data:
                                for candidate in data['candidates']:
                                    if 'content' in candidate:
                                        for part in candidate['content'].get('parts', []):
                                            if 'text' in part:
                                                full_text.append(part['text'])
                                            elif 'inlineData' in part:
                                                # Image data
                                                image_data = part['inlineData']
                                                images.append({
                                                    'mimeType': image_data.get('mimeType', 'image/png'),
                                                    'data': image_data.get('data')  # base64 encoded
                                                })
                        except json.JSONDecodeError:
                            # May be partial JSON in streaming
                            continue
                
                # Return combined response
                result = {
                    'model': settings['model_id'],
                    'prompt': prompt,
                    'text': ''.join(full_text) if full_text else None,
                    'images': images if images else None,
                    'provider': 'gemini_exp',
                    'modalities_generated': []
                }
                
                if result['text']:
                    result['modalities_generated'].append('TEXT')
                if result['images']:
                    result['modalities_generated'].append('IMAGE')
                
                return json.dumps(result)
                
        except Exception as e:
            logger.error(f"Gemini Exp generation error: {e}")
            raise Exception(f"Gemini Exp error: {e}")
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        resolved = aliases.get(alias, alias)
        
        # Map all aliases to the experimental model
        alias_mapping = {
            'gemini-exp': 'gemini-2.0-flash-exp-image',
            'gemini-exp-image': 'gemini-2.0-flash-exp-image',
            'gemini-2-exp': 'gemini-2.0-flash-exp-image',
            'gemini-image': 'gemini-2.0-flash-exp-image',
            'gemini-2.0-flash-exp': 'gemini-2.0-flash-exp-image'
        }
        
        if resolved in alias_mapping:
            resolved = alias_mapping[resolved]
        
        # Always default to the experimental model
        if resolved not in self.model_settings:
            logger.info(f"Model {resolved} mapped to gemini-2.0-flash-exp-image")
            return 'gemini-2.0-flash-exp-image'
        
        return resolved
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for multimodal generation"""
        total_chars = len(prompt) + len(response)
        estimated_tokens = int(total_chars / 3.5)
        logger.debug(f"Estimated {estimated_tokens} tokens")
        return estimated_tokens
    
    def estimate_cost(self, tokens: int, model: str, **kwargs) -> float:
        """Estimate cost for multimodal generation"""
        settings = self.model_settings['gemini-2.0-flash-exp-image']
        
        # For cost estimation, assume 70% input, 30% output
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        input_cost = (input_tokens / 1_000_000) * settings['cost_per_million_input']
        output_cost = (output_tokens / 1_000_000) * settings['cost_per_million_output']
        
        # Add premium for image generation if requested
        if 'IMAGE' in kwargs.get('response_modalities', ['TEXT', 'IMAGE']):
            output_cost *= 1.5  # 50% premium for image generation
        
        total_cost = input_cost + output_cost
        logger.debug(f"Estimated cost: ${total_cost:.6f}")
        
        return total_cost
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("Gemini Exp session closed")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
        
        await super().close()
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return ['gemini-2.0-flash-exp-image']
    
    def validate_configuration(self) -> dict:
        """Validate provider configuration"""
        status = {
            'provider': 'gemini_exp',
            'authentication': False,
            'models_available': 1,
            'issues': []
        }
        
        # Check authentication
        if self.api_key:
            status['authentication'] = True
            status['auth_method'] = 'api_key'
        else:
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
                    status['auth_method'] = 'gcloud_cli'
            except:
                status['issues'].append("No authentication configured (set GEMINI_API_KEY or use gcloud auth login)")
        
        return status