"""
Google AI Studio (GenAI) provider implementation using API key authentication
Separate from Vertex AI - this uses the Google AI Studio API with simple API key auth
"""
import asyncio
import os
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class GenAIProvider(BaseProvider):
    """Provider for Google AI Studio (GenAI) models using API key authentication"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # API key authentication (simpler than Vertex AI)
        self.api_key = config.get('api_key') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            logger.warning("No Google AI Studio API key found. Set GEMINI_API_KEY or GOOGLE_AI_API_KEY")
        
        # Initialize client (lazy loading)
        self._client = None
        
        # Model-specific settings for Google AI Studio
        self.model_settings = {
            # Gemini 2.5 Pro (best for coding, reasoning, multimodal)
            'gemini-2.5-pro': {
                'max_tokens': 64768,
                'temperature': 1.0,
                'cost_per_million_input': 1.25,  # <=200K tokens
                'cost_per_million_input_long': 2.50,  # >200K tokens
                'cost_per_million_output': 10.00,  # <=200K tokens
                'cost_per_million_output_long': 15.00,  # >200K tokens
                'context_window': 2097152  # 2M tokens
            },
            # Gemini 2.5 Flash (large scale processing, agentic)
            'gemini-2.5-flash': {
                'max_tokens': 64535,
                'temperature': 1.0,
                'cost_per_million_input': 0.075,  # <=128K tokens
                'cost_per_million_input_long': 0.15,  # >128K tokens
                'cost_per_million_output': 0.30,  # <=128K tokens
                'cost_per_million_output_long': 0.60,  # >128K tokens
                'context_window': 1048576  # 1M tokens
            },
            # Gemini 2.5 Flash Lite (ultra fast, free tier available)
            'gemini-2.5-flash-lite': {
                'max_tokens': 8192,
                'temperature': 1.0,
                'cost_per_million_input': 0.10,
                'cost_per_million_output': 0.40,
                'context_window': 1048576  # 1M tokens
            },
            # Legacy models
            'gemini-1.5-pro': {
                'max_tokens': 8192,
                'temperature': 1.0,
                'cost_per_million_input': 3.50,
                'cost_per_million_output': 10.50,
                'context_window': 2097152
            },
            'gemini-1.5-flash': {
                'max_tokens': 8192,
                'temperature': 1.0,
                'cost_per_million_input': 0.075,
                'cost_per_million_output': 0.30,
                'context_window': 1048576
            },
        }
    
    def _get_client(self):
        """Lazy initialization of Google GenAI client with API key"""
        if self._client is None:
            try:
                from google import genai
                
                # Simple API key authentication - no Vertex AI
                if self.api_key:
                    os.environ['GOOGLE_AI_API_KEY'] = self.api_key
                
                self._client = genai.Client()  # No vertexai=True - pure GenAI
                logger.info("Google AI Studio (GenAI) client initialized with API key")
                
            except ImportError as e:
                logger.error("Google GenAI SDK not installed. Install with: pip install google-genai")
                raise ImportError(f"Missing google-genai dependency: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI client: {e}")
                raise
        
        return self._client
    
    async def complete(self, prompt: str, model: str, **kwargs) -> str:
        """Send completion request to Google AI Studio using GenAI SDK"""
        # Resolve model alias to actual model name
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        estimated_tokens = self.estimate_tokens(prompt, "")
        await self._apply_rate_limit(estimated_tokens)
        
        try:
            # Import types here to avoid import issues if SDK not installed
            from google.genai import types
            
            # Get client
            client = self._get_client()
            
            # Get model settings
            settings = self.model_settings.get(actual_model, self.model_settings['gemini-2.5-flash'])
            
            # Extract parameters from kwargs
            temperature = kwargs.get('temperature', settings.get('temperature', 1.0))
            max_tokens = kwargs.get('max_tokens', settings.get('max_tokens', 8192))
            
            # Create config
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Add system instruction if provided
            system_instruction = kwargs.get('system_instruction')
            if system_instruction:
                config.system_instruction = system_instruction
            
            # Generate content using GenAI API
            response = client.models.generate_content(
                model=actual_model,
                contents=prompt,
                config=config
            )
            
            # Extract text from response
            if response and hasattr(response, 'text') and response.text:
                result = response.text
                
                # Track token usage (if available)
                if hasattr(self, '_track_usage'):
                    self._track_usage(estimated_tokens, len(result.split()))
                
                logger.debug(f"GenAI completion successful: {len(result)} chars")
                return result
            else:
                logger.warning("GenAI returned empty response")
                return ""
                
        except Exception as e:
            logger.error(f"GenAI completion error: {e}")
            raise
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, {})
        
        return {
            'provider': 'genai',
            'model': actual_model,
            'alias': model,
            'authentication': 'api_key',
            'service': 'Google AI Studio',
            **settings
        }
    
    def list_available_models(self) -> list:
        """List all available models for this provider"""
        return list(self.model_settings.keys())
    
    def resolve_model_alias(self, alias: str) -> str:
        """Resolve model alias to actual model name"""
        # Check if it's already a direct model name
        if alias in self.model_settings:
            return alias
        
        # For GenAI provider, we don't use complex aliases - direct model names
        return alias
    
    def estimate_tokens(self, prompt: str, response: str = "") -> int:
        """Estimate token count for prompt and response"""
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(response, str):
            response = str(response)
        
        # Rough token estimation: 1 token ≈ 4 characters for text
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4 if response else 0
        
        return prompt_tokens + response_tokens
    
    def estimate_cost(self, prompt: str, response: str = "", model: str = None) -> float:
        """Estimate cost for prompt and response"""
        if not model:
            model = 'gemini-2.5-flash'
        
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings['gemini-2.5-flash'])
        
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4 if response else 0
        
        # Calculate cost based on token counts
        input_cost = (prompt_tokens / 1_000_000) * settings.get('cost_per_million_input', 0.075)
        output_cost = (response_tokens / 1_000_000) * settings.get('cost_per_million_output', 0.30)
        
        return input_cost + output_cost