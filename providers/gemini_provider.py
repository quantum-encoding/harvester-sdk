"""
Simple Gemini Provider using the new google-genai SDK
No Google Cloud credentials required - just an API key!

Copyright (c) 2025 Quantum Encoding Ltd.
"""
import os
import asyncio
import logging
from typing import Dict, Any, Optional

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Simple Gemini provider using google-genai SDK with API key authentication"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.get('api_key') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        # Initialize client (lazy loading)
        self._client = None
        
        # Model settings for Gemini models
        self.model_settings = {
            'gemini-2.5-pro': {
                'max_tokens': 64768,
                'temperature': 1.0,
                'context_window': 2097152  # 2M tokens
            },
            'gemini-2.5-flash': {
                'max_tokens': 65535,
                'temperature': 1.0,
                'context_window': 1048576  # 1M tokens
            },
            'gemini-2.5-flash-lite': {
                'max_tokens': 32768,
                'temperature': 1.0,
                'context_window': 524288  # 512K tokens
            }
        }
        
        logger.info(f"GeminiProvider initialized with API key: {'*' * 10}{self.api_key[-4:]}")
    
    def _get_client(self):
        """Lazy initialize the google-genai client"""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Google GenAI client initialized successfully")
            except ImportError:
                raise ImportError("google-genai not installed. Run: pip install google-genai")
        return self._client
    
    async def complete(self, prompt: str, model: str = "gemini-2.5-flash", **kwargs) -> str:
        """
        Generate text completion using Gemini
        
        Args:
            prompt: The prompt text
            model: Model name (default: gemini-2.5-flash)
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        try:
            client = self._get_client()
            
            # Get model settings
            settings = self.model_settings.get(model, self.model_settings['gemini-2.5-flash'])
            
            # Prepare generation config
            generation_config = {
                'temperature': kwargs.get('temperature', settings['temperature']),
                'max_output_tokens': kwargs.get('max_tokens', settings['max_tokens']),
                'top_p': kwargs.get('top_p', 0.95),
                'top_k': kwargs.get('top_k', 40)
            }
            
            # Generate content synchronously then wrap in async
            # (google-genai SDK doesn't have native async yet)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=generation_config
                )
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError(f"Unexpected response format from Gemini: {response}")
                
        except Exception as e:
            logger.error(f"Gemini completion error: {str(e)}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    def generate_text(self, prompt: str, model: str = "gemini-2.5-flash", **kwargs) -> str:
        """
        Synchronous text generation (for backwards compatibility)
        
        Args:
            prompt: The prompt text
            model: Model name
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        try:
            client = self._get_client()
            
            # Get model settings
            settings = self.model_settings.get(model, self.model_settings['gemini-2.5-flash'])
            
            # Prepare generation config
            generation_config = {
                'temperature': kwargs.get('temperature', settings['temperature']),
                'max_output_tokens': kwargs.get('max_tokens', settings['max_tokens']),
                'top_p': kwargs.get('top_p', 0.95),
                'top_k': kwargs.get('top_k', 40)
            }
            
            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError(f"Unexpected response format from Gemini: {response}")
                
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def close(self):
        """Cleanup resources"""
        self._client = None
        logger.info("GeminiProvider closed")
    
    def resolve_model_alias(self, alias: str) -> str:
        """Resolve model alias to actual model name"""
        # For Gemini, we just return the model name as-is
        return alias
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost for given token counts"""
        # Cost per million tokens (rough estimates)
        costs = {
            'gemini-2.5-pro': {'input': 1.25, 'output': 10.00},
            'gemini-2.5-flash': {'input': 0.30, 'output': 2.50},
            'gemini-2.5-flash-lite': {'input': 0.10, 'output': 0.80}
        }
        
        model_costs = costs.get(model, costs['gemini-2.5-flash'])
        
        input_cost = (prompt_tokens / 1_000_000) * model_costs['input']
        output_cost = (completion_tokens / 1_000_000) * model_costs['output']
        
        return input_cost + output_cost