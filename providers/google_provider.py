"""
Google Vertex AI provider implementation using service account authentication
This is separate from GenAI provider - uses Vertex AI with service accounts
"""
import asyncio
import os
from typing import Dict, Any, Optional
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class GoogleProvider(BaseProvider):
    """Provider for Google Vertex AI models using service account authentication"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Vertex AI configuration
        self.project_id = config.get('project_id', 'gen-lang-client-0685363971')
        self.location = config.get('location', 'global')
        
        # Set up authentication - prefer service account from config, fallback to environment
        credentials_path = config.get('api_key') or config.get('credentials_path')
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            logger.info(f"Using service account credentials from: {credentials_path}")
        elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            logger.info("Using service account credentials from environment variable")
        else:
            logger.warning("No Google service account credentials found")
        
        # Initialize client (lazy loading)
        self._client = None
        
        # Model-specific settings (Vertex AI Gemini models with vtx- prefix)
        self.model_settings = {
            # Vertex AI Gemini 2.5 Pro (best for coding, reasoning, multimodal)
            'vtx-gemini-2.5-pro': {
                'max_tokens': 64768,
                'temperature': 1.0,
                'cost_per_million_input': 1.25,  # <=200K tokens
                'cost_per_million_input_long': 2.50,  # >200K tokens
                'cost_per_million_output': 10.00,  # <=200K tokens
                'cost_per_million_output_long': 15.00,  # >200K tokens
                'context_window': 2097152  # 2M tokens
            },
            # Vertex AI Gemini 2.5 Flash (large scale processing, agentic)
            'vtx-gemini-2.5-flash': {
                'max_tokens': 64535,
                'temperature': 1.0,
                'cost_per_million_input': 0.30,
                'cost_per_million_output': 2.50,
                'context_window': 1048576  # 1M tokens
            },
            # Vertex AI Gemini 2.5 Flash-Lite (lowest cost, high volume)
            'vtx-gemini-2.5-flash-lite': {
                'max_tokens': 32768,
                'temperature': 1.0,
                'cost_per_million_input': 0.10,
                'cost_per_million_output': 0.40,
                'context_window': 1048576  # 1M tokens
            },
            # Additional Vertex AI models
            'vtx-gemini-1.5-pro': {
                'max_tokens': 8192,
                'temperature': 1.0,
                'cost_per_million_input': 3.50,
                'cost_per_million_output': 10.50,
                'context_window': 2097152
            },
            'vtx-gemini-1.5-flash': {
                'max_tokens': 8192,
                'temperature': 1.0,
                'cost_per_million_input': 0.075,
                'cost_per_million_output': 0.30,
                'context_window': 1048576
            },
        }
    
    def _get_client(self):
        """Lazy initialization of Google Vertex AI client"""
        if self._client is None:
            try:
                from google import genai
                
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                logger.info(f"Google Vertex AI client initialized for project: {self.project_id}")
                
            except ImportError as e:
                logger.error("Google GenAI SDK not installed. Install with: pip install google-genai")
                raise ImportError(f"Missing google-genai dependency: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI client: {e}")
                raise
        
        return self._client
    
    async def complete(self, prompt: str, model: str, **kwargs) -> str:
        """Send completion request to Google Vertex AI using GenAI SDK"""
        # Resolve model alias to actual model name
        actual_model = self.resolve_model_alias(model)
        
        # Strip vtx- prefix for actual API call (API expects just 'gemini-2.5-pro')
        api_model = actual_model.replace('vtx-', '') if actual_model.startswith('vtx-') else actual_model
        
        # Apply rate limiting
        estimated_tokens = self.estimate_tokens(prompt, "")
        await self._apply_rate_limit(estimated_tokens)
        
        try:
            # Import types here to avoid import issues if SDK not installed
            from google.genai import types
            
            # Get client
            client = self._get_client()
            
            # Get model settings
            settings = self.model_settings.get(actual_model, self.model_settings['vtx-gemini-2.5-flash'])
            
            # Prepare content
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt)
                    ]
                )
            ]
            
            # Extract parameters from kwargs
            temperature = kwargs.get('temperature', settings.get('temperature', 1.0))
            max_tokens = kwargs.get('max_tokens', settings.get('max_tokens', 8192))
            
            # Configure generation
            generate_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_tokens,
                # Disable safety filters for code processing
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ]
            )
            
            # Run the generation in a thread pool since the SDK is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_content_sync,
                client,
                api_model,
                contents,
                generate_config
            )
            
            return response
            
        except ImportError as e:
            logger.error("Google GenAI SDK not available")
            raise Exception(f"Google GenAI SDK required but not installed: {e}")
        except Exception as e:
            logger.error(f"Vertex AI completion error: {e}")
            raise Exception(f"Vertex AI API error: {e}")
    
    def _generate_content_sync(self, client, model, contents, config):
        """Synchronous wrapper for Vertex AI generation"""
        try:
            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            
            # Extract text from response
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Extract from candidates if direct text not available
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts
                    if parts and hasattr(parts[0], 'text'):
                        return parts[0].text
            
            raise Exception("No text content in response")
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        resolved = aliases.get(alias, alias)
        logger.debug(f"Resolved model alias {alias} -> {resolved}")
        return resolved
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count for Gemini models
        
        Gemini uses a different tokenization than OpenAI models.
        This is a rough approximation.
        """
        total_chars = len(prompt) + len(response)
        
        # Gemini tokenization is roughly 1 token per 3.5-4 characters for English text
        # For code, it's typically closer to 1 token per 3 characters
        # We'll use a conservative estimate
        estimated_tokens = int(total_chars / 3.5)
        
        logger.debug(f"Estimated {estimated_tokens} tokens for {total_chars} characters")
        return estimated_tokens
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage based on Gemini pricing"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(actual_model, self.model_settings['gemini-2.5-flash'])
        
        # For cost estimation, assume 70% input tokens, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        # Handle tiered pricing for Gemini 2.5 Pro
        if actual_model == 'gemini-2.5-pro' and input_tokens > 200000:
            # Use long context pricing for >200K tokens
            input_cost = (input_tokens / 1_000_000) * settings.get('cost_per_million_input_long', settings['cost_per_million_input'])
            output_cost = (output_tokens / 1_000_000) * settings.get('cost_per_million_output_long', settings['cost_per_million_output'])
        else:
            # Use standard pricing
            input_cost = (input_tokens / 1_000_000) * settings['cost_per_million_input']
            output_cost = (output_tokens / 1_000_000) * settings['cost_per_million_output']
        
        total_cost = input_cost + output_cost
        logger.debug(f"Estimated cost for {tokens} tokens: ${total_cost:.6f}")
        
        return total_cost
    
    async def close(self):
        """Clean up resources"""
        if self._client:
            try:
                # Google GenAI client doesn't require explicit cleanup
                self._client = None
                logger.debug("Google GenAI client cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up Google client: {e}")
        
        await super().close()
    
    def get_available_models(self) -> list:
        """Get list of available models for this provider"""
        return list(self.model_settings.keys())
    
    def validate_configuration(self) -> dict:
        """Validate provider configuration"""
        status = {
            'provider': 'google',
            'authentication': False,
            'models_available': len(self.model_settings),
            'issues': []
        }
        
        # Check authentication
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            creds_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            if os.path.exists(creds_path):
                status['authentication'] = True
                status['credentials_path'] = creds_path
            else:
                status['issues'].append(f"Credentials file not found: {creds_path}")
        else:
            status['issues'].append("GOOGLE_APPLICATION_CREDENTIALS not set")
        
        # Check if SDK is available
        try:
            import google.genai
            status['sdk_available'] = True
        except ImportError:
            status['sdk_available'] = False
            status['issues'].append("google-genai SDK not installed")
        
        return status
