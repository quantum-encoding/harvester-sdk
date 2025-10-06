"""
Google Vertex AI provider implementation
"""
import aiohttp
import json
import os
import subprocess
import time
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class VertexProvider(BaseProvider):
    """Provider for Google Vertex AI models with working gcloud authentication"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Vertex AI configuration
        self.project_id = config.get('project_id', 'gen-lang-client-0685363971')
        self.location = config.get('location', 'us-east5')
        
        # Build endpoints for different publishers
        self.google_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models"
        self.anthropic_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/anthropic/models"
        
        # Model-specific settings
        self.model_settings = {
            # Anthropic Claude models via Vertex AI
            'claude-opus-4': {
                'max_tokens': 32000,
                'temperature': 1.0,
                'cost_per_million_input': 15.0,
                'cost_per_million_output': 75.0,
                'publisher': 'anthropic'
            },
            'claude-sonnet-4': {
                'max_tokens': 64000,
                'temperature': 1.0,
                'cost_per_million_input': 3.0,
                'cost_per_million_output': 15.0,
                'publisher': 'anthropic'
            },
            'claude-3-5-haiku': {
                'max_tokens': 4096,
                'temperature': 1.0,
                'cost_per_million_input': 1.0,
                'cost_per_million_output': 5.0,
                'publisher': 'anthropic'
            },
        }
        
        # Use gcloud auth (proven working approach from json_processor)
        self.access_token = None
        self.token_expiry = None
    
    def _get_access_token(self) -> str:
        """Get Google Cloud access token using gcloud CLI (proven working approach)"""
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
        """Send completion request to Vertex AI"""
        # Resolve model alias
        actual_model = self.resolve_model_alias(model)
        
        # Apply rate limiting
        estimated_tokens = self.estimate_tokens(prompt, "")
        await self._apply_rate_limit(estimated_tokens)
        
        # Get access token
        access_token = await self.get_access_token()
        
        # Get model settings to determine publisher
        settings = self.model_settings.get(actual_model)
        if not settings:
            raise ValueError(f"Unknown Vertex AI model: {actual_model}")
        
        # Route to appropriate completion method based on publisher
        if settings.get('publisher') == 'anthropic':
            return await self._complete_anthropic(prompt, actual_model, access_token)
        elif 'gemini' in actual_model:
            return await self._complete_gemini(prompt, actual_model, access_token)
        elif 'bison' in actual_model:
            return await self._complete_palm(prompt, actual_model, access_token)
        else:
            raise ValueError(f"Unknown Vertex AI model: {actual_model}")
    
    async def _complete_anthropic(self, prompt: str, model: str, access_token: str) -> str:
        """Complete using Anthropic Claude models via Vertex AI (exact pattern from working curl scripts)"""
        url = f"{self.anthropic_url}/{model}:rawPredict"
        
        settings = self.model_settings.get(model)
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        # Exact payload structure from working curl scripts
        payload = {
            "anthropic_version": "vertex-2023-10-16",
            "stream": False,
            "max_tokens": settings['max_tokens'],
            "temperature": settings['temperature'],
            "top_p": 1,
            "top_k": 1,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Vertex AI Anthropic error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Extract content from Anthropic response
                if 'content' in data and data['content']:
                    return data['content'][0].get('text', '')
                
                raise Exception("No response content from Vertex AI Anthropic")
                
        except Exception as e:
            logger.error(f"Vertex AI Anthropic completion error: {e}")
            raise
    
    async def _complete_gemini(self, prompt: str, model: str, access_token: str) -> str:
        """Complete using Gemini models"""
        url = f"{self.google_url}/{model}:generateContent"
        
        # Only Claude models remain in Vertex provider (Gemini moved to google provider)
        settings = self.model_settings.get(model)
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": settings['temperature'],
                "maxOutputTokens": settings['max_tokens'],
                "topP": 0.95,
                "topK": 40
            }
        }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Vertex AI error: {response.status} - {error_text}")
                
                data = await response.json()
                
                if 'candidates' in data and data['candidates']:
                    content = data['candidates'][0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        return parts[0].get('text', '')
                
                raise Exception("No response content from Vertex AI")
                
        except Exception as e:
            logger.error(f"Vertex AI Gemini completion error: {e}")
            raise
    
    async def _complete_palm(self, prompt: str, model: str, access_token: str) -> str:
        """Complete using PaLM models (text-bison, code-bison)"""
        url = f"{self.google_url}/{model}:predict"
        
        settings = self.model_settings.get(model, self.model_settings['text-bison'])
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "instances": [{
                "prompt": prompt
            }],
            "parameters": {
                "temperature": settings['temperature'],
                "maxOutputTokens": settings['max_tokens'],
                "topP": 0.95,
                "topK": 40
            }
        }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Vertex AI error: {response.status} - {error_text}")
                
                data = await response.json()
                
                if 'predictions' in data and data['predictions']:
                    return data['predictions'][0].get('content', '')
                
                raise Exception("No response content from Vertex AI")
                
        except Exception as e:
            logger.error(f"Vertex AI PaLM completion error: {e}")
            raise
    
    def resolve_model_alias(self, alias: str) -> str:
        """Convert alias to actual model name"""
        aliases = self.config.get('aliases', {})
        return aliases.get(alias, alias)
    
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for Vertex AI models"""
        total_chars = len(prompt) + len(response)
        return int(total_chars / 4)
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        actual_model = self.resolve_model_alias(model)
        # Only Claude models remain in Vertex provider
        settings = self.model_settings.get(actual_model)
        
        # Assume 70% input, 30% output for cost calculation
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        input_cost = (input_tokens / 1_000_000) * settings['cost_per_million_input']
        output_cost = (output_tokens / 1_000_000) * settings['cost_per_million_output']
        
        return input_cost + output_cost
