### 6. **providers/base_provider.py** - Abstract Provider Interface
"""
Abstract base provider interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        self.request_times = []
        self.token_count = 0
        self.token_reset_time = time.time()
        
        self.lock = asyncio.Lock()
    
    async def acquire(self, estimated_tokens: int = 0):
        """Wait if necessary to respect rate limits"""
        async with self.lock:
            now = time.time()
            
            # Clean old request times
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            # Check request rate
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit: sleeping {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                    now = time.time()
            
            # Check token rate
            if now - self.token_reset_time >= 60:
                self.token_count = 0
                self.token_reset_time = now
            
            if self.token_count + estimated_tokens > self.tokens_per_minute:
                sleep_time = 60 - (now - self.token_reset_time)
                if sleep_time > 0:
                    logger.debug(f"Token limit: sleeping {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                    self.token_count = 0
                    self.token_reset_time = time.time()
            
            # Record request
            self.request_times.append(now)
            self.token_count += estimated_tokens

class BaseProvider(ABC):
    """Base class for all AI providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter = self._create_rate_limiter()
        self.session = None
    
    @abstractmethod
    async def complete(self, prompt: str, model: str) -> str:
        """
        Send completion request to provider
        
        Args:
            prompt: The prompt to send
            model: Model identifier (can be alias)
            
        Returns:
            The completion response
        """
        pass
    
    @abstractmethod
    def resolve_model_alias(self, alias: str) -> str:
        """
        Convert alias (gpt-1) to actual model name
        
        Args:
            alias: Model alias
            
        Returns:
            Actual model identifier
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count for prompt and response
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Estimated total token count
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, tokens: int, model: str) -> float:
        """
        Estimate cost for token usage
        
        Args:
            tokens: Total token count
            model: Model identifier
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    def _create_rate_limiter(self) -> Optional[RateLimiter]:
        """Create rate limiter from config"""
        rate_limits = self.config.get('rate_limits', {})
        
        if rate_limits:
            return RateLimiter(
                requests_per_minute=rate_limits.get('requests_per_minute', 60),
                tokens_per_minute=rate_limits.get('tokens_per_minute', 90000)
            )
        
        return None
    
    async def _apply_rate_limit(self, estimated_tokens: int = 0):
        """Apply rate limiting if configured"""
        if self.rate_limiter:
            await self.rate_limiter.acquire(estimated_tokens)
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return self.config.get('headers', {})
    
    def get_endpoint(self) -> str:
        """Get API endpoint"""
        return self.config.get('endpoint', '')
    
    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
