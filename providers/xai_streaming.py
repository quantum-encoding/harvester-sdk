"""
xAI Streaming Module for Real-time Response Generation
Supports streaming for all text models (not image generation)
"""
import logging
import json
import asyncio
import aiohttp
from typing import Any, Dict, List, Union, Optional, AsyncGenerator, Tuple
import os
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StreamChunk:
    """Represents a single chunk from the stream"""
    content: str
    role: Optional[str] = None
    finish_reason: Optional[str] = None
    index: int = 0
    
@dataclass
class StreamResponse:
    """Accumulates chunks into a complete response"""
    content: str = ""
    role: str = "assistant"
    finish_reason: Optional[str] = None
    usage: Optional[Dict] = None
    citations: List[str] = None
    reasoning_content: Optional[str] = None
    created_at: Optional[int] = None
    model: Optional[str] = None
    
    def add_chunk(self, chunk: StreamChunk):
        """Add a chunk to the accumulated response"""
        if chunk.content:
            self.content += chunk.content
        if chunk.role:
            self.role = chunk.role
        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason

class XaiStreamingProvider:
    """
    Streaming provider for xAI models using Server-Sent Events (SSE).
    Provides real-time text generation with support for reasoning models.
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 3600):
        """
        Initialize streaming provider.
        
        Args:
            api_key: xAI API key (or from environment)
            timeout: Request timeout in seconds (default 3600 for reasoning models)
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key required for streaming")
        
        self.base_url = "https://api.x.ai/v1"
        self.timeout = timeout
        
        # Model configurations
        self.reasoning_models = {"grok-4", "grok-3-mini", "grok-3-mini-fast"}
        self.supports_reasoning_effort = {"grok-3-mini", "grok-3-mini-fast"}
        
        logger.info(f"xAI Streaming provider initialized with {timeout}s timeout")
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-4",
        temperature: float = 0.7,
        max_tokens: int = 131072,
        reasoning_effort: Optional[str] = None,
        search_parameters: Optional[Dict] = None,
        **kwargs
    ) -> AsyncGenerator[Tuple[StreamResponse, StreamChunk], None]:
        """
        Stream a completion from xAI models.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            reasoning_effort: For supported models, "low" or "high"
            search_parameters: Live Search configuration
            **kwargs: Additional parameters
            
        Yields:
            Tuple of (accumulated_response, current_chunk)
        """
        # Build request
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True  # Enable streaming
        }
        
        # Add reasoning effort for supported models
        if model in self.supports_reasoning_effort and reasoning_effort:
            request_data["reasoning_effort"] = reasoning_effort
            logger.info(f"Streaming with reasoning_effort: {reasoning_effort}")
        
        # Add search parameters if provided
        if search_parameters:
            request_data["search_parameters"] = search_parameters
            logger.info("Streaming with Live Search enabled")
        
        # Remove unsupported parameters for reasoning models
        if model in self.reasoning_models:
            for param in ["presence_penalty", "frequency_penalty", "stop"]:
                request_data.pop(param, None)
                kwargs.pop(param, None)
        
        # Add any additional parameters
        request_data.update(kwargs)
        
        # Create streaming request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        accumulated_response = StreamResponse()
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Stream request failed: {response.status} - {error_text}")
                
                # Process SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line or line.startswith(':'):
                        # Skip empty lines and comments
                        continue
                    
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        
                        if data == '[DONE]':
                            # Stream complete
                            logger.info("Stream completed")
                            break
                        
                        try:
                            # Parse JSON chunk
                            chunk_data = json.loads(data)
                            
                            # Extract chunk information
                            chunk = self._parse_chunk(chunk_data)
                            
                            # Update accumulated response
                            accumulated_response.add_chunk(chunk)
                            
                            # Update metadata from chunk
                            if 'created' in chunk_data:
                                accumulated_response.created_at = chunk_data['created']
                            if 'model' in chunk_data:
                                accumulated_response.model = chunk_data['model']
                            if 'usage' in chunk_data:
                                accumulated_response.usage = chunk_data['usage']
                            
                            # Yield both accumulated and current chunk
                            yield accumulated_response, chunk
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse chunk: {e}")
                            continue
    
    def _parse_chunk(self, chunk_data: Dict) -> StreamChunk:
        """Parse a chunk from the SSE stream"""
        chunk = StreamChunk(content="")
        
        if 'choices' in chunk_data and chunk_data['choices']:
            choice = chunk_data['choices'][0]
            
            if 'delta' in choice:
                delta = choice['delta']
                if 'content' in delta:
                    chunk.content = delta['content']
                if 'role' in delta:
                    chunk.role = delta['role']
            
            if 'finish_reason' in choice:
                chunk.finish_reason = choice['finish_reason']
            
            if 'index' in choice:
                chunk.index = choice['index']
        
        return chunk
    
    async def stream_with_search(
        self,
        prompt: str,
        model: str = "grok-4",
        search_mode: str = "auto",
        sources: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncGenerator[Tuple[StreamResponse, StreamChunk], None]:
        """
        Stream a completion with Live Search enabled.
        
        Args:
            prompt: User prompt
            model: Model to use
            search_mode: "auto", "on", or "off"
            sources: List of search source configurations
            **kwargs: Additional parameters
            
        Yields:
            Tuple of (accumulated_response, current_chunk)
        """
        # Build search parameters
        search_params = {
            "mode": search_mode,
            "return_citations": True
        }
        
        if sources:
            search_params["sources"] = sources
        else:
            # Default sources
            search_params["sources"] = [
                {"type": "web"},
                {"type": "x"},
                {"type": "news"}
            ]
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are Grok, a helpful AI assistant with access to real-time information."},
            {"role": "user", "content": prompt}
        ]
        
        # Stream with search
        async for response, chunk in self.stream_completion(
            messages=messages,
            model=model,
            search_parameters=search_params,
            **kwargs
        ):
            yield response, chunk
    
    async def stream_reasoning(
        self,
        prompt: str,
        model: str = "grok-3-mini",
        reasoning_effort: str = "high",
        return_reasoning: bool = False,
        **kwargs
    ) -> AsyncGenerator[Tuple[StreamResponse, StreamChunk], None]:
        """
        Stream a reasoning model completion.
        
        Args:
            prompt: Problem to solve
            model: Reasoning model to use
            reasoning_effort: "low" or "high" (grok-3-mini/fast only)
            return_reasoning: Whether to request reasoning trace
            **kwargs: Additional parameters
            
        Yields:
            Tuple of (accumulated_response, current_chunk)
        """
        if model not in self.reasoning_models:
            logger.warning(f"{model} is not a reasoning model")
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are Grok, a highly intelligent reasoning AI."},
            {"role": "user", "content": prompt}
        ]
        
        # Stream with reasoning
        async for response, chunk in self.stream_completion(
            messages=messages,
            model=model,
            reasoning_effort=reasoning_effort if model in self.supports_reasoning_effort else None,
            **kwargs
        ):
            # Note: reasoning_content is typically only available in the final response
            # not in streaming chunks
            yield response, chunk

# Convenience functions for simple streaming

async def stream_simple(
    prompt: str,
    model: str = "grok-4",
    api_key: Optional[str] = None,
    print_output: bool = True
) -> str:
    """
    Simple streaming function that prints output and returns full response.
    
    Args:
        prompt: User prompt
        model: Model to use
        api_key: API key (optional, uses environment)
        print_output: Whether to print chunks as they arrive
        
    Returns:
        Complete response text
    """
    provider = XaiStreamingProvider(api_key=api_key)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    full_response = ""
    async for response, chunk in provider.stream_completion(messages, model):
        if print_output and chunk.content:
            print(chunk.content, end="", flush=True)
        full_response = response.content
    
    if print_output:
        print()  # Final newline
    
    return full_response

async def stream_with_callback(
    prompt: str,
    callback: callable,
    model: str = "grok-4",
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Stream with a custom callback function for each chunk.
    
    Args:
        prompt: User prompt
        callback: Function called with each chunk (chunk_content, accumulated_content)
        model: Model to use
        api_key: API key
        **kwargs: Additional parameters
        
    Returns:
        Complete response text
    """
    provider = XaiStreamingProvider(api_key=api_key)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    full_response = ""
    async for response, chunk in provider.stream_completion(messages, model, **kwargs):
        if chunk.content:
            await callback(chunk.content, response.content)
        full_response = response.content
    
    return full_response

# Example usage
if __name__ == "__main__":
    async def example_streaming():
        # Simple streaming
        print("=== Simple Streaming ===")
        response = await stream_simple(
            "Tell me a short story about a robot learning to paint.",
            model="grok-4"
        )
        print(f"\nFull response length: {len(response)} chars")
        
        # Streaming with search
        print("\n=== Streaming with Live Search ===")
        provider = XaiStreamingProvider()
        async for response, chunk in provider.stream_with_search(
            "What are the latest AI announcements this week?",
            search_mode="on"
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()
        
        # Streaming with reasoning
        print("\n=== Streaming with Reasoning ===")
        async for response, chunk in provider.stream_reasoning(
            "Solve: If x + 2y = 10 and 3x - y = 5, find x and y.",
            model="grok-3-mini",
            reasoning_effort="high"
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()
        
        # Usage stats
        if response.usage:
            print(f"\nToken usage: {response.usage}")
    
    # Run example
    asyncio.run(example_streaming())