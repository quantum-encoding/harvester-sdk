"""
XAI Async Processor - High-Throughput Concurrent Processing

Native async processing with XAI's AsyncClient for maximum throughput
without the complexity of batch APIs.

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import os
import time

logger = logging.getLogger(__name__)


class XAIAsyncProcessor:
    """
    XAI Async Processing with native AsyncClient
    
    This leverages XAI's unique approach - no batch API, but excellent
    async support with built-in rate limiting via semaphores.
    """
    
    def __init__(self, 
                 api_key: str = None,
                 max_concurrent: int = 10,
                 timeout: int = 3600,
                 model: str = "grok-4"):
        """
        Initialize XAI Async Processor
        
        Args:
            api_key: XAI API key (or set XAI_API_KEY)
            max_concurrent: Maximum concurrent requests (respect rate limits!)
            timeout: Request timeout in seconds
            model: Default model (grok-4, grok-4-turbo)
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI API key required")
        
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.model = model
        
        # Initialize async client (lazy loading)
        self._client = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"⚡ XAI Async Processor initialized")
        logger.info(f"🚀 Max concurrent: {max_concurrent}")
        logger.info(f"🤖 Default model: {model}")
    
    async def _get_client(self):
        """Lazy initialize the async client"""
        if self._client is None:
            try:
                from xai_sdk import AsyncClient
                self._client = AsyncClient(
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            except ImportError:
                # Fallback to OpenAI-compatible client
                logger.warning("xai_sdk not found, using OpenAI-compatible client")
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1",
                    timeout=self.timeout
                )
        return self._client
    
    async def process_request(self, 
                             prompt: str,
                             system: str = None,
                             model: str = None,
                             max_tokens: int = 8192,
                             temperature: float = 0.7,
                             custom_id: str = None) -> Dict:
        """
        Process a single request
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            model: Override default model
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            custom_id: Custom identifier
            
        Returns:
            Response dictionary
        """
        client = await self._get_client()
        model = model or self.model
        
        try:
            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            # Make request based on client type
            if hasattr(client, 'chat'):
                # xai_sdk client
                from xai_sdk.chat import user, system as sys_msg
                
                chat = client.chat.create(
                    model=model,
                    max_tokens=max_tokens
                )
                
                if system:
                    chat.append(sys_msg(system))
                chat.append(user(prompt))
                
                response = await chat.sample()
                
                return {
                    'custom_id': custom_id,
                    'status': 'success',
                    'content': response.content,
                    'model': model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'reasoning_content': getattr(response.message, 'reasoning_content', None)
                }
            else:
                # OpenAI-compatible client
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                return {
                    'custom_id': custom_id,
                    'status': 'success',
                    'content': response.choices[0].message.content,
                    'model': model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'reasoning_content': getattr(response.choices[0].message, 'reasoning_content', None)
                }
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return {
                'custom_id': custom_id,
                'status': 'failed',
                'error': str(e),
                'model': model
            }
    
    async def process_batch_async(self,
                                 requests: List[Dict],
                                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process multiple requests concurrently with semaphore rate limiting
        
        This is the XAI way - no batch API, just excellent async concurrency.
        
        Args:
            requests: List of request dictionaries
            progress_callback: Optional callback for progress updates
            
        Returns:
            Batch results with statistics
        """
        
        self.stats['start_time'] = datetime.now()
        self.stats['total_requests'] = len(requests)
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        logger.info(f"🚀 Processing {len(requests)} requests with {self.max_concurrent} concurrent workers")
        
        completed = 0
        
        async def process_single(request: Dict, index: int) -> Dict:
            nonlocal completed
            
            async with semaphore:
                # Log progress
                logger.debug(f"Processing request {index + 1}/{len(requests)}")
                
                # Process the request
                result = await self.process_request(
                    prompt=request.get('prompt', ''),
                    system=request.get('system'),
                    model=request.get('model', self.model),
                    max_tokens=request.get('max_tokens', 8192),
                    temperature=request.get('temperature', 0.7),
                    custom_id=request.get('custom_id', f'request-{index}')
                )
                
                # Update statistics
                completed += 1
                if result['status'] == 'success':
                    self.stats['successful_requests'] += 1
                    if 'usage' in result:
                        self.stats['total_tokens'] += result['usage']['total_tokens']
                else:
                    self.stats['failed_requests'] += 1
                
                # Progress callback
                if progress_callback:
                    await progress_callback({
                        'completed': completed,
                        'total': len(requests),
                        'progress_percent': (completed / len(requests)) * 100,
                        'result': result
                    })
                
                # Log progress every 10 requests
                if completed % 10 == 0 or completed == len(requests):
                    logger.info(f"⚡ Progress: {completed}/{len(requests)} ({completed/len(requests)*100:.1f}%)")
                
                return result
        
        # Create all tasks
        tasks = [process_single(req, idx) for idx, req in enumerate(requests)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'custom_id': f'request-{idx}',
                    'status': 'failed',
                    'error': str(result)
                })
                self.stats['failed_requests'] += 1
            else:
                final_results.append(result)
        
        self.stats['end_time'] = datetime.now()
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # Prepare final report
        batch_report = {
            'batch_id': f"xai_async_{int(self.stats['start_time'].timestamp())}",
            'status': 'completed',
            'statistics': {
                **self.stats,
                'duration_seconds': duration,
                'requests_per_second': len(requests) / duration if duration > 0 else 0,
                'success_rate': self.stats['successful_requests'] / len(requests) if requests else 0,
                'average_tokens_per_request': self.stats['total_tokens'] / self.stats['successful_requests'] 
                                              if self.stats['successful_requests'] > 0 else 0
            },
            'results': final_results
        }
        
        logger.info(f"✅ Batch complete!")
        logger.info(f"📊 Successful: {self.stats['successful_requests']}/{len(requests)}")
        logger.info(f"⏱️  Duration: {duration:.1f}s")
        logger.info(f"🚀 Throughput: {len(requests)/duration:.1f} req/s")
        logger.info(f"💰 Total tokens: {self.stats['total_tokens']:,}")
        
        return batch_report
    
    async def process_with_retry(self,
                                requests: List[Dict],
                                max_retries: int = 3,
                                retry_delay: float = 1.0) -> Dict:
        """
        Process requests with automatic retry on failures
        
        Args:
            requests: List of request dictionaries
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (exponential backoff)
            
        Returns:
            Batch results with retry statistics
        """
        
        all_results = []
        remaining_requests = requests.copy()
        attempt = 0
        
        while remaining_requests and attempt < max_retries:
            attempt += 1
            logger.info(f"🔄 Attempt {attempt}/{max_retries} - Processing {len(remaining_requests)} requests")
            
            # Process batch
            batch_result = await self.process_batch_async(remaining_requests)
            
            # Separate successful and failed
            failed_requests = []
            for idx, result in enumerate(batch_result['results']):
                if result['status'] == 'success':
                    all_results.append(result)
                else:
                    # Prepare for retry
                    original_request = remaining_requests[idx]
                    failed_requests.append(original_request)
            
            remaining_requests = failed_requests
            
            # If we have failures and more retries, wait before retrying
            if remaining_requests and attempt < max_retries:
                wait_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(f"⏳ Retrying {len(remaining_requests)} failed requests in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        # Add any remaining failures
        for req in remaining_requests:
            all_results.append({
                'custom_id': req.get('custom_id', 'unknown'),
                'status': 'failed',
                'error': f'Failed after {max_retries} attempts'
            })
        
        successful = sum(1 for r in all_results if r['status'] == 'success')
        
        return {
            'batch_id': f"xai_async_retry_{int(time.time())}",
            'total_requests': len(requests),
            'successful': successful,
            'failed': len(requests) - successful,
            'retry_attempts': attempt,
            'results': all_results
        }
    
    def reset_stats(self):
        """Reset statistics for fresh batch"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'start_time': None,
            'end_time': None
        }