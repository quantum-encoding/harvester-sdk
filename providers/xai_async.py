"""
xAI Asynchronous Request Handler
Efficiently process multiple requests in parallel with rate limiting
"""
import asyncio
import logging
from typing import Any, Dict, List, Union, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import time
from openai import AsyncOpenAI
import aiofiles
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RequestResult:
    """Result from an async request"""
    request_id: str
    prompt: str
    response: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    processing_time: float = 0.0
    model: str = ""
    
@dataclass
class BatchResults:
    """Results from a batch of async requests"""
    total_requests: int
    successful: int
    failed: int
    total_tokens: int
    total_time: float
    results: List[RequestResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return (self.successful / self.total_requests * 100) if self.total_requests > 0 else 0

class XaiAsyncProcessor:
    """
    Asynchronous request processor for xAI models.
    Handles rate limiting, error recovery, and parallel processing.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        max_retries: int = 3,
        timeout: int = 300,
        rate_limit_per_minute: int = 60
    ):
        """
        Initialize async processor.
        
        Args:
            api_key: xAI API key
            max_concurrent: Maximum concurrent requests
            max_retries: Maximum retries per request
            timeout: Request timeout in seconds
            rate_limit_per_minute: API rate limit
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key required")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
            timeout=timeout
        )
        
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.rate_limit_per_minute = rate_limit_per_minute
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Rate limiting
        self.request_times: List[float] = []
        self.rate_limit_lock = asyncio.Lock()
        
        logger.info(f"Async processor initialized: {max_concurrent} concurrent, {rate_limit_per_minute} req/min")
    
    async def _wait_for_rate_limit(self):
        """Enforce rate limiting"""
        async with self.rate_limit_lock:
            now = time.time()
            # Remove timestamps older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.rate_limit_per_minute:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_time = 60 - (now - oldest_request) + 0.1
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
            
            self.request_times.append(now)
    
    async def process_single_request(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        request_id: str,
        model: str = "grok-4",
        **kwargs
    ) -> RequestResult:
        """
        Process a single request with retry logic.
        
        Args:
            prompt: Text prompt or message list
            request_id: Unique request identifier
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            RequestResult with response or error
        """
        start_time = time.time()
        result = RequestResult(
            request_id=request_id,
            prompt=prompt if isinstance(prompt, str) else json.dumps(prompt),
            model=model
        )
        
        # Prepare messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                # Wait for rate limit
                await self._wait_for_rate_limit()
                
                # Make request with semaphore
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                
                # Extract response
                result.response = response.choices[0].message.content
                result.tokens_used = response.usage.total_tokens if response.usage else 0
                result.processing_time = time.time() - start_time
                
                logger.debug(f"Request {request_id} completed: {result.tokens_used} tokens")
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout on attempt {attempt + 1}"
                logger.warning(f"Request {request_id}: {error_msg}")
                result.error = error_msg
                
            except Exception as e:
                error_msg = f"Error on attempt {attempt + 1}: {str(e)}"
                logger.error(f"Request {request_id}: {error_msg}")
                result.error = error_msg
                
                # Exponential backoff for retries
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying request {request_id} in {wait_time}s")
                    await asyncio.sleep(wait_time)
        
        result.processing_time = time.time() - start_time
        return result
    
    async def process_batch(
        self,
        requests: List[Dict[str, Any]],
        model: str = "grok-4",
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> BatchResults:
        """
        Process multiple requests in parallel.
        
        Args:
            requests: List of request dictionaries with 'prompt' and optional 'id'
            model: Model to use for all requests
            progress_callback: Optional callback for progress updates
            **kwargs: Additional parameters for all requests
            
        Returns:
            BatchResults with all responses
        """
        start_time = time.time()
        tasks = []
        
        # Create tasks for each request
        for i, req in enumerate(requests):
            request_id = req.get('id', f"req_{i}")
            prompt = req.get('prompt')
            
            if not prompt:
                logger.warning(f"Skipping request {request_id}: no prompt")
                continue
            
            # Merge request-specific kwargs with global kwargs
            req_kwargs = {**kwargs, **req.get('kwargs', {})}
            
            task = self.process_single_request(
                prompt=prompt,
                request_id=request_id,
                model=req.get('model', model),
                **req_kwargs
            )
            tasks.append(task)
        
        # Process with progress updates
        results = []
        if progress_callback:
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                results.append(result)
                await progress_callback(i + 1, len(tasks), result)
        else:
            results = await asyncio.gather(*tasks)
        
        # Calculate statistics
        successful = sum(1 for r in results if r.response and not r.error)
        failed = sum(1 for r in results if r.error)
        total_tokens = sum(r.tokens_used for r in results)
        total_time = time.time() - start_time
        
        batch_results = BatchResults(
            total_requests=len(results),
            successful=successful,
            failed=failed,
            total_tokens=total_tokens,
            total_time=total_time,
            results=results
        )
        
        logger.info(
            f"Batch completed: {successful}/{len(results)} successful, "
            f"{total_tokens} tokens, {total_time:.1f}s"
        )
        
        return batch_results
    
    async def process_file_batch(
        self,
        input_file: str,
        output_file: str,
        model: str = "grok-4",
        **kwargs
    ) -> BatchResults:
        """
        Process requests from a JSONL file.
        
        Args:
            input_file: Path to JSONL file with requests
            output_file: Path to save results
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            BatchResults
        """
        # Load requests from file
        requests = []
        async with aiofiles.open(input_file, 'r') as f:
            async for line in f:
                try:
                    req = json.loads(line.strip())
                    requests.append(req)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON line: {line}")
        
        logger.info(f"Loaded {len(requests)} requests from {input_file}")
        
        # Process batch
        results = await self.process_batch(requests, model, **kwargs)
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(output_file, 'w') as f:
            for result in results.results:
                result_dict = {
                    'id': result.request_id,
                    'prompt': result.prompt,
                    'response': result.response,
                    'error': result.error,
                    'tokens': result.tokens_used,
                    'time': result.processing_time,
                    'model': result.model
                }
                await f.write(json.dumps(result_dict) + '\n')
        
        logger.info(f"Results saved to {output_file}")
        return results
    
    async def close(self):
        """Close the async client"""
        await self.client.close()

# Convenience functions

async def quick_async_batch(
    prompts: List[str],
    model: str = "grok-4",
    max_concurrent: int = 10,
    **kwargs
) -> List[str]:
    """
    Quick function to process multiple prompts asynchronously.
    
    Args:
        prompts: List of text prompts
        model: Model to use
        max_concurrent: Maximum concurrent requests
        **kwargs: Additional parameters
        
    Returns:
        List of responses
    """
    processor = XaiAsyncProcessor(max_concurrent=max_concurrent)
    
    requests = [{'prompt': p} for p in prompts]
    results = await processor.process_batch(requests, model, **kwargs)
    
    await processor.close()
    
    return [r.response for r in results.results if r.response]

async def process_with_progress(
    prompts: List[str],
    model: str = "grok-4"
) -> BatchResults:
    """
    Process prompts with progress bar.
    
    Args:
        prompts: List of prompts
        model: Model to use
        
    Returns:
        BatchResults
    """
    processor = XaiAsyncProcessor()
    
    async def progress(completed, total, result):
        success = "✓" if result.response else "✗"
        print(f"Progress: {completed}/{total} [{success}] - {result.request_id}")
    
    requests = [{'prompt': p, 'id': f"req_{i}"} for i, p in enumerate(prompts)]
    results = await processor.process_batch(
        requests,
        model,
        progress_callback=progress
    )
    
    await processor.close()
    
    print(f"\nCompleted: {results.success_rate:.1f}% success rate")
    print(f"Total tokens: {results.total_tokens}")
    print(f"Total time: {results.total_time:.1f}s")
    
    return results

# Example usage
if __name__ == "__main__":
    async def example():
        # Example 1: Quick batch
        prompts = [
            "Tell me a joke",
            "Write a haiku about AI",
            "Generate a funny X post",
            "Say something unhinged"
        ]
        
        print("=== Quick Batch ===")
        responses = await quick_async_batch(
            prompts,
            model="grok-3-mini",
            max_concurrent=2,
            temperature=0.9
        )
        
        for i, response in enumerate(responses):
            print(f"\n{i+1}. {response[:100]}...")
        
        # Example 2: With progress tracking
        print("\n=== With Progress ===")
        math_problems = [
            "What is 2+2?",
            "Solve x^2 - 5x + 6 = 0",
            "What's the derivative of sin(x)?",
            "Calculate the factorial of 7"
        ]
        
        results = await process_with_progress(math_problems, "grok-3-mini")
        
        # Example 3: Advanced batch with mixed parameters
        print("\n=== Advanced Batch ===")
        processor = XaiAsyncProcessor(max_concurrent=3)
        
        advanced_requests = [
            {
                'id': 'joke_1',
                'prompt': 'Tell me a programming joke',
                'kwargs': {'temperature': 1.0, 'max_tokens': 100}
            },
            {
                'id': 'code_1',
                'prompt': 'Write a Python function to reverse a string',
                'model': 'grok-4',
                'kwargs': {'temperature': 0.3}
            },
            {
                'id': 'reason_1',
                'prompt': 'Why is the sky blue?',
                'model': 'grok-3-mini',
                'kwargs': {'reasoning_effort': 'high'}
            }
        ]
        
        batch_results = await processor.process_batch(advanced_requests)
        
        for result in batch_results.results:
            status = "Success" if result.response else f"Failed: {result.error}"
            print(f"{result.request_id}: {status} ({result.tokens_used} tokens)")
        
        await processor.close()
    
    asyncio.run(example())