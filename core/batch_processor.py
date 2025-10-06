"""
Parallel batch processor for API calls with enhanced JSON serialization and AI Council support
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import json
import aiofiles
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles complex objects like dataclasses, 
    Path objects, and datetime objects
    """
    
    def default(self, obj):
        """Convert non-serializable objects to serializable format"""
        
        # Handle objects with to_dict method (like ScanResult)
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle dataclass objects
        if hasattr(obj, '__dataclass_fields__'):
            return {
                field.name: getattr(obj, field.name) 
                for field in obj.__dataclass_fields__.values()
            }
        
        # Handle sets (convert to list)
        if isinstance(obj, set):
            return list(obj)
        
        # Let the base class handle other types
        return super().default(obj)

@dataclass
class BatchJob:
    """Represents a single job in the batch"""
    id: str
    prompt: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = 'pending'  # pending, processing, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BatchJob to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'model': self.model,
            'status': self.status,
            'metadata': self.metadata,
            'error': self.error,
            'attempts': self.attempts,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'has_result': bool(self.result),
            'result_length': len(self.result) if self.result else 0,
            'result': self.result  # Include the actual result content
        }

@dataclass
class BatchResult:
    """Result of batch processing"""
    total_jobs: int
    successful: int
    failed: int
    duration: float
    results: List[BatchJob]
    stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BatchResult to dictionary for JSON serialization"""
        return {
            'total_jobs': self.total_jobs,
            'successful': self.successful,
            'failed': self.failed,
            'duration': self.duration,
            'stats': self.stats,
            'results': [job.to_dict() for job in self.results]
        }

class BatchProcessor:
    """Handles parallel processing of API calls with rate limiting, robust serialization, and AI Council support"""
    
    def __init__(self, provider_factory, config: Optional[Dict] = None):
        self.provider_factory = provider_factory
        self.config = config or {}
        
        # Concurrency control
        self.max_concurrent = self.config.get('max_concurrent', 100)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Retry configuration
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Progress tracking
        self.progress_callback = None
        self.stats = {
            'total_tokens': 0,
            'total_cost': 0,
            'api_calls': 0,
            'cache_hits': 0
        }
        
        logger.info(f"BatchProcessor initialized with max_concurrent={self.max_concurrent}")
    
    async def process_council_request(self, job: BatchJob, providers: List[str]) -> Dict[str, Any]:
        """
        Process a single job across multiple providers in parallel (for AI Council mode)
        
        Args:
            job: The job to process
            providers: List of provider/model aliases to use
            
        Returns:
            Dictionary mapping provider names to their responses
        """
        logger.info(f"Processing council request for job {job.id} with {len(providers)} providers")
        
        # Create a job for each provider
        council_jobs = []
        for provider_alias in providers:
            council_job = BatchJob(
                id=f"{job.id}_{provider_alias}",  # Fixed the incomplete line
                prompt=job.prompt,
                model=provider_alias,
                metadata={
                    **job.metadata,
                    'council_member': provider_alias,
                    'original_job_id': job.id
                }
            )
            council_jobs.append(council_job)
        
        # Process all jobs in parallel
        tasks = []
        for council_job in council_jobs:
            task = self._process_single_job(council_job)
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect responses
        council_responses = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Council member {council_jobs[i].model} failed: {result}")
                council_responses[council_jobs[i].model] = {
                    'error': str(result),
                    'status': 'failed'
                }
                continue
                
            if council_jobs[i].status == 'completed' and council_jobs[i].result:
                council_responses[council_jobs[i].model] = {
                    'result': council_jobs[i].result,
                    'status': 'completed',
                    'metadata': council_jobs[i].metadata,
                    'attempts': council_jobs[i].attempts,
                    'completed_at': council_jobs[i].completed_at.isoformat() if council_jobs[i].completed_at else None
                }
            else:
                council_responses[council_jobs[i].model] = {
                    'error': council_jobs[i].error or 'Unknown error',
                    'status': council_jobs[i].status,
                    'attempts': council_jobs[i].attempts
                }
        
        logger.info(f"Council request completed: {len(council_responses)} responses collected")
        return council_responses

    async def process_batch(
        self, 
        jobs: List[BatchJob],
        progress_callback: Optional[Callable] = None,
        save_intermediate: bool = True
    ) -> BatchResult:
        """
        Process batch of jobs in parallel
        
        Args:
            jobs: List of batch jobs to process
            progress_callback: Optional callback for progress updates
            save_intermediate: Save results as they complete
            
        Returns:
            BatchResult with all results and statistics
        """
        if not jobs:
            logger.warning("No jobs to process")
            return BatchResult(0, 0, 0, 0.0, [], {})
        
        start_time = time.time()
        self.progress_callback = progress_callback
        
        # Reset stats
        self.stats = {
            'total_tokens': 0,
            'total_cost': 0,
            'api_calls': 0,
            'cache_hits': 0,
            'by_model': {},
            'errors': {}
        }
        
        logger.info(f"Starting batch processing of {len(jobs)} jobs")
        
        # Create progress bar
        progress_bar = tqdm(
            total=len(jobs),
            desc="Processing jobs",
            unit="job",
            position=0,
            leave=True
        )
        
        # Process jobs with small stagger to avoid rate limiting
        tasks = []
        for i, job in enumerate(jobs):
            # Add small delay between task creation for different models to avoid 429 errors
            if i > 0 and i < 10:  # Only stagger first 10 to avoid slowdown
                await asyncio.sleep(0.1)  # 100ms stagger
            task = self._process_job_with_progress(job, progress_bar)
            tasks.append(task)
        
        # Wait for all jobs to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        progress_bar.close()
        
        # Handle any exceptions in results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Job {jobs[i].id} failed with exception: {result}")
                jobs[i].status = 'failed'
                jobs[i].error = str(result)
        
        # Calculate results
        duration = time.time() - start_time
        successful = sum(1 for job in jobs if job.status == 'completed')
        failed = sum(1 for job in jobs if job.status == 'failed')
        
        batch_result = BatchResult(
            total_jobs=len(jobs),
            successful=successful,
            failed=failed,
            duration=duration,
            results=jobs,
            stats=self.stats
        )
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed in {duration:.1f}s")
        
        # Save final results
        if save_intermediate:
            try:
                await self._save_results(batch_result)
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
        
        return batch_result
    
    async def _process_job_with_progress(self, job: BatchJob, progress_bar: tqdm):
        """Process a single job and update progress"""
        try:
            result = await self._process_single_job(job)
            progress_bar.update(1)
            
            # Update progress description
            status_emoji = "✅" if job.status == 'completed' else "❌"
            progress_bar.set_postfix({
                'status': f"{status_emoji} {job.status}",
                'model': job.model[:10]  # Truncate long model names
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing job {job.id}: {e}")
            job.status = 'failed'
            job.error = str(e)
            progress_bar.update(1)
            return job
    
    async def _process_single_job(self, job: BatchJob) -> BatchJob:
        """Process a single job with retries and proper error handling"""
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    job.attempts = attempt + 1
                    job.status = 'processing'
                    
                    # Get provider for model
                    provider = self.provider_factory.get_provider(job.model)
                    
                    # Make API call - detect if this is an image generation model
                    start_time = time.time()
                    if job.model.endswith('-img') or 'img' in job.model:
                        # Image generation
                        response = await provider.generate_image(job.prompt, job.model)
                    else:
                        # Text completion
                        response = await provider.complete(job.prompt, job.model)
                    api_time = time.time() - start_time
                    
                    # Update job
                    job.status = 'completed'
                    job.result = response
                    job.completed_at = datetime.now()
                    
                    # Update stats
                    await self._update_stats(job, provider, api_time)
                    
                    # Call progress callback
                    if self.progress_callback:
                        try:
                            # Check if callback is async
                            import inspect
                            if inspect.iscoroutinefunction(self.progress_callback):
                                await self.progress_callback(job)
                            else:
                                self.progress_callback(job)
                        except Exception as e:
                            logger.debug(f"Progress callback: {e}")
                    
                    logger.debug(f"Job {job.id} completed successfully in {api_time:.2f}s")
                    return job
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for job {job.id}: {error_msg}")
                    job.error = error_msg
                    
                    # Track error types
                    error_type = type(e).__name__
                    if error_type not in self.stats['errors']:
                        self.stats['errors'][error_type] = 0
                    self.stats['errors'][error_type] += 1
                    
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        delay = self.retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        job.status = 'failed'
                        job.completed_at = datetime.now()
                        logger.error(f"Job {job.id} failed permanently after {self.max_retries} attempts")
            
            return job
    
    async def _update_stats(self, job: BatchJob, provider: Any, api_time: float):
        """Update processing statistics"""
        try:
            # Estimate tokens (provider should implement this)
            prompt_tokens = getattr(provider, 'estimate_tokens', lambda x: len(x.split()))(job.prompt)
            response_tokens = getattr(provider, 'estimate_tokens', lambda x: len(x.split()))(job.result or "")
            total_tokens = prompt_tokens + response_tokens
            
            # Estimate cost (provider should implement this)
            # Try to call estimate_cost if it exists, otherwise use default
            if hasattr(provider, 'estimate_cost'):
                try:
                    cost = provider.estimate_cost(total_tokens)
                except TypeError:
                    # Method might expect different parameters
                    cost = total_tokens * 0.0001
            else:
                cost = total_tokens * 0.0001
            
            self.stats['total_tokens'] += total_tokens
            self.stats['total_cost'] += cost
            self.stats['api_calls'] += 1
            
            # Update model-specific stats
            if job.model not in self.stats['by_model']:
                self.stats['by_model'][job.model] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost': 0,
                    'avg_time': 0,
                    'success_rate': 0
                }
            
            model_stats = self.stats['by_model'][job.model]
            model_stats['calls'] += 1
            model_stats['tokens'] += total_tokens
            model_stats['cost'] += cost
            
            # Update average time
            prev_avg = model_stats['avg_time']
            model_stats['avg_time'] = (prev_avg * (model_stats['calls'] - 1) + api_time) / model_stats['calls']
            
            # Update success rate (for completed jobs only)
            if job.status == 'completed':
                successful_calls = sum(1 for j in [job] if j.status == 'completed')  # This is always 1 for completed jobs
                model_stats['success_rate'] = successful_calls / model_stats['calls']
                
        except Exception as e:
            logger.warning(f"Failed to update stats for job {job.id}: {e}")
    
    async def _save_results(self, result: BatchResult):
        """Save results to file with robust JSON serialization"""
        output_dir = Path(self.config.get('output_dir', './staging'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'batch_results_{timestamp}.json'
        
        try:
            # Use the custom encoder for serialization
            json_data = json.dumps(result.to_dict(), indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
            
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(json_data)
            
            logger.info(f"Results saved to {output_file}")
            
            # Also save a summary file
            summary_file = output_dir / f'batch_summary_{timestamp}.json'
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'total_jobs': result.total_jobs,
                'successful': result.successful,
                'failed': result.failed,
                'duration': result.duration,
                'stats': result.stats
            }
            
            async with aiofiles.open(summary_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(summary_data, indent=2, cls=CustomJSONEncoder))
            
            logger.debug(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
            
            # Try to save a minimal version without full results
            try:
                minimal_data = {
                    'error': 'Failed to serialize full results',
                    'summary': {
                        'total_jobs': result.total_jobs,
                        'successful': result.successful,
                        'failed': result.failed,
                        'duration': result.duration
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                minimal_file = output_dir / f'batch_minimal_{timestamp}.json'
                async with aiofiles.open(minimal_file, 'w') as f:
                    await f.write(json.dumps(minimal_data, indent=2))
                
                logger.info(f"Minimal results saved to {minimal_file}")
                
            except Exception as inner_e:
                logger.error(f"Failed to save even minimal results: {inner_e}")
    
    def estimate_time(self, num_jobs: int, avg_job_time: Optional[float] = None) -> float:
        """Estimate time to process given number of jobs"""
        if avg_job_time is None:
            # Use historical data if available, otherwise default estimate
            if self.stats.get('by_model'):
                avg_times = [model['avg_time'] for model in self.stats['by_model'].values() if model['avg_time'] > 0]
                avg_job_time = sum(avg_times) / len(avg_times) if avg_times else 2.0
            else:
                avg_job_time = 2.0  # Default 2 seconds per job
        
        # Account for concurrency and overhead
        batches = (num_jobs + self.max_concurrent - 1) // self.max_concurrent
        estimated_time = batches * avg_job_time * 1.2  # Add 20% overhead
        
        return estimated_time
    
    def estimate_cost(self, num_jobs: int, avg_prompt_size: int = 1000, model: str = 'goo-2') -> float:
        """Estimate cost for processing jobs"""
        # Model-specific pricing (tokens per dollar)
        pricing = {
            'goo-1': 0.50,   # Gemini 2.5 Pro
            'goo-2': 0.35,   # Gemini 2.5 Flash
            'goo-3': 0.25,   # Gemini 2.5 Flash-Lite 
            'ant-1': 75.0,   # Claude 4Opus
            'ant-2': 15.0,    # Claude 4Sonnet
            'gpt-5': 40.0,   # GPT-5
            'gpt-5-mini': 20.0,   # GPT-5 Mini
            'gpt-5-nano': 10.0,   # GPT-5 Nano
        }
        
        # Default pricing if model not found
        cost_per_million = pricing.get(model, 1.0)
        
        # Estimate tokens (prompt + response)
        tokens_per_job = avg_prompt_size * 2  # Assume response ~= prompt size
        total_tokens = num_jobs * tokens_per_job
        
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million
        return estimated_cost
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up BatchProcessor resources")
        
        if hasattr(self.provider_factory, 'cleanup'):
            await self.provider_factory.cleanup()
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of processing statistics"""
        return {
            'total_api_calls': self.stats['api_calls'],
            'total_tokens': self.stats['total_tokens'],
            'total_cost': round(self.stats['total_cost'], 4),
            'cache_hits': self.stats['cache_hits'],
            'models_used': list(self.stats['by_model'].keys()),
            'error_types': self.stats.get('errors', {}),
            'average_cost_per_call': round(self.stats['total_cost'] / max(self.stats['api_calls'], 1), 4)
        }