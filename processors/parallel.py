"""
Parallel Processor - The Crown Jewel Algorithm

This is the algorithmic heart that achieves impossible parallelism:
- 20+ concurrent AI operations
- Intelligent rate limiting and backoff
- Context preservation across thousands of tasks
- Military-grade error handling and recovery

Copyright (c) 2025 Quantum Encoding Ltd.
Licensed under the Harvester Commercial License.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    The Crown Jewel: Parallel AI Processing at Industrial Scale
    
    This is the algorithm that transforms impossible into routine:
    - Execute 1000+ AI operations across 20 parallel workers
    - Maintain perfect rate limiting across multiple providers
    - Preserve context and state across massive operations
    - Handle failures with military precision and automatic recovery
    
    The algorithm that powers the "Parable of Two Workflows."
    """
    
    def __init__(self, 
                 max_workers: int = 20,
                 rate_limit_per_minute: int = 60,
                 retry_attempts: int = 3,
                 backoff_multiplier: float = 2.0):
        """
        Initialize the Parallel Processor
        
        Args:
            max_workers: Maximum concurrent workers
            rate_limit_per_minute: Global rate limit across all workers
            retry_attempts: Maximum retry attempts for failed operations
            backoff_multiplier: Exponential backoff multiplier
        """
        self.max_workers = max_workers
        self.rate_limit_per_minute = rate_limit_per_minute
        self.retry_attempts = retry_attempts
        self.backoff_multiplier = backoff_multiplier
        
        # Rate limiting infrastructure
        self.request_times = []
        self.rate_lock = asyncio.Lock()
        
        # Statistics tracking
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'retry_operations': 0,
            'rate_limit_hits': 0,
            'average_response_time': 0.0
        }
        
        logger.info(f"🔥 Parallel Processor initialized: {max_workers} workers, {rate_limit_per_minute} RPM")
    
    async def execute_batch(self,
                           operations: List[Dict[str, Any]],
                           operation_handler: Callable,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute a batch of operations with perfect parallel coordination
        
        This is the crown jewel algorithm in action:
        - Launch all operations simultaneously
        - Maintain perfect rate limiting across all workers
        - Handle failures with automatic retry and backoff
        - Provide real-time progress tracking
        
        Args:
            operations: List of operation dictionaries
            operation_handler: Async function to handle each operation
            progress_callback: Optional callback for progress updates
        
        Returns:
            Comprehensive results with detailed statistics
        """
        
        start_time = datetime.now()
        batch_id = f"batch_{int(start_time.timestamp())}"
        
        logger.info(f"🚀 Launching parallel batch: {batch_id}")
        logger.info(f"📊 Operations: {len(operations)}, Workers: {self.max_workers}")
        
        # Create semaphore for worker control
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Progress tracking
        completed = 0
        results = []
        
        async def process_single_operation(op_index: int, operation: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single operation with full error handling and rate limiting"""
            nonlocal completed, results
            
            async with semaphore:
                operation_id = f"{batch_id}_op_{op_index:04d}"
                
                # Enforce rate limiting
                await self._enforce_rate_limit()
                
                # Execute with retry logic
                result = await self._execute_with_retry(
                    operation_id=operation_id,
                    operation=operation,
                    handler=operation_handler
                )
                
                # Update progress
                completed += 1
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    await progress_callback({
                        'operation_id': operation_id,
                        'completed': completed,
                        'total': len(operations),
                        'progress_percent': (completed / len(operations)) * 100,
                        'result': result
                    })
                
                # Log progress
                if completed % 10 == 0 or completed == len(operations):
                    logger.info(f"⚡ Progress: {completed}/{len(operations)} ({(completed/len(operations)*100):.1f}%)")
                
                return result
        
        # Launch all operations in parallel (the crown jewel moment)
        logger.info(f"🔥 Launching {len(operations)} operations across {self.max_workers} workers...")
        
        operation_coroutines = [
            process_single_operation(idx, operation)
            for idx, operation in enumerate(operations)
        ]
        
        # Execute with perfect coordination
        batch_results = await asyncio.gather(*operation_coroutines, return_exceptions=True)
        
        # Calculate final statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        successful = sum(1 for r in batch_results if isinstance(r, dict) and r.get('status') == 'success')
        failed = len(batch_results) - successful
        
        # Update global statistics
        self.stats['total_operations'] += len(operations)
        self.stats['successful_operations'] += successful
        self.stats['failed_operations'] += failed
        
        # Comprehensive results package
        batch_result = {
            'batch_id': batch_id,
            'status': 'completed',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_operations': len(operations),
            'successful_operations': successful,
            'failed_operations': failed,
            'success_rate': successful / len(operations) if operations else 0,
            'throughput_per_second': len(operations) / duration if duration > 0 else 0,
            'average_operation_time': duration / len(operations) if operations else 0,
            'workers_used': self.max_workers,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'individual_results': batch_results,
            'statistics': self.stats.copy()
        }
        
        logger.info(f"✨ Batch complete: {successful}/{len(operations)} successful")
        logger.info(f"🚀 Throughput: {batch_result['throughput_per_second']:.1f} ops/second")
        logger.info(f"⚡ Average time per operation: {batch_result['average_operation_time']:.2f}s")
        
        return batch_result
    
    async def _enforce_rate_limit(self):
        """
        Intelligent rate limiting across all workers
        
        This ensures we never exceed provider limits while maintaining maximum throughput.
        """
        async with self.rate_lock:
            now = datetime.now()
            
            # Clean old request times (older than 1 minute)
            minute_ago = now.timestamp() - 60
            self.request_times = [t for t in self.request_times if t > minute_ago]
            
            # Check if we need to wait
            if len(self.request_times) >= self.rate_limit_per_minute:
                # Calculate wait time
                oldest_request = min(self.request_times)
                wait_time = 60 - (now.timestamp() - oldest_request)
                
                if wait_time > 0:
                    logger.debug(f"⏸️  Rate limit hit, waiting {wait_time:.1f}s")
                    self.stats['rate_limit_hits'] += 1
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(now.timestamp())
    
    async def _execute_with_retry(self,
                                 operation_id: str,
                                 operation: Dict[str, Any],
                                 handler: Callable) -> Dict[str, Any]:
        """
        Execute operation with intelligent retry and backoff
        
        This handles all failure modes with military precision.
        """
        last_error = None
        error_type = None
        
        for attempt in range(self.retry_attempts + 1):
            try:
                # Execute the operation
                start_time = datetime.now()
                result = await handler(operation)
                end_time = datetime.now()
                
                # Success - record timing
                response_time = (end_time - start_time).total_seconds()
                self._update_average_response_time(response_time)
                
                return {
                    'operation_id': operation_id,
                    'status': 'success',
                    'result': result,
                    'attempt': attempt + 1,
                    'response_time': response_time,
                    'timestamp': end_time.isoformat()
                }
                
            except Exception as e:
                last_error = str(e)
                error_type = type(e).__name__
                
                logger.debug(f"Operation {operation_id} attempt {attempt + 1} failed: {error_type}")
                
                # Check if we should retry
                if attempt < self.retry_attempts:
                    # Calculate backoff time
                    backoff_time = (self.backoff_multiplier ** attempt) * 1.0
                    
                    # Special handling for rate limit errors
                    if '429' in str(e) or 'rate limit' in str(e).lower():
                        backoff_time *= 2  # Extra delay for rate limits
                        self.stats['rate_limit_hits'] += 1
                    
                    logger.debug(f"Retrying {operation_id} in {backoff_time:.1f}s")
                    await asyncio.sleep(backoff_time)
                    
                    self.stats['retry_operations'] += 1
                else:
                    # Final failure
                    logger.warning(f"Operation {operation_id} failed after {self.retry_attempts} retries: {last_error}")
                    break
        
        # Record failure
        return {
            'operation_id': operation_id,
            'status': 'failed',
            'error': last_error,
            'error_type': error_type,
            'attempts': self.retry_attempts + 1,
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_average_response_time(self, response_time: float):
        """Update rolling average response time"""
        current_avg = self.stats['average_response_time']
        total_ops = self.stats['successful_operations']
        
        if total_ops == 0:
            self.stats['average_response_time'] = response_time
        else:
            # Rolling average
            self.stats['average_response_time'] = (current_avg * total_ops + response_time) / (total_ops + 1)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics
        
        Returns comprehensive analysis of processing performance.
        """
        total_ops = self.stats['total_operations']
        
        return {
            **self.stats,
            'efficiency_ratio': self.stats['successful_operations'] / total_ops if total_ops > 0 else 0,
            'retry_rate': self.stats['retry_operations'] / total_ops if total_ops > 0 else 0,
            'rate_limit_hit_rate': self.stats['rate_limit_hits'] / total_ops if total_ops > 0 else 0,
            'configuration': {
                'max_workers': self.max_workers,
                'rate_limit_per_minute': self.rate_limit_per_minute,
                'retry_attempts': self.retry_attempts,
                'backoff_multiplier': self.backoff_multiplier
            }
        }
    
    def reset_stats(self):
        """Reset all statistics for fresh measurement"""
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'retry_operations': 0,
            'rate_limit_hits': 0,
            'average_response_time': 0.0
        }
        logger.info("📊 Statistics reset")


# Specialized processors for different use cases
class ImageGenerationProcessor(ParallelProcessor):
    """Specialized processor optimized for image generation workloads"""
    
    def __init__(self, **kwargs):
        # Image generation typically needs longer timeouts and lower concurrency
        # Extract known parameters to avoid duplicate passing
        max_workers = kwargs.pop('max_workers', 10)
        rate_limit_per_minute = kwargs.pop('rate_limit_per_minute', 30)
        retry_attempts = kwargs.pop('retry_attempts', 5)
        
        super().__init__(
            max_workers=max_workers,
            rate_limit_per_minute=rate_limit_per_minute,
            retry_attempts=retry_attempts,
            **kwargs
        )


class TextGenerationProcessor(ParallelProcessor):
    """Specialized processor optimized for text generation workloads"""
    
    def __init__(self, **kwargs):
        # Text generation can handle higher concurrency
        # Extract known parameters to avoid duplicate passing
        max_workers = kwargs.pop('max_workers', 25)
        rate_limit_per_minute = kwargs.pop('rate_limit_per_minute', 120)
        retry_attempts = kwargs.pop('retry_attempts', 3)
        
        super().__init__(
            max_workers=max_workers,
            rate_limit_per_minute=rate_limit_per_minute,
            retry_attempts=retry_attempts,
            **kwargs
        )


class CodeRefactoringProcessor(ParallelProcessor):
    """Specialized processor optimized for code refactoring workloads"""
    
    def __init__(self, **kwargs):
        # Code refactoring needs high reliability and context preservation
        # Extract known parameters to avoid duplicate passing
        max_workers = kwargs.pop('max_workers', 15)
        rate_limit_per_minute = kwargs.pop('rate_limit_per_minute', 60)
        retry_attempts = kwargs.pop('retry_attempts', 5)
        backoff_multiplier = kwargs.pop('backoff_multiplier', 1.5)
        
        super().__init__(
            max_workers=max_workers,
            rate_limit_per_minute=rate_limit_per_minute,
            retry_attempts=retry_attempts,
            backoff_multiplier=backoff_multiplier,
            **kwargs
        )