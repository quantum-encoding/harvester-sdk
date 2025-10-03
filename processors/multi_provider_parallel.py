"""
Multi-Provider Parallel Processor - The Ultimate Scaling Algorithm

This extends the Crown Jewel to achieve provider-parallel execution:
- Spawn separate worker pools per provider
- Execute 100+ concurrent operations across multiple AI providers
- Maintain provider-specific rate limits
- Aggregate results across all providers

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiProviderParallelProcessor:
    """
    Provider-Parallel Processing at Unprecedented Scale
    
    This algorithm enables:
    - Separate worker pools for each provider
    - Provider-specific rate limiting
    - Aggregate processing across all available models
    - 100+ concurrent operations when using --model all
    """
    
    def __init__(self, provider_configs: Optional[Dict[str, Dict]] = None):
        """
        Initialize multi-provider parallel processor
        
        Args:
            provider_configs: Dict mapping provider names to their configurations
                {
                    'openai': {'max_workers': 20, 'rpm': 500},
                    'anthropic': {'max_workers': 15, 'rpm': 400},
                    'google': {'max_workers': 20, 'rpm': 300},
                    'deepseek': {'max_workers': 10, 'rpm': 60},
                    'xai': {'max_workers': 10, 'rpm': 100}
                }
        """
        # Default provider configurations
        self.provider_configs = provider_configs or {
            'openai': {'max_workers': 20, 'rpm': 500, 'models': ['gpt-5', 'gpt-5-nano', 'gpt-5-mini']},
            'anthropic': {'max_workers': 15, 'rpm': 400, 'models': ['claude-3-5-opus', 'claude-3-5-sonnet']},
            'google': {'max_workers': 20, 'rpm': 300, 'models': ['gemini-2.5-pro', 'gemini-2.5-flash']},
            'gemini': {'max_workers': 20, 'rpm': 300, 'models': ['gemini-2.5-pro', 'gemini-2.5-flash']},
            'deepseek': {'max_workers': 10, 'rpm': 60, 'models': ['deepseek-chat', 'deepseek-reasoner']},
            'xai': {'max_workers': 10, 'rpm': 100, 'models': ['grok-4', 'grok-3', 'grok-3-mini']}
        }
        
        # Create provider-specific rate limiters
        self.rate_limiters = {}
        for provider, config in self.provider_configs.items():
            self.rate_limiters[provider] = {
                'semaphore': asyncio.Semaphore(config['max_workers']),
                'request_times': [],
                'lock': asyncio.Lock(),
                'rpm': config['rpm']
            }
        
        # Global statistics
        self.global_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'operations_by_provider': {},
            'operations_by_model': {}
        }
        
        logger.info(f"🚀 Multi-Provider Processor initialized with {len(self.provider_configs)} providers")
        total_workers = sum(c['max_workers'] for c in self.provider_configs.values())
        logger.info(f"💪 Total parallel capacity: {total_workers} workers across all providers")
    
    async def execute_all_providers(self,
                                   operations: List[Dict[str, Any]],
                                   operation_handler: Callable,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute operations across ALL providers in parallel
        
        This is the ultimate scaling: if you have 20 prompts and 5 providers,
        this creates 100 parallel operations distributed across provider-specific worker pools.
        
        Args:
            operations: List of base operations (prompts)
            operation_handler: Handler that accepts (operation, model) parameters
            progress_callback: Optional progress tracking
            
        Returns:
            Comprehensive results from all providers
        """
        start_time = datetime.now()
        batch_id = f"multi_provider_batch_{int(start_time.timestamp())}"
        
        # Calculate total operations
        total_models = sum(len(config.get('models', [])) for config in self.provider_configs.values())
        total_operations = len(operations) * total_models
        
        logger.info(f"🌟 MEGA BATCH: {batch_id}")
        logger.info(f"📊 Scale: {len(operations)} prompts × {total_models} models = {total_operations} operations")
        logger.info(f"🔥 Launching across {len(self.provider_configs)} providers simultaneously!")
        
        # Create provider-specific operation queues
        provider_operations = {}
        for provider, config in self.provider_configs.items():
            provider_operations[provider] = []
            for model in config.get('models', []):
                for operation in operations:
                    # Clone operation and add model/provider info
                    op = operation.copy()
                    op['model'] = model
                    op['provider'] = provider
                    provider_operations[provider].append(op)
        
        # Launch provider-parallel execution
        provider_tasks = []
        for provider, provider_ops in provider_operations.items():
            if provider_ops:
                task = self._execute_provider_batch(
                    provider=provider,
                    operations=provider_ops,
                    operation_handler=operation_handler,
                    progress_callback=progress_callback,
                    batch_id=batch_id
                )
                provider_tasks.append(task)
        
        # Execute all providers in parallel
        logger.info(f"⚡ Executing {total_operations} operations across {len(provider_tasks)} provider pools...")
        provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)
        
        # Aggregate results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        all_results = []
        total_successful = 0
        total_failed = 0
        
        for provider_result in provider_results:
            if isinstance(provider_result, dict):
                all_results.extend(provider_result.get('results', []))
                total_successful += provider_result.get('successful', 0)
                total_failed += provider_result.get('failed', 0)
            else:
                logger.error(f"Provider batch failed: {provider_result}")
                total_failed += len(operations)
        
        # Comprehensive result package
        return {
            'batch_id': batch_id,
            'execution_mode': 'multi_provider_parallel',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_operations': total_operations,
            'successful_operations': total_successful,
            'failed_operations': total_failed,
            'success_rate': total_successful / total_operations if total_operations > 0 else 0,
            'throughput_per_second': total_operations / duration if duration > 0 else 0,
            'providers_used': list(self.provider_configs.keys()),
            'total_workers': sum(c['max_workers'] for c in self.provider_configs.values()),
            'results_by_provider': self._group_results_by_provider(all_results),
            'results_by_model': self._group_results_by_model(all_results),
            'all_results': all_results,
            'statistics': self.global_stats
        }
    
    async def _execute_provider_batch(self,
                                     provider: str,
                                     operations: List[Dict],
                                     operation_handler: Callable,
                                     progress_callback: Optional[Callable],
                                     batch_id: str) -> Dict[str, Any]:
        """Execute operations for a specific provider with its own worker pool"""
        
        rate_limiter = self.rate_limiters[provider]
        semaphore = rate_limiter['semaphore']
        results = []
        successful = 0
        failed = 0
        
        async def process_single(operation):
            """Process single operation with provider-specific rate limiting"""
            async with semaphore:
                # Enforce provider-specific rate limit
                await self._enforce_provider_rate_limit(provider)
                
                try:
                    # Execute operation
                    result = await operation_handler(operation, operation['model'])
                    
                    results.append({
                        'status': 'success',
                        'provider': provider,
                        'model': operation['model'],
                        'operation': operation,
                        'result': result
                    })
                    
                    nonlocal successful
                    successful += 1
                    
                except Exception as e:
                    logger.error(f"Operation failed for {provider}/{operation['model']}: {e}")
                    results.append({
                        'status': 'error',
                        'provider': provider,
                        'model': operation['model'],
                        'operation': operation,
                        'error': str(e)
                    })
                    
                    nonlocal failed
                    failed += 1
                
                # Progress callback
                if progress_callback:
                    await progress_callback({
                        'provider': provider,
                        'completed': successful + failed,
                        'total': len(operations),
                        'successful': successful,
                        'failed': failed
                    })
        
        # Launch all operations for this provider
        provider_coroutines = [process_single(op) for op in operations]
        await asyncio.gather(*provider_coroutines, return_exceptions=True)
        
        logger.info(f"✅ {provider} complete: {successful}/{len(operations)} successful")
        
        return {
            'provider': provider,
            'results': results,
            'successful': successful,
            'failed': failed,
            'total': len(operations)
        }
    
    async def _enforce_provider_rate_limit(self, provider: str):
        """Enforce rate limiting for specific provider"""
        rate_limiter = self.rate_limiters[provider]
        
        async with rate_limiter['lock']:
            now = datetime.now()
            request_times = rate_limiter['request_times']
            rpm = rate_limiter['rpm']
            
            # Remove requests older than 1 minute
            request_times[:] = [t for t in request_times if (now - t).total_seconds() < 60]
            
            # Check if we're at the limit
            if len(request_times) >= rpm:
                # Calculate wait time
                oldest_request = request_times[0]
                wait_time = 60 - (now - oldest_request).total_seconds()
                
                if wait_time > 0:
                    logger.debug(f"Rate limit for {provider}: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    
                    # Clean up old requests again
                    now = datetime.now()
                    request_times[:] = [t for t in request_times if (now - t).total_seconds() < 60]
            
            # Record this request
            request_times.append(now)
    
    def _group_results_by_provider(self, results: List[Dict]) -> Dict[str, List]:
        """Group results by provider for analysis"""
        grouped = {}
        for result in results:
            provider = result.get('provider', 'unknown')
            if provider not in grouped:
                grouped[provider] = []
            grouped[provider].append(result)
        return grouped
    
    def _group_results_by_model(self, results: List[Dict]) -> Dict[str, List]:
        """Group results by model for analysis"""
        grouped = {}
        for result in results:
            model = result.get('model', 'unknown')
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(result)
        return grouped
    
    def get_provider_stats(self) -> Dict[str, Dict]:
        """Get statistics for each provider"""
        stats = {}
        for provider, config in self.provider_configs.items():
            stats[provider] = {
                'max_workers': config['max_workers'],
                'rpm': config['rpm'],
                'models': config.get('models', []),
                'operations_completed': self.global_stats['operations_by_provider'].get(provider, 0)
            }
        return stats