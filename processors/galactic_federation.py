"""
The Galactic Federation Processor - The Great Unification Part II

This is the sovereign heart of the new architecture:
- Each provider is a sovereign nation with its own army
- The Federation coordinates but does not command
- True parallel warfare across multiple dimensions

This replaces the monarchy of the single ParallelProcessor
with the federation of multiple sovereign processors.

Copyright (c) 2025 Quantum Encoding Ltd.
The Crown Jewel Algorithm, Evolved.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GalacticFederation:
    """
    The Galactic Federation - Multi-Provider Parallel Processing Architecture
    
    This is not an enhancement. This is a revolution.
    
    The transformation from sovereign monarchy to galactic federation enables:
    - Independent provider nations with sovereign worker armies
    - Provider-specific rate limiting and resource management  
    - True parallel execution across dimensional boundaries
    - 75+ concurrent operations when all nations mobilize
    
    "The age of the monarchy is over. The age of the federation has begun."
    """
    
    # The Federal Constitution - Provider Nation Configurations
    PROVIDER_NATIONS = {
        'openai': {
            'max_workers': 20,
            'rpm': 500,
            'models': ['gpt-5', 'gpt-5-nano', 'gpt-5-mini'],
            'tier': 'premium'
        },
        'anthropic': {
            'max_workers': 15,
            'rpm': 400,
            'models': ['claude-3-5-opus', 'claude-3-5-sonnet', 'claude-3-5-haiku'],
            'tier': 'premium'
        },
        'google': {
            'max_workers': 20,
            'rpm': 300,
            'models': ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite'],
            'tier': 'standard'
        },
        'gemini': {  # Alias for Google
            'max_workers': 20,
            'rpm': 300,
            'models': ['gemini-2.5-pro', 'gemini-2.5-flash'],
            'tier': 'standard'
        },
        'deepseek': {
            'max_workers': 10,
            'rpm': 60,
            'models': ['deepseek-chat', 'deepseek-reasoner'],
            'tier': 'budget'
        },
        'xai': {
            'max_workers': 10,
            'rpm': 100,
            'models': ['grok-4', 'grok-3', 'grok-3-mini'],
            'tier': 'experimental'
        }
    }
    
    def __init__(self, 
                 provider_configs: Optional[Dict[str, Dict]] = None,
                 federation_mode: str = 'balanced'):
        """
        Initialize the Galactic Federation
        
        Args:
            provider_configs: Override default nation configurations
            federation_mode: 'balanced', 'aggressive', or 'conservative'
        """
        # The Federal Council
        self.provider_nations = provider_configs or self.PROVIDER_NATIONS.copy()
        self.federation_mode = federation_mode
        
        # Sovereign rate limiters for each nation
        self.nation_limiters = {}
        for nation, config in self.provider_nations.items():
            self.nation_limiters[nation] = {
                'semaphore': asyncio.Semaphore(config['max_workers']),
                'request_times': [],
                'lock': asyncio.Lock(),
                'rpm': config['rpm'],
                'active_workers': 0,
                'total_processed': 0
            }
        
        # Federal Statistics Department
        self.federal_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'operations_by_nation': {},
            'operations_by_model': {},
            'federation_start_time': datetime.now()
        }
        
        # Calculate total federal capacity
        self.total_worker_capacity = sum(c['max_workers'] for c in self.provider_nations.values())
        self.total_rpm_capacity = sum(c['rpm'] for c in self.provider_nations.values())
        
        logger.info(f"🌌 GALACTIC FEDERATION INITIALIZED")
        logger.info(f"🌍 Nations: {len(self.provider_nations)}")
        logger.info(f"💪 Total Worker Capacity: {self.total_worker_capacity}")
        logger.info(f"⚡ Total RPM Capacity: {self.total_rpm_capacity}")
        logger.info(f"🎯 Federation Mode: {federation_mode}")
    
    async def mobilize_all_nations(self,
                                  operations: List[Dict[str, Any]],
                                  operation_handler: Callable,
                                  progress_callback: Optional[Callable] = None,
                                  model_filter: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Mobilize all nations of the federation for parallel warfare
        
        This is the --model all implementation: Full federation mobilization.
        
        Args:
            operations: Base operations (prompts) to execute
            operation_handler: Handler function for each operation
            progress_callback: Real-time progress tracking
            model_filter: 'all', specific models, or model groups
            
        Returns:
            Comprehensive federal report of the campaign
        """
        start_time = datetime.now()
        campaign_id = f"federation_campaign_{int(start_time.timestamp())}"
        
        # Determine which models to mobilize
        models_to_deploy = self._resolve_model_deployment(model_filter)
        
        # Calculate scale of operations
        total_operations = len(operations) * len(models_to_deploy)
        
        logger.info(f"🌟 FEDERAL MOBILIZATION: {campaign_id}")
        logger.info(f"📊 Campaign Scale: {len(operations)} operations × {len(models_to_deploy)} models")
        logger.info(f"💥 Total Operations: {total_operations}")
        logger.info(f"🚀 Mobilizing {len(self.provider_nations)} sovereign nations!")
        
        # Create nation-specific operation queues
        nation_deployments = self._create_nation_deployments(operations, models_to_deploy)
        
        # Federal progress tracker
        federal_progress = {
            'total': total_operations,
            'completed': 0,
            'by_nation': {nation: 0 for nation in self.provider_nations}
        }
        
        async def nation_progress_handler(nation: str, progress: dict):
            """Handle progress from individual nations"""
            federal_progress['completed'] += 1
            federal_progress['by_nation'][nation] += 1
            
            if progress_callback:
                await progress_callback({
                    'campaign_id': campaign_id,
                    'total_operations': total_operations,
                    'completed': federal_progress['completed'],
                    'progress_percent': (federal_progress['completed'] / total_operations * 100),
                    'nation_progress': federal_progress['by_nation'],
                    'current_nation': nation,
                    'current_operation': progress
                })
        
        # Launch sovereign nation campaigns in parallel
        nation_campaigns = []
        for nation, deployments in nation_deployments.items():
            if deployments:
                campaign = self._execute_nation_campaign(
                    nation=nation,
                    operations=deployments,
                    operation_handler=operation_handler,
                    progress_handler=lambda p, n=nation: nation_progress_handler(n, p),
                    campaign_id=campaign_id
                )
                nation_campaigns.append(campaign)
        
        # Execute all nation campaigns simultaneously
        logger.info(f"⚔️ LAUNCHING {len(nation_campaigns)} PARALLEL NATION CAMPAIGNS!")
        nation_results = await asyncio.gather(*nation_campaigns, return_exceptions=True)
        
        # Compile federal report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Aggregate results
        all_results = []
        successful_total = 0
        failed_total = 0
        nation_reports = {}
        
        for idx, result in enumerate(nation_results):
            if isinstance(result, dict):
                nation_name = result.get('nation', f'nation_{idx}')
                nation_reports[nation_name] = result
                all_results.extend(result.get('results', []))
                successful_total += result.get('successful', 0)
                failed_total += result.get('failed', 0)
            else:
                logger.error(f"Nation campaign failed: {result}")
                failed_total += len(operations)
        
        # Update federal statistics
        self.federal_stats['total_operations'] += total_operations
        self.federal_stats['successful_operations'] += successful_total
        self.federal_stats['failed_operations'] += failed_total
        
        # The Federal Report
        federal_report = {
            'campaign_id': campaign_id,
            'federation_mode': self.federation_mode,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_operations': total_operations,
            'successful_operations': successful_total,
            'failed_operations': failed_total,
            'success_rate': successful_total / total_operations if total_operations > 0 else 0,
            'throughput_per_second': total_operations / duration if duration > 0 else 0,
            'nations_mobilized': len(nation_campaigns),
            'total_worker_capacity': self.total_worker_capacity,
            'models_deployed': models_to_deploy,
            'nation_reports': nation_reports,
            'results_by_nation': self._group_by_nation(all_results),
            'results_by_model': self._group_by_model(all_results),
            'federal_statistics': self.federal_stats.copy(),
            'all_results': all_results
        }
        
        # Victory announcement
        logger.info(f"🎊 FEDERAL CAMPAIGN COMPLETE!")
        logger.info(f"✨ Success Rate: {federal_report['success_rate']*100:.1f}%")
        logger.info(f"⚡ Throughput: {federal_report['throughput_per_second']:.1f} ops/sec")
        logger.info(f"🌍 Nations Reporting:")
        for nation, report in nation_reports.items():
            logger.info(f"  {nation}: {report.get('successful', 0)}/{report.get('total', 0)} successful")
        
        return federal_report
    
    async def _execute_nation_campaign(self,
                                      nation: str,
                                      operations: List[Dict],
                                      operation_handler: Callable,
                                      progress_handler: Optional[Callable],
                                      campaign_id: str) -> Dict[str, Any]:
        """Execute a sovereign nation's campaign with its own army"""
        
        nation_config = self.provider_nations[nation]
        nation_limiter = self.nation_limiters[nation]
        semaphore = nation_limiter['semaphore']
        
        results = []
        successful = 0
        failed = 0
        
        logger.info(f"🌍 {nation.upper()} mobilizing {nation_config['max_workers']} workers for {len(operations)} operations")
        
        async def execute_operation(operation):
            """Execute single operation with nation's sovereign rate limiting"""
            async with semaphore:
                nation_limiter['active_workers'] += 1
                
                try:
                    # Enforce nation's rate limit
                    await self._enforce_nation_rate_limit(nation)
                    
                    # Execute the operation
                    result = await operation_handler(operation, operation.get('model'))
                    
                    results.append({
                        'status': 'success',
                        'nation': nation,
                        'model': operation.get('model'),
                        'operation': operation,
                        'result': result
                    })
                    
                    nonlocal successful
                    successful += 1
                    
                except Exception as e:
                    logger.error(f"{nation} operation failed: {e}")
                    results.append({
                        'status': 'error',
                        'nation': nation,
                        'model': operation.get('model'),
                        'operation': operation,
                        'error': str(e)
                    })
                    
                    nonlocal failed
                    failed += 1
                
                finally:
                    nation_limiter['active_workers'] -= 1
                    nation_limiter['total_processed'] += 1
                
                # Progress callback
                if progress_handler:
                    await progress_handler({
                        'nation': nation,
                        'completed': successful + failed,
                        'total': len(operations),
                        'successful': successful,
                        'failed': failed
                    })
        
        # Launch all operations for this nation
        nation_tasks = [execute_operation(op) for op in operations]
        await asyncio.gather(*nation_tasks, return_exceptions=True)
        
        logger.info(f"✅ {nation.upper()} campaign complete: {successful}/{len(operations)} successful")
        
        return {
            'nation': nation,
            'campaign_id': campaign_id,
            'results': results,
            'successful': successful,
            'failed': failed,
            'total': len(operations),
            'workers_used': nation_config['max_workers'],
            'rpm_limit': nation_config['rpm']
        }
    
    async def _enforce_nation_rate_limit(self, nation: str):
        """Enforce sovereign rate limiting for a specific nation"""
        limiter = self.nation_limiters[nation]
        
        async with limiter['lock']:
            now = datetime.now()
            request_times = limiter['request_times']
            rpm = limiter['rpm']
            
            # Remove requests older than 1 minute
            request_times[:] = [t for t in request_times if (now - t).total_seconds() < 60]
            
            # Check if at limit
            if len(request_times) >= rpm:
                oldest = request_times[0]
                wait_time = 60 - (now - oldest).total_seconds()
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    now = datetime.now()
                    request_times[:] = [t for t in request_times if (now - t).total_seconds() < 60]
            
            request_times.append(now)
    
    def _resolve_model_deployment(self, model_filter: Optional[Union[str, List[str]]]) -> List[str]:
        """Resolve which models to deploy based on filter"""
        if model_filter is None or model_filter == 'all':
            # Deploy all models from all nations
            all_models = []
            for config in self.provider_nations.values():
                all_models.extend(config.get('models', []))
            return list(set(all_models))  # Remove duplicates
        
        elif isinstance(model_filter, list):
            return model_filter
        
        elif isinstance(model_filter, str):
            # Check if it's a group
            if model_filter.startswith('grp-'):
                # This would load from config
                return self._load_model_group(model_filter)
            else:
                return [model_filter]
        
        return []
    
    def _create_nation_deployments(self, operations: List[Dict], models: List[str]) -> Dict[str, List]:
        """Create deployment orders for each nation based on their models"""
        deployments = {nation: [] for nation in self.provider_nations}
        
        for nation, config in self.provider_nations.items():
            nation_models = config.get('models', [])
            
            for model in models:
                if model in nation_models:
                    for operation in operations:
                        deployment = operation.copy()
                        deployment['model'] = model
                        deployment['provider'] = nation
                        deployments[nation].append(deployment)
        
        return deployments
    
    def _load_model_group(self, group_name: str) -> List[str]:
        """Load model group from configuration"""
        # This would load from providers.yaml
        groups = {
            'grp-fast': ['gpt-5-nano', 'gemini-2.5-flash', 'grok-3-mini', 'deepseek-chat'],
            'grp-quality': ['gpt-5', 'claude-3-5-opus', 'gemini-2.5-pro', 'grok-4'],
            'grp-budget': ['deepseek-chat', 'gemini-2.5-flash-lite', 'gpt-5-nano']
        }
        return groups.get(group_name, [])
    
    def _group_by_nation(self, results: List[Dict]) -> Dict[str, List]:
        """Group results by nation for federal analysis"""
        grouped = {}
        for result in results:
            nation = result.get('nation', 'unknown')
            if nation not in grouped:
                grouped[nation] = []
            grouped[nation].append(result)
        return grouped
    
    def _group_by_model(self, results: List[Dict]) -> Dict[str, List]:
        """Group results by model for federal analysis"""
        grouped = {}
        for result in results:
            model = result.get('model', 'unknown')
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(result)
        return grouped
    
    def get_federal_status(self) -> Dict[str, Any]:
        """Get current status of the federation"""
        status = {
            'federation_uptime': (datetime.now() - self.federal_stats['federation_start_time']).total_seconds(),
            'total_operations_processed': self.federal_stats['total_operations'],
            'success_rate': (
                self.federal_stats['successful_operations'] / 
                self.federal_stats['total_operations']
                if self.federal_stats['total_operations'] > 0 else 0
            ),
            'nations': {}
        }
        
        for nation, limiter in self.nation_limiters.items():
            config = self.provider_nations[nation]
            status['nations'][nation] = {
                'max_workers': config['max_workers'],
                'rpm': config['rpm'],
                'active_workers': limiter['active_workers'],
                'total_processed': limiter['total_processed'],
                'models': config.get('models', [])
            }
        
        return status