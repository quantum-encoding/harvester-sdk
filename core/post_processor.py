"""
Post-Processor with Tool Calling Capability

Handles post-processing tasks after batch completion, including external tool calls,
notifications, file operations, and integration with other services.
"""
import asyncio
import subprocess
import json
import requests
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
import yaml

logger = logging.getLogger(__name__)

class PostProcessorConfig:
    """Configuration for post-processor actions"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = {}
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Default configuration
        self.default_config = {
            'enabled': True,
            'timeout': 300,  # 5 minutes
            'retry_attempts': 3,
            'retry_delay': 5,  # seconds
            'notifications': {
                'email': False,
                'webhook': False,
                'slack': False
            },
            'file_operations': {
                'cleanup_temp': True,
                'archive_results': True,
                'generate_summary': True
            },
            'tool_calls': {
                'enabled': True,
                'allowed_tools': ['curl', 'python', 'node'],
                'max_concurrent': 5
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value with fallback to defaults"""
        keys = key.split('.')
        value = self.config
        default_value = self.default_config
        
        try:
            for k in keys:
                value = value[k]
                default_value = default_value[k]
            return value
        except (KeyError, TypeError):
            try:
                for k in keys:
                    default_value = default_value[k]
                return default_value
            except (KeyError, TypeError):
                return default

class PostProcessorResult:
    """Result of post-processing operations"""
    
    def __init__(self):
        self.success = True
        self.actions_executed = []
        self.errors = []
        self.start_time = datetime.now()
        self.end_time = None
        self.metadata = {}
    
    def add_action(self, action_name: str, success: bool, details: str = ""):
        """Record an executed action"""
        self.actions_executed.append({
            'action': action_name,
            'success': success,
            'timestamp': datetime.now(),
            'details': details
        })
        
        if not success:
            self.success = False
            self.errors.append(f"{action_name}: {details}")
    
    def complete(self):
        """Mark post-processing as complete"""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'success': self.success,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'actions_executed': self.actions_executed,
            'errors': self.errors,
            'metadata': self.metadata
        }

class ImageBatchPostProcessor:
    """Post-processor for image generation batch results"""
    
    def __init__(self, config: Optional[PostProcessorConfig] = None):
        self.config = config or PostProcessorConfig()
        self.session = None
        
        # Hook registry for custom actions
        self.hooks = {
            'before_processing': [],
            'after_file_ops': [],
            'after_notifications': [],
            'after_tool_calls': [],
            'on_error': [],
            'on_complete': []
        }
    
    async def process_batch_completion(
        self,
        batch_results: Dict[str, Any],
        output_directory: Path,
        metadata: Dict[str, Any] = None
    ) -> PostProcessorResult:
        """
        Process batch completion with configured actions
        
        Args:
            batch_results: Results from the batch processing
            output_directory: Directory containing generated images/results
            metadata: Additional metadata about the batch
            
        Returns:
            PostProcessorResult with details of all actions performed
        """
        result = PostProcessorResult()
        result.metadata = metadata or {}
        
        if not self.config.get('enabled', True):
            logger.info("Post-processor disabled, skipping")
            result.complete()
            return result
        
        logger.info(f"Starting post-processing for batch in {output_directory}")
        
        try:
            # Execute hooks: before_processing
            await self._execute_hooks('before_processing', batch_results, result)
            
            # File operations
            if self.config.get('file_operations.enabled', True):
                await self._handle_file_operations(batch_results, output_directory, result)
                await self._execute_hooks('after_file_ops', batch_results, result)
            
            # Notifications
            if self.config.get('notifications.enabled', True):
                await self._send_notifications(batch_results, output_directory, result)
                await self._execute_hooks('after_notifications', batch_results, result)
            
            # Tool calls
            if self.config.get('tool_calls.enabled', True):
                await self._execute_tool_calls(batch_results, output_directory, result)
                await self._execute_hooks('after_tool_calls', batch_results, result)
            
            # Execute completion hooks
            await self._execute_hooks('on_complete', batch_results, result)
            
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            result.add_action('post_processing', False, str(e))
            await self._execute_hooks('on_error', batch_results, result)
        
        finally:
            result.complete()
            await self._cleanup()
        
        logger.info(f"Post-processing completed in {result.end_time - result.start_time}")
        return result
    
    async def _handle_file_operations(
        self,
        batch_results: Dict[str, Any],
        output_directory: Path,
        result: PostProcessorResult
    ):
        """Handle file operations like cleanup, archiving, summary generation"""
        
        # Generate summary
        if self.config.get('file_operations.generate_summary', True):
            try:
                summary_path = output_directory / 'batch_summary.json'
                summary = self._generate_batch_summary(batch_results, output_directory)
                
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                result.add_action('generate_summary', True, f"Summary saved to {summary_path}")
                
            except Exception as e:
                result.add_action('generate_summary', False, str(e))
        
        # Archive results
        if self.config.get('file_operations.archive_results', False):
            try:
                archive_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                archive_path = output_directory.parent / archive_name
                
                # Create archive (simplified - could use zipfile for more control)
                import shutil
                shutil.make_archive(str(archive_path.with_suffix('')), 'zip', output_directory)
                
                result.add_action('archive_results', True, f"Archive created: {archive_path}")
                
            except Exception as e:
                result.add_action('archive_results', False, str(e))
        
        # Cleanup temporary files
        if self.config.get('file_operations.cleanup_temp', True):
            try:
                temp_files = list(output_directory.glob('*.tmp'))
                for temp_file in temp_files:
                    temp_file.unlink()
                
                result.add_action('cleanup_temp', True, f"Cleaned {len(temp_files)} temp files")
                
            except Exception as e:
                result.add_action('cleanup_temp', False, str(e))
    
    async def _send_notifications(
        self,
        batch_results: Dict[str, Any],
        output_directory: Path,
        result: PostProcessorResult
    ):
        """Send notifications about batch completion"""
        
        # Webhook notification
        webhook_url = self.config.get('notifications.webhook_url')
        if webhook_url:
            try:
                payload = {
                    'event': 'batch_completed',
                    'timestamp': datetime.now().isoformat(),
                    'batch_id': batch_results.get('batch_id'),
                    'total_images': batch_results.get('total_successful', 0),
                    'output_directory': str(output_directory),
                    'summary': self._generate_notification_summary(batch_results)
                }
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        result.add_action('webhook_notification', True, f"Webhook sent to {webhook_url}")
                    else:
                        result.add_action('webhook_notification', False, f"HTTP {response.status}")
                        
            except Exception as e:
                result.add_action('webhook_notification', False, str(e))
        
        # Email notification (if configured)
        email_config = self.config.get('notifications.email')
        if email_config and email_config.get('enabled'):
            try:
                await self._send_email_notification(batch_results, output_directory, email_config)
                result.add_action('email_notification', True, "Email sent successfully")
            except Exception as e:
                result.add_action('email_notification', False, str(e))
    
    async def _execute_tool_calls(
        self,
        batch_results: Dict[str, Any],
        output_directory: Path,
        result: PostProcessorResult
    ):
        """Execute configured tool calls"""
        
        tool_calls = self.config.get('tool_calls.commands', [])
        if not tool_calls:
            return
        
        max_concurrent = self.config.get('tool_calls.max_concurrent', 5)
        timeout = self.config.get('timeout', 300)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Prepare context variables for tool calls
        context = {
            'batch_id': batch_results.get('batch_id', 'unknown'),
            'output_dir': str(output_directory),
            'total_images': batch_results.get('total_successful', 0),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        async def execute_tool_call(tool_config):
            async with semaphore:
                return await self._execute_single_tool_call(tool_config, context, timeout)
        
        # Execute all tool calls concurrently
        tasks = [execute_tool_call(tool_config) for tool_config in tool_calls]
        tool_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, tool_result in enumerate(tool_results):
            tool_name = tool_calls[i].get('name', f'tool_{i}')
            
            if isinstance(tool_result, Exception):
                result.add_action(f'tool_call_{tool_name}', False, str(tool_result))
            else:
                success, details = tool_result
                result.add_action(f'tool_call_{tool_name}', success, details)
    
    async def _execute_single_tool_call(
        self,
        tool_config: Dict[str, Any],
        context: Dict[str, str],
        timeout: int
    ) -> tuple[bool, str]:
        """Execute a single tool call"""
        
        command = tool_config.get('command', '')
        if not command:
            return False, "No command specified"
        
        # Substitute context variables in command
        for key, value in context.items():
            command = command.replace(f'{{{key}}}', str(value))
        
        try:
            # Check if tool is allowed
            allowed_tools = self.config.get('tool_calls.allowed_tools', [])
            tool_name = command.split()[0]
            
            if allowed_tools and tool_name not in allowed_tools:
                return False, f"Tool '{tool_name}' not in allowed list: {allowed_tools}"
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                if process.returncode == 0:
                    return True, f"Command executed successfully: {stdout.decode()[:200]}"
                else:
                    return False, f"Command failed (exit {process.returncode}): {stderr.decode()[:200]}"
                    
            except asyncio.TimeoutError:
                process.kill()
                return False, f"Command timed out after {timeout} seconds"
                
        except Exception as e:
            return False, f"Tool execution error: {e}"
    
    async def _execute_hooks(
        self,
        hook_name: str,
        batch_results: Dict[str, Any],
        result: PostProcessorResult
    ):
        """Execute registered hooks"""
        hooks = self.hooks.get(hook_name, [])
        
        for hook_func in hooks:
            try:
                if asyncio.iscoroutinefunction(hook_func):
                    await hook_func(batch_results, result)
                else:
                    hook_func(batch_results, result)
                    
                result.add_action(f'hook_{hook_name}', True, f"Hook {hook_func.__name__} executed")
                
            except Exception as e:
                result.add_action(f'hook_{hook_name}', False, f"Hook {hook_func.__name__} failed: {e}")
    
    def register_hook(self, hook_name: str, func: Callable):
        """Register a custom hook function"""
        if hook_name in self.hooks:
            self.hooks[hook_name].append(func)
        else:
            logger.warning(f"Unknown hook name: {hook_name}")
    
    def _generate_batch_summary(self, batch_results: Dict[str, Any], output_directory: Path) -> Dict[str, Any]:
        """Generate comprehensive batch summary"""
        return {
            'batch_id': batch_results.get('batch_id'),
            'completed_at': datetime.now().isoformat(),
            'output_directory': str(output_directory),
            'statistics': {
                'total_requested': batch_results.get('total_requested', 0),
                'total_successful': batch_results.get('total_successful', 0),
                'total_failed': batch_results.get('total_failed', 0),
                'success_rate': batch_results.get('success_rate', 0),
                'total_cost': batch_results.get('total_cost', 0),
                'processing_time': batch_results.get('processing_time', 0)
            },
            'models_used': batch_results.get('models_used', []),
            'file_locations': {
                'images': list(output_directory.glob('*.png')) + list(output_directory.glob('*.jpg')),
                'metadata': list(output_directory.glob('*.json')),
                'logs': list(output_directory.glob('*.log'))
            }
        }
    
    def _generate_notification_summary(self, batch_results: Dict[str, Any]) -> str:
        """Generate short summary for notifications"""
        total = batch_results.get('total_successful', 0)
        failed = batch_results.get('total_failed', 0)
        cost = batch_results.get('total_cost', 0)
        
        return f"Batch completed: {total} images generated, {failed} failed, ${cost:.2f} total cost"
    
    async def _send_email_notification(
        self,
        batch_results: Dict[str, Any],
        output_directory: Path,
        email_config: Dict[str, Any]
    ):
        """Send email notification (placeholder - implement with your email service)"""
        # This would integrate with your email service (SendGrid, AWS SES, etc.)
        logger.info("Email notification would be sent here")
        pass
    
    async def _cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None


# Utility function to create post-processor from config file
def create_post_processor(config_path: Optional[Path] = None) -> ImageBatchPostProcessor:
    """Create post-processor with configuration"""
    config = PostProcessorConfig(config_path)
    return ImageBatchPostProcessor(config)