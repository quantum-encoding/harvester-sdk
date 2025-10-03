"""
Harvester SDK - Complete AI Processing Platform

Commercial interface to the enterprise Harvesting Engine.
Imports and delegates to the proven, battle-tested core components.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Import the proven engine components
sys.path.append(str(Path(__file__).parent.parent))
from providers.provider_factory import ProviderFactory
from core.batch_processor import BatchProcessor, BatchJob, BatchResult
from utils.output_manager import OutputManager as EngineOutputManager

# Import structured output for Premium tier features
try:
    from core.structured_output import (
        StructuredOutputProcessor, 
        StructuredResponse, 
        StructuredOutputError
    )
    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError:
    STRUCTURED_OUTPUT_AVAILABLE = False
    StructuredOutputProcessor = None
    StructuredResponse = None
    StructuredOutputError = Exception

# Import function calling for Professional+ tier features
try:
    from core.function_calling import (
        FunctionRegistry,
        FunctionCall,
        FunctionResult,
        get_function_registry
    )
    FUNCTION_CALLING_AVAILABLE = True
except ImportError:
    FUNCTION_CALLING_AVAILABLE = False
    FunctionRegistry = None
    FunctionCall = None
    FunctionResult = None
    get_function_registry = lambda: None


class HarvesterLicense:
    """License validation and tier management for commercial SDK"""
    
    def __init__(self, license_key: Optional[str] = None):
        self.license_key = license_key or os.environ.get('HARVESTER_LICENSE_KEY')
        self.tier = self._determine_tier()
        self.is_valid = True  # Always valid for now
    
    def _determine_tier(self) -> str:
        """Determine license tier from key format"""
        if not self.license_key:
            return "freemium"
        
        if self.license_key.startswith("HSK-ENT-"):
            return "enterprise"
        elif self.license_key.startswith("HSK-PRO-"):
            return "professional"
        elif self.license_key.startswith("HSK-PRE-"):
            return "premium"
        else:
            return "freemium"
    
    def get_max_concurrent(self, requested: int = 20) -> int:
        """Get maximum concurrent workers based on tier"""
        limits = {
            "freemium": 5,
            "professional": 25, 
            "premium": 100,
            "enterprise": requested  # No limit
        }
        return min(requested, limits.get(self.tier, 5))
    
    def get_max_batch_size(self, requested: int = 1000) -> int:
        """Get maximum batch size based on tier"""
        limits = {
            "freemium": 10,
            "professional": 100,
            "premium": 10000,
            "enterprise": requested  # No limit
        }
        return min(requested, limits.get(self.tier, 10))


class HarvesterSDK:
    """
    Harvester SDK: Commercial interface for the enterprise Harvesting Engine
    
    Delegates all operations to the proven, battle-tested engine components:
    - ProviderFactory for robust provider management
    - BatchProcessor for parallel processing with exponential backoff  
    - OutputManager for intelligent file organization
    """
    
    def __init__(
        self, 
        license_key: Optional[str] = None,
        max_concurrent: int = 20,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize Harvester SDK
        
        Args:
            license_key: Your Harvester license key (or set HARVESTER_LICENSE_KEY env var)
            max_concurrent: Maximum parallel workers (limited by license)
            output_dir: Custom output directory (defaults to ~/harvest_sdk/)
        """
        # License management
        self.license = HarvesterLicense(license_key)
        self.max_concurrent = self.license.get_max_concurrent(max_concurrent)
        
        # Initialize the proven engine components
        self.provider_factory = ProviderFactory()
        
        # Configure batch processor with license-enforced limits
        processor_config = {
            'max_concurrent': self.max_concurrent,
            'max_retries': 3,
            'retry_delay': 1.0,
            'output_dir': output_dir or Path.home() / 'harvest_sdk'
        }
        self.batch_processor = BatchProcessor(self.provider_factory, processor_config)
        
        # Output manager for organized saves  
        self.output_dir = output_dir or Path.home() / 'harvest_sdk'
        self.output_manager = EngineOutputManager(base_output_dir=self.output_dir)
        
        # Initialize structured output processor (Premium feature)
        self.structured_processor = None
        if STRUCTURED_OUTPUT_AVAILABLE:
            self.structured_processor = StructuredOutputProcessor()
        
        # Initialize function registry (Professional+ feature)
        self.function_registry = None
        if FUNCTION_CALLING_AVAILABLE:
            self.function_registry = get_function_registry()
    
    def _get_processor_for_operation(self, models: Union[str, List[str]]):
        """
        Get the appropriate processor based on license tier and operation type
        Uses the Divine Arbiter to enforce stratification of power
        """
        # Initialize Divine Arbiter if not already done
        if not hasattr(self, 'divine_arbiter'):
            from core.divine_arbiter import get_divine_arbiter
            self.divine_arbiter = get_divine_arbiter()
            # Ensure tier alignment
            if self.license.tier != self.divine_arbiter.current_tier:
                self.divine_arbiter.upgrade_tier(self.license.tier)
        
        # Determine operation mode
        if isinstance(models, str):
            if models.lower() == 'all':
                operation_mode = 'multi'
                models_list = None  # Will use all available
            elif ',' in models:
                operation_mode = 'multi'
                models_list = [m.strip() for m in models.split(',')]
            else:
                operation_mode = 'single'
                models_list = [models]
        else:
            models_list = models
            operation_mode = 'multi' if len(models_list) > 1 else 'single'
        
        # Summon the appropriate processor through the Divine Arbiter
        return self.divine_arbiter.summon_processor(operation_mode, models_list)
    
    async def process_batch(
        self,
        prompts: List[str],
        model_alias: str = "gpt-2",  # Default to fast model
        save_results: bool = True,
        template_name: str = "general"
    ) -> BatchResult:
        """
        Process batch of prompts using the enterprise engine
        
        Args:
            prompts: List of text prompts to process
            model_alias: Model alias (gpt-1, ant-2, goo-1, etc.)
            save_results: Whether to save results to organized output structure
            template_name: Template name for categorization
            
        Returns:
            BatchResult with all processing details and statistics
        """
        # Enforce license limits
        max_batch = self.license.get_max_batch_size(len(prompts))
        if len(prompts) > max_batch:
            raise ValueError(f"Batch size {len(prompts)} exceeds license limit of {max_batch}")
        
        # Create BatchJob objects for the engine
        jobs = []
        for i, prompt in enumerate(prompts):
            job = BatchJob(
                id=f"sdk_job_{i}",
                prompt=prompt,
                model=model_alias,
                metadata={
                    'template_name': template_name,
                    'sdk_version': '1.0',
                    'license_tier': self.license.tier
                }
            )
            jobs.append(job)
        
        # Delegate to the proven batch processor
        result = await self.batch_processor.process_batch(
            jobs=jobs,
            save_intermediate=save_results
        )
        
        return result
    
    def generate_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Generate text using specified model (synchronous)
        
        Args:
            prompt: The input prompt
            model: Model alias (defaults to license default)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text response
        """
        if not model:
            model = "gemini-2.5-flash"  # Default model
        
        provider = self.provider_factory.get_provider(model)
        return provider.generate_text(prompt=prompt, model=model, **kwargs)
    
    async def async_generate_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Generate text using specified model (asynchronous for parallel processing)
        
        Args:
            prompt: The input prompt
            model: Model alias (defaults to license default)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text response
        """
        if not model:
            model = "gemini-2.5-flash"  # Default model
        
        provider = self.provider_factory.get_provider(model)
        
        # Check if provider has async support
        if hasattr(provider, 'async_generate_text'):
            return await provider.async_generate_text(prompt=prompt, model=model, **kwargs)
        elif hasattr(provider, 'complete'):
            # GoogleProvider and others use 'complete' method
            return await provider.complete(prompt, model)
        elif hasattr(provider, 'generate_text'):
            # Fallback to sync version wrapped in executor
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                provider.generate_text,
                prompt,
                model,
                **kwargs
            )
        else:
            raise AttributeError(f"Provider {provider.__class__.__name__} has no compatible generation method")
    
    async def process_federation(
        self,
        prompts: List[str],
        use_all_models: bool = True,
        template_name: str = "federation"
    ) -> Dict[str, Any]:
        """
        Process prompts using the Galactic Federation (Premium/Enterprise only)
        
        This unleashes the full power of multi-provider parallel processing:
        - 75+ concurrent workers across all providers
        - Provider-isolated execution (no bottlenecks)
        - Automatic load balancing
        
        Args:
            prompts: List of prompts to process
            use_all_models: If True, uses ALL available models across ALL providers
            template_name: Template name for categorization
            
        Returns:
            Comprehensive results from all providers and models
        """
        # Check Federation access through Divine Arbiter
        processor = self._get_processor_for_operation('all' if use_all_models else None)
        
        # Check if we got the Federation
        from processors.galactic_federation import GalacticFederation
        if not isinstance(processor, GalacticFederation):
            logger.warning("🚫 Federation access denied - reverting to Monarchy processing")
            # Fall back to standard batch processing
            return await self.process_batch(prompts, model_alias="gpt-2", template_name=template_name)
        
        logger.info("🌌 GALACTIC FEDERATION ACTIVATED")
        logger.info(f"⚡ Processing {len(prompts)} prompts across ALL providers")
        
        # Create operation handler
        async def operation_handler(operation, model):
            provider = self.provider_factory.get_provider(model)
            result = await self.async_generate_text(
                prompt=operation.get('prompt'),
                model=model
            )
            return result
        
        # Execute through the Federation
        federation_result = await processor.execute_all_providers(
            operations=[{'prompt': p, 'id': i} for i, p in enumerate(prompts)],
            operation_handler=operation_handler
        )
        
        return federation_result
    
    async def generate_structured(self,
                                prompt: str,
                                schema_class,
                                model: str = "gpt-5-nano",
                                **kwargs) -> StructuredResponse:
        """
        Generate structured output with schema validation (Premium/Enterprise only)
        
        This ensures AI responses always conform to your defined Pydantic schemas:
        - Automatic validation and retry
        - Type-safe response guarantees
        - Multi-provider support
        
        Args:
            prompt: The input prompt
            schema_class: Pydantic BaseModel class defining the expected schema
            model: Model to use (defaults to gpt-5-nano for best structured output support)
            **kwargs: Additional provider arguments
            
        Returns:
            StructuredResponse with validated data
            
        Raises:
            StructuredOutputError: If tier doesn't support structured output
        """
        # Check Divine Arbiter permission
        if not hasattr(self, 'divine_arbiter'):
            from core.divine_arbiter import get_divine_arbiter
            self.divine_arbiter = get_divine_arbiter()
            if self.license.tier != self.divine_arbiter.current_tier:
                self.divine_arbiter.upgrade_tier(self.license.tier)
        
        if not self.divine_arbiter.check_structured_output_permission():
            raise StructuredOutputError("Structured outputs require Premium tier or higher")
        
        # Check if structured output processor is available
        if not self.structured_processor:
            raise StructuredOutputError(
                "Structured output processor not available. Install pydantic: pip install pydantic"
            )
        
        # Get provider and model info
        provider = self.provider_factory.get_provider(model)
        provider_name = provider.__class__.__name__.lower().replace('provider', '')
        
        logger.info(f"🎯 Generating structured output with {model} using schema {schema_class.__name__}")
        
        # Create async wrapper for provider
        async def provider_handler(prompt: str, model: str, **kwargs) -> str:
            return await self.async_generate_text(prompt, model, **kwargs)
        
        # Process with structured output engine
        result = await self.structured_processor.process_structured_request(
            prompt=prompt,
            schema_class=schema_class,
            provider_handler=provider_handler,
            model=model,
            provider=provider_name,
            **kwargs
        )
        
        logger.info(f"✅ Structured output generated successfully with {result.validation_attempts} attempt(s)")
        return result
    
    async def call_function(self,
                          function_name: str,
                          arguments: Dict[str, Any],
                          **kwargs) -> FunctionResult:
        """
        Execute a function call with tier-based access control (Professional+ feature)
        
        This enables agentic AI that can interact with external systems:
        - File operations (read, write, list)
        - Web requests and data fetching
        - Code execution in secure sandbox
        - System tools and utilities
        
        Args:
            function_name: Name of the function to call
            arguments: Dictionary of function arguments
            **kwargs: Additional execution parameters
            
        Returns:
            FunctionResult with execution outcome
            
        Raises:
            ValueError: If tier doesn't support function calling or function not found
        """
        # Check Divine Arbiter permission
        if not hasattr(self, 'divine_arbiter'):
            from core.divine_arbiter import get_divine_arbiter
            self.divine_arbiter = get_divine_arbiter()
            if self.license.tier != self.divine_arbiter.current_tier:
                self.divine_arbiter.upgrade_tier(self.license.tier)
        
        if not self.divine_arbiter.check_function_calling_permission():
            tier = self.divine_arbiter.current_tier
            raise ValueError(f"Function calling requires Professional tier or higher (current: {tier})")
        
        # Check if function registry is available
        if not self.function_registry:
            raise ValueError(
                "Function calling not available. Install dependencies: pip install requests aiohttp"
            )
        
        # Get current tier and available tools
        tier = self.divine_arbiter.current_tier
        available_tools = self.function_registry.get_tools_for_tier(tier)
        
        # Check if function is available for current tier
        if function_name not in available_tools:
            available_names = list(available_tools.keys())
            raise ValueError(
                f"Function '{function_name}' not available for {tier} tier. "
                f"Available functions: {available_names}"
            )
        
        logger.info(f"🔧 Executing function '{function_name}' with {tier} tier access")
        
        # Create function call object
        function_call = FunctionCall(
            name=function_name,
            arguments=arguments
        )
        
        # Execute the function
        result = await self.function_registry.execute_function(function_call)
        
        if result.success:
            logger.info(f"✅ Function '{function_name}' executed successfully")
        else:
            logger.error(f"❌ Function '{function_name}' failed: {result.error}")
        
        return result
    
    def list_available_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        List all functions available for the current tier
        
        Returns:
            Dictionary mapping function names to their definitions
        """
        if not hasattr(self, 'divine_arbiter'):
            from core.divine_arbiter import get_divine_arbiter
            self.divine_arbiter = get_divine_arbiter()
            if self.license.tier != self.divine_arbiter.current_tier:
                self.divine_arbiter.upgrade_tier(self.license.tier)
        
        if not self.function_registry:
            return {}
        
        # Get tier and available functions
        tier = self.divine_arbiter.current_tier
        available_tools = self.function_registry.get_tools_for_tier(tier)
        
        # Format function info
        function_info = {}
        for tool in available_tools:
            function_info[tool.name] = {
                'description': tool.description,
                'category': tool.category,
                'security_level': tool.security_level,
                'parameters': tool.parameters
            }
        
        return function_info
    
    async def process_council(
        self,
        prompt: str,
        models: List[str] = ["gpt-2", "ant-2", "goo-1"],
        template_name: str = "council"
    ) -> Dict[str, Any]:
        """
        Process single prompt across multiple models (AI Council mode)
        
        Args:
            prompt: The prompt to send to all models
            models: List of model aliases to use
            template_name: Template name for categorization
            
        Returns:
            Dictionary mapping model aliases to their responses
        """
        # Create a single job for council processing
        job = BatchJob(
            id="council_job",
            prompt=prompt,
            model=models[0],  # Primary model, others handled by council mode
            metadata={
                'template_name': template_name,
                'council_mode': True,
                'council_models': models,
                'license_tier': self.license.tier
            }
        )
        
        # Use the engine's council processing capability
        council_responses = await self.batch_processor.process_council_request(job, models)
        return council_responses
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available model aliases"""
        return self.provider_factory.list_models()
    
    def get_license_info(self) -> Dict[str, Any]:
        """Get current license information with Divine Arbiter capabilities"""
        # Get base license info
        base_info = {
            "tier": self.license.tier,
            "max_concurrent": self.max_concurrent,
            "valid": self.license.is_valid,
            "key_set": bool(self.license.license_key)
        }
        
        # Get Divine Arbiter capabilities
        if hasattr(self, 'divine_arbiter'):
            capabilities = self.divine_arbiter.get_tier_capabilities()
            base_info.update({
                "sovereignty": capabilities['sovereignty'],
                "parallel_providers": capabilities['parallel_providers'],
                "model_all_enabled": capabilities['model_all'],
                "structured_output": capabilities.get('structured_output', False),
                "function_calling": capabilities.get('function_calling', 'none'),
                "max_providers": capabilities['max_providers'],
                "total_workers": capabilities['max_workers'],
                "description": capabilities['description']
            })
        
        return base_info
    
    def display_tier_status(self):
        """Display current tier status and capabilities"""
        info = self.get_license_info()
        
        print("\n" + "="*60)
        print(f"🎫 HARVESTER SDK LICENSE STATUS")
        print("="*60)
        print(f"📊 Tier: {info['tier'].upper()}")
        print(f"⚖️ Sovereignty: {info.get('sovereignty', 'monarchy').upper()}")
        print(f"👷 Max Workers: {info.get('total_workers', self.max_concurrent)}")
        print(f"🌐 Multi-Provider: {'✅ ENABLED' if info.get('parallel_providers') else '❌ DISABLED'}")
        print(f"🎯 --model all: {'✅ ENABLED' if info.get('model_all_enabled') else '❌ DISABLED'}")
        print(f"🎯 Structured Output: {'✅ ENABLED' if info.get('structured_output') else '❌ DISABLED'}")
        print(f"🔧 Function Calling: {info.get('function_calling', 'none').upper()}")
        print("="*60)
        
        if info['tier'] in ['freemium', 'professional']:
            print("\n💎 UPGRADE TO PREMIUM FOR:")
            print("  • Galactic Federation access (75+ workers)")
            print("  • Multi-provider parallel execution")
            print("  • --model all flag support")
            print("  • 🎯 Structured Output with schema validation")
            print("  • 🔧 Function Calling & Tool Use (web, code, data)")
            print("  • 10x throughput on batch operations")
            print("\n🌟 Visit: https://quantumencoding.io/premium")
        
        print()
    
    async def cleanup(self):
        """Cleanup SDK resources"""
        await self.batch_processor.cleanup()


class HarvesterClient:
    """
    Simplified client interface for quick operations
    """
    
    def __init__(self, license_key: Optional[str] = None, output_dir: Optional[Path] = None):
        """
        Initialize simplified client
        
        Args:
            license_key: Harvester license key
            output_dir: Custom output directory (defaults to ~/harvest_sdk/)
        """
        self.sdk = HarvesterSDK(license_key=license_key, output_dir=output_dir)
    
    async def process(
        self, 
        prompts: List[str], 
        model_alias: str = "gpt-2",
        save_results: bool = True
    ) -> List[str]:
        """
        Simple batch processing that returns clean response strings
        
        Args:
            prompts: List of text prompts
            model_alias: Model alias (gpt-1, ant-2, goo-1, etc.)
            save_results: Whether to save results to organized output structure
        
        Returns:
            List of response strings
        """
        # Use the SDK's batch processing
        batch_result = await self.sdk.process_batch(
            prompts=prompts,
            model_alias=model_alias,
            save_results=save_results
        )
        
        # Extract clean response strings from BatchResult
        response_strings = []
        for job in batch_result.results:
            if job.result and job.status == 'completed':
                response_strings.append(job.result)
            else:
                response_strings.append(f"Error: {job.error or 'Unknown error'}")
        
        return response_strings
    
    async def process_council(
        self,
        prompt: str,
        models: List[str] = ["gpt-2", "ant-2", "goo-1"]
    ) -> Dict[str, str]:
        """
        Process single prompt across multiple models
        
        Args:
            prompt: The prompt to send to all models
            models: List of model aliases to use
            
        Returns:
            Dictionary mapping model aliases to response strings
        """
        
        # Check if this spans multiple providers - delegate to SDK
        # The SDK will handle Divine Arbiter access checks
        if len(models) > 3:
            logger.warning("⚠️  Large model council may require Federation access")
            # Let the SDK handle tier restrictions
        
        council_result = await self.sdk.process_council(prompt, models)
        
        # Extract clean responses
        clean_responses = {}
        for model, response_data in council_result.items():
            if response_data.get('status') == 'completed':
                clean_responses[model] = response_data.get('result', '')
            else:
                clean_responses[model] = f"Error: {response_data.get('error', 'Unknown error')}"
        
        return clean_responses


# Convenience functions for quick operations
async def quick_process(
    prompts: List[str], 
    model_alias: str = "gpt-2",
    license_key: Optional[str] = None
) -> List[str]:
    """
    Quick batch processing function
    
    Args:
        prompts: List of prompts to process
        model_alias: Model alias (gpt-1, ant-2, goo-1, etc.)
        license_key: Optional license key
    
    Returns:
        List of response strings
    """
    client = HarvesterClient(license_key=license_key)
    return await client.process(prompts, model_alias=model_alias)


async def quick_council(
    prompt: str,
    models: List[str] = ["gpt-2", "ant-2", "goo-1"],
    license_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Quick AI council processing function
    
    Args:
        prompt: Single prompt to send to multiple models
        models: List of model aliases to use
        license_key: Optional license key
    
    Returns:
        Dictionary mapping model aliases to response strings
    """
    client = HarvesterClient(license_key=license_key)
    return await client.process_council(prompt, models)


# Main exports
__all__ = [
    'HarvesterSDK',
    'HarvesterClient', 
    'HarvesterLicense',
    'quick_process',
    'quick_council'
]