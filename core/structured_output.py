"""
Structured Output Engine - Premium Tier Feature

This module provides structured output capabilities with Pydantic schema validation,
automatic retries, and multi-provider support. Ensures AI responses always conform
to your defined JSON schemas.

Key Features:
- Pydantic schema validation
- Automatic retry on invalid responses  
- Multi-provider structured output support
- Streaming structured responses
- Schema compilation and optimization

Copyright (c) 2025 Quantum Encoding Ltd.
Licensed under the Harvester Commercial License.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Type, get_origin, get_args
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import asyncio

try:
    from pydantic import BaseModel, ValidationError, create_model
    from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ValidationError = Exception

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """Base exception for structured output errors"""
    pass


class SchemaValidationError(StructuredOutputError):
    """Raised when response doesn't match schema"""
    pass


class ProviderNotSupportedError(StructuredOutputError):
    """Raised when provider doesn't support structured output"""
    pass


@dataclass
class StructuredResponse:
    """Container for structured output response"""
    raw_response: str
    parsed_data: Any
    schema_used: str
    model_used: str
    provider_used: str
    validation_attempts: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'raw_response': self.raw_response,
            'parsed_data': self.parsed_data,
            'schema_used': self.schema_used,
            'model_used': self.model_used,
            'provider_used': self.provider_used,
            'validation_attempts': self.validation_attempts,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    PYDANTIC = "pydantic"
    DATACLASS = "dataclass"


class StructuredOutputProcessor:
    """
    The Structured Output Engine - Premium Tier Feature
    
    Ensures AI responses always conform to your defined schemas with:
    - Automatic validation and retry
    - Multi-provider support
    - Schema optimization
    - Type safety guarantees
    """
    
    # Provider capabilities for structured output
    PROVIDER_SUPPORT = {
        'openai': {
            'native_structured': True,
            'models': ['gpt-5', 'gpt-5-nano', 'gpt-5-mini'],
            'max_retries': 3
        },
        'anthropic': {
            'native_structured': False,  # Uses prompt engineering
            'models': ['claude-3-5-sonnet', 'claude-3-5-haiku'],
            'max_retries': 5
        },
        'google': {
            'native_structured': True,
            'models': ['gemini-2.5-pro', 'gemini-2.5-flash'],
            'max_retries': 3
        },
        'deepseek': {
            'native_structured': True,  # Has JSON mode via response_format
            'models': ['deepseek-chat', 'deepseek-reasoner'],
            'max_retries': 3  # JSON mode is more reliable
        },
        'xai': {
            'native_structured': False,
            'models': ['grok-4', 'grok-3'],
            'max_retries': 4
        }
    }
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        """
        Initialize the Structured Output Processor
        
        Args:
            max_retries: Maximum retry attempts for validation failures
            timeout: Timeout for each attempt in seconds
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for structured outputs. "
                "Install with: pip install pydantic"
            )
        
        self.max_retries = max_retries
        self.timeout = timeout
        self.schema_cache = {}
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'validation_failures': 0,
            'retry_attempts': 0
        }
        
        logger.info("🎯 Structured Output Processor initialized")
    
    def compile_schema(self, schema_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Compile Pydantic model to JSON schema with optimizations
        
        Args:
            schema_class: Pydantic BaseModel class
            
        Returns:
            Optimized JSON schema dictionary
        """
        if not issubclass(schema_class, BaseModel):
            raise ValueError("Schema must be a Pydantic BaseModel")
        
        # Check cache first
        schema_name = schema_class.__name__
        if schema_name in self.schema_cache:
            return self.schema_cache[schema_name]
        
        # Generate JSON schema
        json_schema = schema_class.model_json_schema()
        
        # Optimize for structured output requirements
        optimized_schema = self._optimize_schema(json_schema)
        
        # Cache the compiled schema
        self.schema_cache[schema_name] = optimized_schema
        
        logger.debug(f"📋 Compiled schema for {schema_name}")
        return optimized_schema
    
    def _optimize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize JSON schema for structured output providers
        
        Ensures compatibility with provider requirements:
        - All fields marked as required
        - additionalProperties: false
        - Proper enum handling
        """
        optimized = schema.copy()
        
        # Ensure all properties are required for structured output
        if 'properties' in optimized:
            optimized['required'] = list(optimized['properties'].keys())
        
        # Disable additional properties
        optimized['additionalProperties'] = False
        
        # Recursively optimize nested objects
        if 'properties' in optimized:
            for prop_name, prop_schema in optimized['properties'].items():
                if isinstance(prop_schema, dict) and prop_schema.get('type') == 'object':
                    optimized['properties'][prop_name] = self._optimize_schema(prop_schema)
        
        # Handle array items
        if optimized.get('type') == 'array' and 'items' in optimized:
            if isinstance(optimized['items'], dict):
                optimized['items'] = self._optimize_schema(optimized['items'])
        
        return optimized
    
    def validate_response(self, 
                         response: str, 
                         schema_class: Type[BaseModel]) -> Any:
        """
        Validate response against Pydantic schema
        
        Args:
            response: Raw response text from model
            schema_class: Pydantic schema class
            
        Returns:
            Validated and parsed data
            
        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            # First, try to parse as JSON
            try:
                raw_data = json.loads(response)
            except json.JSONDecodeError as e:
                raise SchemaValidationError(f"Invalid JSON: {e}")
            
            # Validate against Pydantic schema
            validated_data = schema_class.model_validate(raw_data)
            
            logger.debug(f"✅ Response validated against {schema_class.__name__}")
            return validated_data
            
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append(f"{error['loc']}: {error['msg']}")
            
            raise SchemaValidationError(
                f"Schema validation failed: {'; '.join(error_details)}"
            )
    
    def get_provider_capability(self, provider: str, model: str) -> Dict[str, Any]:
        """
        Get provider capabilities for structured output
        
        Args:
            provider: Provider name (openai, anthropic, etc.)
            model: Model name
            
        Returns:
            Provider capability information
        """
        if provider not in self.PROVIDER_SUPPORT:
            raise ProviderNotSupportedError(f"Provider {provider} not supported")
        
        capabilities = self.PROVIDER_SUPPORT[provider].copy()
        
        # Check model support
        supported_models = capabilities.get('models', [])
        model_supported = any(
            model.startswith(supported_model.split('-')[0]) 
            for supported_model in supported_models
        )
        
        capabilities['model_supported'] = model_supported
        return capabilities
    
    def build_structured_prompt(self, 
                               original_prompt: str,
                               schema: Dict[str, Any],
                               provider: str) -> str:
        """
        Build provider-specific prompt for structured output
        
        For providers without native structured output support,
        this enhances the prompt with schema requirements.
        """
        capabilities = self.get_provider_capability(provider, "")
        
        if capabilities.get('native_structured', False):
            # Provider has native support, return original prompt
            return original_prompt
        
        # Build enhanced prompt for prompt-engineering approach
        schema_str = json.dumps(schema, indent=2)
        
        enhanced_prompt = f"""
{original_prompt}

CRITICAL: You MUST respond with valid JSON that exactly matches this schema:

```json
{schema_str}
```

Requirements:
- Return ONLY valid JSON, no additional text
- Include ALL required fields
- Match exact field names and types
- No additional properties beyond the schema
- Ensure proper JSON formatting

Your response:"""
        
        return enhanced_prompt
    
    async def process_structured_request(self,
                                       prompt: str,
                                       schema_class: Type[BaseModel],
                                       provider_handler,
                                       model: str,
                                       provider: str,
                                       **kwargs) -> StructuredResponse:
        """
        Process a request with structured output requirements
        
        Args:
            prompt: Original prompt
            schema_class: Pydantic schema class
            provider_handler: Provider's generate method
            model: Model name
            provider: Provider name
            **kwargs: Additional provider arguments
            
        Returns:
            StructuredResponse with validated data
        """
        self.stats['total_requests'] += 1
        
        # Compile schema
        json_schema = self.compile_schema(schema_class)
        
        # Get provider capabilities
        capabilities = self.get_provider_capability(provider, model)
        
        if not capabilities.get('model_supported', False):
            logger.warning(f"⚠️ Model {model} may not fully support structured output")
        
        # Determine retry limit
        max_attempts = min(self.max_retries, capabilities.get('max_retries', 3))
        
        last_error = None
        validation_attempts = 0
        
        for attempt in range(max_attempts):
            try:
                validation_attempts += 1
                
                # Build appropriate prompt
                if capabilities.get('native_structured', False):
                    # Use provider's native structured output
                    response = await self._call_with_native_structured(
                        provider_handler, prompt, json_schema, model, **kwargs
                    )
                else:
                    # Use prompt engineering approach
                    enhanced_prompt = self.build_structured_prompt(
                        prompt, json_schema, provider
                    )
                    response = await provider_handler(
                        prompt=enhanced_prompt, model=model, **kwargs
                    )
                
                # Validate response
                validated_data = self.validate_response(response, schema_class)
                
                # Success!
                self.stats['successful_requests'] += 1
                if attempt > 0:
                    self.stats['retry_attempts'] += attempt
                
                return StructuredResponse(
                    raw_response=response,
                    parsed_data=validated_data,
                    schema_used=schema_class.__name__,
                    model_used=model,
                    provider_used=provider,
                    validation_attempts=validation_attempts,
                    metadata={
                        'schema': json_schema,
                        'capabilities': capabilities,
                        'success_on_attempt': attempt + 1
                    }
                )
                
            except (SchemaValidationError, json.JSONDecodeError) as e:
                last_error = e
                self.stats['validation_failures'] += 1
                
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"🔄 Validation failed (attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            
            except Exception as e:
                logger.error(f"❌ Unexpected error in structured output: {e}")
                raise StructuredOutputError(f"Processing failed: {e}")
        
        # All attempts failed
        raise SchemaValidationError(
            f"Failed to get valid structured output after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )
    
    async def _call_with_native_structured(self,
                                         provider_handler,
                                         prompt: str,
                                         schema: Dict[str, Any],
                                         model: str,
                                         **kwargs) -> str:
        """
        Call provider with native structured output support

        This method handles provider-specific implementations
        for native structured output APIs.
        """
        # Pass JSON schema to provider via kwargs for native support
        # Provider will detect json_schema and use appropriate API format
        return await provider_handler(
            prompt=prompt,
            model=model,
            json_schema=schema,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_requests'] / self.stats['total_requests']
                if self.stats['total_requests'] > 0 else 0
            ),
            'average_retries': (
                self.stats['retry_attempts'] / self.stats['successful_requests']
                if self.stats['successful_requests'] > 0 else 0
            ),
            'schemas_cached': len(self.schema_cache)
        }
    
    def clear_cache(self):
        """Clear schema cache"""
        self.schema_cache.clear()
        logger.info("🗑️ Schema cache cleared")


# Common Pydantic schemas for immediate use
class BasicResponse(BaseModel):
    """Basic structured response schema"""
    content: str
    confidence: float
    metadata: Dict[str, Any] = {}


class AnalysisResult(BaseModel):
    """Analysis result schema"""
    summary: str
    key_points: List[str]
    sentiment: str
    confidence_score: float
    tags: List[str] = []


class CodeAnalysis(BaseModel):
    """Code analysis schema"""
    language: str
    complexity: str
    issues: List[str]
    suggestions: List[str]
    quality_score: float
    lines_of_code: int


class DataExtraction(BaseModel):
    """Data extraction schema"""
    entities: List[str]
    dates: List[str]
    numbers: List[float]
    locations: List[str]
    organizations: List[str]


# Export commonly used items
__all__ = [
    'StructuredOutputProcessor',
    'StructuredResponse', 
    'StructuredOutputError',
    'SchemaValidationError',
    'ProviderNotSupportedError',
    'OutputFormat',
    'BasicResponse',
    'AnalysisResult', 
    'CodeAnalysis',
    'DataExtraction'
]