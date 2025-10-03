"""
Provider factory for instantiating AI providers with enhanced authentication
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from importlib import import_module

# Import dotenv for loading .env files
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables at module level
except ImportError:
    logging.warning("python-dotenv not installed, falling back to os.environ")

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class ProviderFactory:
    """Factory for creating provider instances with proper authentication"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / 'config'
        self.providers_config = self._load_providers_config()
        self.provider_instances = {}
        self.api_keys = self._load_api_keys()
        
        # Log available API keys (without exposing values)
        available_keys = [k for k, v in self.api_keys.items() if v]
        logger.info(f"Loaded API keys for: {', '.join(available_keys) if available_keys else 'none'}")
    
    def get_provider(self, model_alias: str) -> BaseProvider:
        """
        Get provider instance for given model alias
        
        Args:
            model_alias: Model alias like 'goo-1', 'ant-2', etc.
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If model alias is unknown or provider cannot be created
        """
        # Determine provider from alias
        provider_name = self._get_provider_name(model_alias)
        
        if not provider_name:
            available_models = list(self.list_models().keys())
            raise ValueError(
                f"Unknown model alias: {model_alias}. "
                f"Available models: {', '.join(available_models[:10])}..."
            )
        
        # Return cached instance or create new one
        cache_key = f"{provider_name}_{model_alias}"
        if cache_key not in self.provider_instances:
            self.provider_instances[cache_key] = self._create_provider(provider_name, model_alias)
        
        return self.provider_instances[cache_key]
    
    def _get_provider_name(self, model_alias: str) -> Optional[str]:
        """Determine provider name from model alias"""
        # Priority order: specialized providers first, then general providers
        # This ensures image models route to image providers, not text providers
        provider_priority = []
        
        # First pass: collect specialized providers (image, video, etc.)
        for provider_name in self.providers_config.keys():
            if '_' in provider_name:  # Specialized providers have underscores
                provider_priority.append(provider_name)
        
        # Second pass: add general providers
        for provider_name in self.providers_config.keys():
            if '_' not in provider_name:  # General providers don't have underscores
                provider_priority.append(provider_name)
        
        # Check providers in priority order
        for provider_name in provider_priority:
            config = self.providers_config[provider_name]
            aliases = config.get('aliases', {})
            if model_alias in aliases:
                logger.debug(f"Model {model_alias} maps to provider {provider_name}")
                return provider_name
        
        logger.debug(f"No provider found for model alias: {model_alias}")
        return None
    
    def _create_provider(self, provider_name: str, model_alias: str) -> BaseProvider:
        """
        Create provider instance with proper authentication
        
        Args:
            provider_name: Name of provider (google, anthropic, etc.)
            model_alias: Model alias being requested
            
        Returns:
            Configured provider instance
        """
        logger.info(f"Creating provider: {provider_name} for model: {model_alias}")
        
        # Get provider config
        config = self.providers_config.get(provider_name, {}).copy()
        
        # Add API key to config based on provider type
        api_key = self._get_api_key_for_provider(provider_name)
        if api_key:
            config['api_key'] = api_key
            logger.debug(f"API key configured for {provider_name}")
        else:
            # Special message for providers that use gcloud auth
            if provider_name in ['vertex_image', 'vertex_video', 'google', 'vertex']:
                logger.info(f"No API key found for {provider_name} (using gcloud credentials)")
            # Providers that check env vars themselves (provider will handle)
            elif provider_name in ['xai_image', 'genai_imagen']:
                logger.debug(f"No API key in config for {provider_name} (will check environment)")
            else:
                logger.warning(f"No API key found for {provider_name}")
            # Don't fail here - let the provider handle missing keys
        
        # Add model alias to config
        config['model_alias'] = model_alias
        
        # Import and create provider
        try:
            provider_instance = self._instantiate_provider(provider_name, config)
            logger.info(f"Successfully created {provider_name} provider")
            return provider_instance
            
        except Exception as e:
            logger.error(f"Failed to create provider {provider_name}: {e}")
            
            # Fall back to a generic provider if available
            try:
                from .generic_provider import GenericProvider
                logger.warning(f"Falling back to GenericProvider for {provider_name}")
                return GenericProvider(config)
            except ImportError:
                # If no fallback available, re-raise the original error
                raise ValueError(f"Cannot create provider {provider_name}: {e}")
    
    def _get_api_key_for_provider(self, provider_name: str) -> Optional[str]:
        """Get API key for specific provider"""
        # Standard environment variable naming
        key_mappings = {
            'google': 'GOOGLE_APPLICATION_CREDENTIALS',  # Vertex AI uses service account
            'genai': 'GEMINI_API_KEY',  # Google AI Studio uses API key
            'gemini': 'GEMINI_API_KEY',  # Legacy name for genai
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'vertex': 'GOOGLE_APPLICATION_CREDENTIALS',  # For service account
            'deepseek': 'DEEPSEEK_API_KEY',
            'xai': 'XAI_API_KEY',  # xAI Grok models
            'xai_image': 'XAI_API_KEY',  # xAI Grok Image uses same key
            'gemini_exp': 'GEMINI_API_KEY',  # Gemini experimental
            'openai_image': 'OPENAI_API_KEY',  # Image generation uses same key
            'dalle3': 'OPENAI_API_KEY',  # DALL-E 3 uses OpenAI key
            'gpt_image': 'OPENAI_API_KEY',  # GPT Image uses OpenAI key
            'vertex_image': 'GOOGLE_APPLICATION_CREDENTIALS',  # Image generation uses same auth
            'genai_imagen': 'GEMINI_API_KEY',  # Google AI Studio Image (uses same Gemini key)
            'genai_veo3': 'GEMINI_API_KEY',  # Google AI Studio Video (Veo 3)
            'genai_lyria': 'GEMINI_API_KEY',  # Google AI Studio Music (Lyria)
            'genai_embeddings': 'GEMINI_API_KEY'  # Google AI Studio Embeddings
        }
        
        env_key = key_mappings.get(provider_name)
        if env_key:
            api_key = self.api_keys.get(env_key)
            if api_key:
                return api_key
        
        # Fallback: try provider name directly
        fallback_key = f"{provider_name.upper()}_API_KEY"
        return self.api_keys.get(fallback_key)
    
    def _instantiate_provider(self, provider_name: str, config: Dict[str, Any]) -> BaseProvider:
        """Instantiate provider class dynamically"""
        # Import provider module
        module_name = f"providers.{provider_name}_provider"
        
        try:
            # Try relative import first
            module = import_module(module_name)
        except ImportError:
            # Try absolute import
            module = import_module(f".{provider_name}_provider", package="providers")
        
        # Handle special naming for providers
        if provider_name == 'openai_image':
            class_name = "OpenAIImageProvider"
        elif provider_name == 'vertex_image':
            class_name = "VertexImageProvider"
        elif provider_name == 'genai_imagen':
            class_name = "GenAIImagenProvider"
        elif provider_name == 'genai_veo3':
            class_name = "GenAIVeo3Provider"
        elif provider_name == 'genai_lyria':
            class_name = "GenAILyriaProvider"
        elif provider_name == 'genai_embeddings':
            class_name = "GenAIEmbeddingsProvider"
        elif provider_name == 'dalle3':
            class_name = "DallE3Provider"
        elif provider_name == 'gpt_image':
            class_name = "GPTImageProvider"
        elif provider_name == 'xai_image':
            class_name = "XaiImageProvider"
        elif provider_name == 'xai':
            class_name = "XAIProvider"
        elif provider_name == 'gemini_exp':
            class_name = "GeminiExpProvider"
        elif provider_name == 'genai':
            class_name = "GenAIProvider"  # New Google AI Studio provider
        elif provider_name == 'google':
            class_name = "GoogleProvider"  # Updated to Vertex AI only
        else:
            # Get provider class (assumes it's named XxxProvider)
            class_name = f"{provider_name.title()}Provider"
        
        provider_class = getattr(module, class_name)
        
        # Create instance
        return provider_class(config)
    
    def _load_providers_config(self) -> Dict[str, Any]:
        """Load providers configuration from YAML"""
        config_path = self.config_dir / 'providers.yaml'
        
        try:
            if not config_path.exists():
                logger.error(f"Providers config not found at {config_path}")
                return {}
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                providers = config.get('providers', {})
                logger.debug(f"Loaded configuration for {len(providers)} providers")
                return providers
                
        except Exception as e:
            logger.error(f"Error loading providers config from {config_path}: {e}")
            return {}
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from multiple sources with precedence:
        1. Environment variables (highest priority)
        2. .env file 
        3. .secrets directory (for service account files)
        """
        keys = {}
        
        # Load from .env file first
        env_file = Path('.env')
        if env_file.exists():
            logger.debug("Loading API keys from .env file")
            try:
                with open(env_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            try:
                                key, value = line.split('=', 1)
                                # Clean up quotes and whitespace
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")
                                if value:  # Only store non-empty values
                                    keys[key] = value
                            except ValueError:
                                logger.warning(f"Malformed line {line_num} in .env file: {line}")
                                
            except Exception as e:
                logger.error(f"Error reading .env file: {e}")
        else:
            logger.debug("No .env file found")
        
        # Override with actual environment variables (highest priority)
        env_keys = [
            'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 
            'DEEPSEEK_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS'
        ]
        
        for key in env_keys:
            if key in os.environ and os.environ[key]:
                keys[key] = os.environ[key]
                logger.debug(f"Loaded {key} from environment")
        
        # Handle Google service account credentials from .secrets directory
        secrets_dir = Path('.secrets')
        if secrets_dir.exists():
            logger.debug("Checking .secrets directory for service account credentials")
            
            # Look for service account JSON files
            for json_file in secrets_dir.glob('*.json'):
                if 'service' in json_file.name.lower() or 'gcp' in json_file.name.lower():
                    keys['GOOGLE_APPLICATION_CREDENTIALS'] = str(json_file.absolute())
                    logger.debug(f"Found service account file: {json_file.name}")
                    break
        
        return keys
    
    def list_models(self) -> Dict[str, str]:
        """List all available model aliases with their provider info"""
        models = {}
        
        for provider_name, config in self.providers_config.items():
            aliases = config.get('aliases', {})
            for alias, model_name in aliases.items():
                models[alias] = f"{model_name} ({provider_name})"
        
        return models
    
    def list_providers(self) -> Dict[str, Dict[str, Any]]:
        """List all configured providers with their status"""
        providers = {}
        
        for provider_name, config in self.providers_config.items():
            api_key = self._get_api_key_for_provider(provider_name)
            providers[provider_name] = {
                'configured': True,
                'has_api_key': api_key is not None,
                'aliases': list(config.get('aliases', {}).keys()),
                'endpoint': config.get('endpoint', 'unknown')
            }
        
        return providers
    
    async def cleanup(self):
        """Cleanup all provider instances"""
        logger.info("Cleaning up provider instances")
        
        for instance_key, provider in self.provider_instances.items():
            try:
                if hasattr(provider, 'close'):
                    await provider.close()
                elif hasattr(provider, 'cleanup'):
                    await provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up provider {instance_key}: {e}")
        
        self.provider_instances.clear()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate provider configuration and return status report"""
        report = {
            'providers_configured': len(self.providers_config),
            'providers_with_keys': 0,
            'total_models': 0,
            'issues': []
        }
        
        for provider_name, config in self.providers_config.items():
            # Check if provider has API key
            api_key = self._get_api_key_for_provider(provider_name)
            if api_key:
                report['providers_with_keys'] += 1
            else:
                report['issues'].append(f"No API key found for {provider_name}")
            
            # Count models
            aliases = config.get('aliases', {})
            report['total_models'] += len(aliases)
            
            # Check if provider module exists
            try:
                module_name = f"providers.{provider_name}_provider"
                import_module(module_name)
            except ImportError:
                report['issues'].append(f"Provider module not found: {module_name}")
        
        return report