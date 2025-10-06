"""
Template Adapter - Provider-Aware Template System
© 2025 Quantum Encoding Ltd

Decouples templates from specific models and providers.
Adapts template format based on provider requirements.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class TemplateAdapter:
    """Adapts templates for different providers and models"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'templates' / 'template_config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.registry = self.config.get('template_registry', {})
        self.formats = self.config.get('provider_formats', {})
        self.auto_selection = self.config.get('auto_selection', {})
        self.aliases = self.config.get('aliases', {})

    def get_template_path(self, template_name: str) -> str:
        """Get full path to template"""
        # Resolve alias
        template_name = self.aliases.get(template_name, template_name)

        if template_name not in self.registry:
            raise ValueError(f"Template '{template_name}' not found in registry")

        template_config = self.registry[template_name]
        base_path = Path(__file__).parent.parent / 'templates'
        return str(base_path / template_config['path'])

    def get_recommended_models(self, template_name: str, provider: Optional[str] = None) -> List[str]:
        """Get recommended models for a template"""
        template_name = self.aliases.get(template_name, template_name)

        if template_name not in self.registry:
            return []

        template_config = self.registry[template_name]
        recommended = template_config.get('recommended_models', {})

        if provider:
            # Get models for specific provider
            provider_models = recommended.get(provider, [])
            if isinstance(provider_models, dict):
                return provider_models.get('models', [])
            return provider_models
        else:
            # Get all recommended models
            all_models = []
            for prov, models in recommended.items():
                if isinstance(models, dict):
                    all_models.extend(models.get('models', []))
                elif isinstance(models, list):
                    all_models.extend(models)
            return all_models

    def get_default_model(self, template_name: str) -> str:
        """Get default model for a template"""
        template_name = self.aliases.get(template_name, template_name)

        if template_name not in self.registry:
            return 'gemini-2.5-flash'  # Fallback

        return self.registry[template_name].get('default', 'gemini-2.5-flash')

    def auto_select_model(self, template_name: str, task_type: Optional[str] = None) -> str:
        """Auto-select best model based on template and task type"""
        template_name = self.aliases.get(template_name, template_name)

        if template_name not in self.registry:
            return 'gemini-2.5-flash'

        template_config = self.registry[template_name]
        category = template_config.get('category', 'general')

        # Determine task type from category if not specified
        if not task_type:
            task_map = {
                'code': 'code_tasks',
                'business': 'content_creation',
                'content': 'content_creation',
                'image': 'image_generation'
            }
            task_type = task_map.get(category, 'quick_tasks')

        # Get priority list for task type
        priority_models = self.auto_selection.get(task_type, {}).get('priority', [])

        # Get recommended models for this template
        recommended = self.get_recommended_models(template_name)

        # Find first priority model that's recommended
        for model in priority_models:
            if model in recommended:
                logger.info(f"Auto-selected model '{model}' for template '{template_name}'")
                return model

        # Fallback to default
        return self.get_default_model(template_name)

    def get_provider_format(self, model: str) -> str:
        """Get format type for a model"""
        # Image models
        if 'dall-e' in model:
            return 'openai_image'
        elif 'flash-image' in model or 'gemini' in model and 'image' in model:
            return 'genai_image'
        elif 'imagen' in model:
            return 'vertex_image'
        else:
            # Text models - standard format
            return 'standard'

    def adapt_template_for_provider(self, template_content: str, provider: str, model: str) -> str:
        """Adapt template based on provider requirements"""
        format_type = self.get_provider_format(model)

        if format_type == 'standard':
            # No adaptation needed for standard text models
            return template_content

        # For image models, we may need to adapt the template
        if format_type in ['openai_image', 'genai_image', 'vertex_image']:
            # Image-specific adaptations
            return self._adapt_image_template(template_content, format_type)

        return template_content

    def _adapt_image_template(self, template: str, format_type: str) -> str:
        """Adapt image template for specific provider format"""

        if format_type == 'openai_image':
            # DALL-E specific adaptations
            # Add any DALL-E specific instructions
            return template

        elif format_type == 'genai_image':
            # GenAI Gemini Flash Image specific
            return template

        elif format_type == 'vertex_image':
            # Vertex Imagen specific
            return template

        return template

    def validate_model_for_template(self, template_name: str, model: str) -> bool:
        """Check if model is recommended for template"""
        template_name = self.aliases.get(template_name, template_name)
        recommended = self.get_recommended_models(template_name)
        return model in recommended

    def get_template_category(self, template_name: str) -> str:
        """Get template category"""
        template_name = self.aliases.get(template_name, template_name)

        if template_name not in self.registry:
            return 'general'

        return self.registry[template_name].get('category', 'general')

    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """List available templates, optionally filtered by category"""
        if category:
            return [
                name for name, config in self.registry.items()
                if config.get('category') == category
            ]
        return list(self.registry.keys())

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get complete info about a template"""
        template_name = self.aliases.get(template_name, template_name)

        if template_name not in self.registry:
            return {}

        config = self.registry[template_name].copy()
        config['name'] = template_name
        config['aliases'] = [k for k, v in self.aliases.items() if v == template_name]

        return config


# Global instance
_adapter = None

def get_template_adapter() -> TemplateAdapter:
    """Get or create global template adapter instance"""
    global _adapter
    if _adapter is None:
        _adapter = TemplateAdapter()
    return _adapter


# Convenience functions
def get_template_path(template_name: str) -> str:
    """Get template path"""
    return get_template_adapter().get_template_path(template_name)

def auto_select_model(template_name: str, task_type: Optional[str] = None) -> str:
    """Auto-select best model for template"""
    return get_template_adapter().auto_select_model(template_name, task_type)

def get_recommended_models(template_name: str, provider: Optional[str] = None) -> List[str]:
    """Get recommended models"""
    return get_template_adapter().get_recommended_models(template_name, provider)
