"""
Template engine for wrapping code in prompts with divine context injection
Integrated with template_adapter for provider-aware template selection
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader, Template
import yaml
import logging

# --- UPGRADE: Import the modules we want to make globally available ---
from datetime import datetime
import uuid
import random

logger = logging.getLogger(__name__)

# Import template adapter for provider-aware templates
try:
    from .template_adapter import get_template_adapter
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False
    logger.warning("Template adapter not available - using legacy template loading")

class Templater:
    """Manages and renders prompt templates with enhanced global context"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / 'templates'
        
        self.template_dir = Path(template_dir)
        
        try:
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # --- UPGRADE: Inject utility modules into the global context ---
            # This makes datetime, uuid, and random available directly in all templates
            self.env.globals['datetime'] = datetime
            self.env.globals['uuid'] = uuid
            self.env.globals['random'] = random
            
            # Add custom filters
            self.env.filters['truncate_code'] = self._truncate_code
            self.env.filters['estimate_tokens'] = self._estimate_tokens
            
            # Load template configs
            self.template_configs = self._load_template_configs()
            
            logger.info(f"Templater initialized with {len(self.template_configs)} templates and global contexts: datetime, uuid, random")
            
        except Exception as e:
            logger.error(f"Failed to initialize Jinja2 environment at {template_dir}: {e}")
            raise
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render template with given context

        Args:
            template_name: Name of template or path to .j2 file
            context: Variables to pass to template

        Returns:
            Rendered template string

        Raises:
            TemplateNotFound: If the template file doesn't exist
            Exception: For any other rendering errors
        """
        try:
            # Validate context before rendering
            if not self.validate_context(template_name, context):
                logger.warning(f"Context validation failed for template '{template_name}', proceeding anyway")

            # Add default context (now with proper datetime import)
            enhanced_context = {
                **self._get_default_context(),
                **context
            }

            # Get template file path
            # First, try template adapter for organized jinja/ templates
            template_file = None
            if ADAPTER_AVAILABLE:
                try:
                    adapter = get_template_adapter()
                    # Check if this is a registered template (supports aliases)
                    if template_name.replace('.j2', '') in adapter.registry or template_name in adapter.aliases:
                        template_file = adapter.get_template_path(template_name.replace('.j2', ''))
                        # Make path relative to template_dir
                        template_file = str(Path(template_file).relative_to(self.template_dir))
                        logger.debug(f"Using template adapter path: {template_file} for template: {template_name}")
                except (ValueError, FileNotFoundError):
                    # Not in registry, fall through to legacy loading
                    pass

            # Fallback to legacy template config or direct file reference
            if not template_file:
                if template_name in self.template_configs:
                    template_file = self.template_configs[template_name]['file']
                    logger.debug(f"Using configured template file: {template_file} for template: {template_name}")
                else:
                    template_file = template_name
                    logger.debug(f"Using direct template file: {template_file}")

            # Load and render template
            template = self.env.get_template(template_file)
            rendered = template.render(**enhanced_context)

            # Post-process for cleaner output
            rendered = self._post_process(rendered)

            logger.debug(f"Successfully rendered template '{template_name}' ({len(rendered)} characters)")
            return rendered

        except Exception as e:
            logger.error(f"Error rendering template '{template_name}': {e}")
            # Re-raise so the calling code can handle template failures appropriately
            raise
    
    def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """
        Render a template from a string with enhanced context
        
        Args:
            template_string: Template content as string
            context: Variables to pass to template
            
        Returns:
            Rendered template string
        """
        try:
            # Create template from string with access to global context
            template = self.env.from_string(template_string)
            enhanced_context = {
                **self._get_default_context(),
                **context
            }
            return template.render(**enhanced_context)
        except Exception as e:
            logger.error(f"Error rendering template string: {e}")
            raise
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a template"""
        if template_name in self.template_configs:
            info = self.template_configs[template_name].copy()
            # Add runtime information
            info['available'] = True
            info['file_path'] = str(self.template_dir / info.get('file', template_name))
            return info
        
        # Check if it's a direct file reference
        template_path = self.template_dir / template_name
        if template_path.exists():
            return {
                'name': template_name,
                'file': template_name,
                'available': True,
                'file_path': str(template_path),
                'configured': False
            }
        
        return {'available': False, 'configured': False}
    
    def list_templates(self) -> List[str]:
        """List all available templates (both configured and files)"""
        templates = set()

        # Add templates from adapter registry if available
        if ADAPTER_AVAILABLE:
            try:
                adapter = get_template_adapter()
                templates.update(adapter.list_templates())
            except Exception as e:
                logger.warning(f"Error getting templates from adapter: {e}")

        # Add templates from legacy config
        templates.update(self.template_configs.keys())

        # Also scan for .j2 files in template directory (including subdirectories)
        try:
            # Scan root templates/
            for f in self.template_dir.glob('*.j2'):
                templates.add(f.stem)

            # Scan jinja/ subdirectories
            jinja_dir = self.template_dir / 'jinja'
            if jinja_dir.exists():
                for f in jinja_dir.glob('**/*.j2'):
                    templates.add(f.stem)
        except Exception as e:
            logger.warning(f"Error scanning template directory: {e}")

        return sorted(templates)
    
    def validate_context(self, template_name: str, context: Dict[str, Any]) -> bool:
        """
        Validate that context has required variables for template
        
        Args:
            template_name: Name of template to validate against
            context: Context dictionary to validate
            
        Returns:
            True if context is valid, False otherwise
        """
        if template_name not in self.template_configs:
            # If template isn't configured, assume it's valid
            return True
        
        required_vars = self.template_configs[template_name].get('variables', [])
        if not required_vars:
            return True
        
        missing = [var for var in required_vars if var not in context]
        
        if missing:
            logger.warning(f"Missing required variables for template '{template_name}': {missing}")
            return False
        
        logger.debug(f"Context validation passed for template '{template_name}'")
        return True
    
    def get_global_context_info(self) -> Dict[str, Any]:
        """Get information about globally available context variables"""
        return {
            'datetime': {
                'description': 'Python datetime module for date/time operations',
                'example_usage': '{{ datetime.now().strftime("%Y-%m-%d %H:%M:%S") }}'
            },
            'uuid': {
                'description': 'Python uuid module for generating unique identifiers',
                'example_usage': '{{ uuid.uuid4() }}'
            },
            'random': {
                'description': 'Python random module for randomization',
                'example_usage': '{{ random.randint(1, 100) }}'
            }
        }
    
    def _load_template_configs(self) -> Dict[str, Any]:
        """Load template configurations from templates.yaml"""
        config_path = Path(__file__).parent.parent / 'config' / 'templates.yaml'
        
        try:
            if not config_path.exists():
                logger.warning(f"Template config file not found at {config_path}")
                return {}
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                templates = config.get('templates', {})
                logger.debug(f"Loaded {len(templates)} template configurations")
                return templates
                
        except Exception as e:
            logger.error(f"Error loading template configs from {config_path}: {e}")
            return {}
    
    def _get_default_context(self) -> Dict[str, Any]:
        """
        Get default context variables available to all templates
        
        Returns:
            Dictionary of default context variables
        """
        # Load Universal Schema from the schema directory
        schema_path = Path(__file__).parent.parent / 'schema' / 'universal_adapter.py'
        universal_schema = ""
        
        try:
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    universal_schema = f.read()
                logger.debug(f"Loaded Universal Schema from {schema_path} ({len(universal_schema)} characters)")
            else:
                logger.warning(f"Universal Schema not found at {schema_path}")
                universal_schema = "# Universal Schema not found - using fallback"
        except Exception as e:
            logger.error(f"Error loading Universal Schema from {schema_path}: {e}")
            universal_schema = f"# Error loading Universal Schema: {e}"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'timestamp_readable': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'engine_version': '1.1.0',
            'max_tokens': 4000,  # Default max tokens for safety
            'template_globals_available': ['datetime', 'uuid', 'random'],
            'universal_schema': universal_schema,  # ← Universal Schema loaded as raw text
        }
    
    def _truncate_code(self, code: str, max_lines: int = 1000) -> str:
        """
        Truncate code to maximum number of lines with informative suffix
        
        Args:
            code: Code string to truncate
            max_lines: Maximum number of lines to keep
            
        Returns:
            Truncated code string
        """
        if not isinstance(code, str):
            return str(code)
            
        lines = code.split('\n')
        if len(lines) <= max_lines:
            return code
        
        truncated = lines[:max_lines]
        lines_removed = len(lines) - max_lines
        truncated.append(f"\n... (truncated {lines_removed} lines for brevity)")
        return '\n'.join(truncated)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text using improved heuristics
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Improved token estimation:
        # - Account for whitespace compression
        # - Consider programming language tokens
        # - Rough approximation: 1 token ≈ 3.5-4 characters for code
        char_count = len(text)
        word_count = len(text.split())
        
        # Use the higher of character-based or word-based estimation
        char_based = char_count // 4
        word_based = int(word_count * 1.3)  # Programming tokens are often longer
        
        return max(char_based, word_based)
    
    def _post_process(self, rendered: str) -> str:
        """
        Post-process rendered template for cleaner output
        
        Args:
            rendered: Raw rendered template string
            
        Returns:
            Cleaned and formatted template string
        """
        if not rendered:
            return rendered
        
        lines = rendered.split('\n')
        processed = []
        blank_count = 0
        
        # Remove excessive blank lines while preserving intentional formatting
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                # Allow max 2 consecutive blank lines
                if blank_count <= 2:
                    processed.append(line)
            else:
                blank_count = 0
                processed.append(line)
        
        # Join and clean up leading/trailing whitespace
        result = '\n'.join(processed).strip()
        
        # Ensure the output ends with a single newline for better formatting
        if result and not result.endswith('\n'):
            result += '\n'
        
        return result