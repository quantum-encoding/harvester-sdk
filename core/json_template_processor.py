"""
JSON Template Processor for Image Ordering

Processes JSON order templates to create structured image generation requests
with automatic model routing, cost estimation, and batch optimization.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ImageOrderProcessor:
    """Processes JSON templates to create optimized image generation orders"""
    
    def __init__(self, templates_file: Optional[Path] = None):
        """
        Initialize with JSON templates
        
        Args:
            templates_file: Path to JSON templates file
        """
        self.templates_file = templates_file or Path(__file__).parent.parent / 'templates' / 'json_order_templates.json'
        self.templates = self._load_templates()
        
        logger.info(f"Loaded {len(self.templates.get('image_order_templates', {}))} template categories")
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load JSON templates from file"""
        try:
            with open(self.templates_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load templates from {self.templates_file}: {e}")
            return {}
    
    def create_order_from_template(
        self,
        template_name: str,
        scene_description: str,
        quantity: int = 1,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create image order from template name and scene description
        
        Args:
            template_name: Name of template (e.g., 'hyper-realistic', 'cyberpunk')
            scene_description: Description of the scene to generate
            quantity: Number of images to generate
            custom_params: Override template parameters
            
        Returns:
            Structured image order
        """
        # Find template in categories
        template_data = self._find_template(template_name)
        if not template_data:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Apply custom parameters
        if custom_params:
            template_data = {**template_data, **custom_params}
        
        # Build full prompt
        prefix = template_data.get('prefix', '')
        suffix = template_data.get('suffix', '')
        full_prompt = f"{prefix} {scene_description}, {suffix}".strip()
        
        # Create order structure
        order = {
            'template_used': template_name,
            'scene_description': scene_description,
            'full_prompt': full_prompt,
            'quantity': quantity,
            'parameters': {
                'style': template_data.get('style', 'vivid'),
                'quality': template_data.get('quality', 'standard'),
                'aspect_ratio': template_data.get('aspect_ratio', '1:1'),
                'model': template_data.get('model_preference', 'o1')
            },
            'metadata': {
                'category': template_data.get('category', 'general'),
                'description': template_data.get('description', ''),
                'cost_tier': template_data.get('cost_tier', 'standard'),
                'created_at': datetime.now().isoformat()
            },
            'cost_estimate': self._estimate_cost(template_data, quantity)
        }
        
        return order
    
    def create_batch_order(
        self,
        preset_name: str,
        scene_descriptions: List[str],
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create batch order from preset and multiple scene descriptions
        
        Args:
            preset_name: Name of batch preset (e.g., 'e_commerce_pack')
            scene_descriptions: List of scene descriptions
            custom_params: Override preset parameters
            
        Returns:
            Structured batch order
        """
        # Get preset configuration
        presets = self.templates.get('batch_order_presets', {})
        if preset_name not in presets:
            raise ValueError(f"Batch preset '{preset_name}' not found")
        
        preset = presets[preset_name]
        template_names = preset['templates']
        models = preset.get('models', ['o1'])
        quantity_per_template = preset.get('quantity_per_template', 1)
        
        # Apply custom parameters
        if custom_params:
            preset = {**preset, **custom_params}
            template_names = custom_params.get('templates', template_names)
            models = custom_params.get('models', models)
            quantity_per_template = custom_params.get('quantity_per_template', quantity_per_template)
        
        # Create individual orders for each scene/template combination
        orders = []
        total_cost = 0.0
        
        for scene_description in scene_descriptions:
            for template_name in template_names:
                try:
                    order = self.create_order_from_template(
                        template_name=template_name,
                        scene_description=scene_description,
                        quantity=quantity_per_template,
                        custom_params={'model': models[0] if models else 'o1'}
                    )
                    orders.append(order)
                    total_cost += order['cost_estimate']
                    
                except Exception as e:
                    logger.warning(f"Failed to create order for {template_name}: {e}")
        
        batch_order = {
            'preset_used': preset_name,
            'preset_description': preset['description'],
            'scene_descriptions': scene_descriptions,
            'individual_orders': orders,
            'batch_metadata': {
                'total_orders': len(orders),
                'total_images': sum(order['quantity'] for order in orders),
                'templates_used': template_names,
                'models_used': models,
                'estimated_total_cost': total_cost,
                'created_at': datetime.now().isoformat()
            }
        }
        
        return batch_order
    
    def create_custom_order(
        self,
        scene_descriptions: List[str],
        templates: List[str],
        models: Optional[List[str]] = None,
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create custom order with specific templates and models
        
        Args:
            scene_descriptions: List of scenes to generate
            templates: List of template names to use
            models: List of models to use (optional)
            custom_settings: Custom parameters for generation
            
        Returns:
            Custom structured order
        """
        if not models:
            models = ['o1']  # Default model
        
        orders = []
        total_cost = 0.0
        
        # Create all combinations of scenes × templates × models
        for scene_description in scene_descriptions:
            for template_name in templates:
                for model in models:
                    try:
                        custom_params = {'model': model}
                        if custom_settings:
                            custom_params.update(custom_settings)
                        
                        order = self.create_order_from_template(
                            template_name=template_name,
                            scene_description=scene_description,
                            quantity=1,
                            custom_params=custom_params
                        )
                        orders.append(order)
                        total_cost += order['cost_estimate']
                        
                    except Exception as e:
                        logger.warning(f"Failed to create custom order for {template_name}: {e}")
        
        custom_order = {
            'order_type': 'custom',
            'scene_descriptions': scene_descriptions,
            'templates_requested': templates,
            'models_requested': models,
            'individual_orders': orders,
            'custom_metadata': {
                'total_orders': len(orders),
                'total_images': len(orders),
                'estimated_total_cost': total_cost,
                'created_at': datetime.now().isoformat()
            }
        }
        
        return custom_order
    
    def optimize_order_for_cost(
        self,
        order: Dict[str, Any],
        target_tier: str = 'budget'
    ) -> Dict[str, Any]:
        """
        Optimize existing order for cost by adjusting models and quality
        
        Args:
            order: Existing order to optimize
            target_tier: Target cost tier ('budget', 'standard', 'premium')
            
        Returns:
            Optimized order
        """
        cost_tiers = self.templates.get('cost_tiers', {})
        if target_tier not in cost_tiers:
            logger.warning(f"Unknown cost tier: {target_tier}")
            return order
        
        tier_config = cost_tiers[target_tier]
        optimized_models = tier_config['models']
        target_quality = tier_config['quality']
        
        # Optimize individual orders
        if 'individual_orders' in order:
            for individual_order in order['individual_orders']:
                # Use first available model from tier
                current_model = individual_order['parameters']['model']
                if current_model not in optimized_models:
                    individual_order['parameters']['model'] = optimized_models[0]
                
                # Adjust quality
                individual_order['parameters']['quality'] = target_quality
                
                # Recalculate cost
                template_data = self._find_template(individual_order['template_used'])
                if template_data:
                    template_data['cost_tier'] = target_tier
                    individual_order['cost_estimate'] = self._estimate_cost(
                        template_data, individual_order['quantity']
                    )
            
            # Update total cost
            if 'batch_metadata' in order:
                order['batch_metadata']['estimated_total_cost'] = sum(
                    o['cost_estimate'] for o in order['individual_orders']
                )
            elif 'custom_metadata' in order:
                order['custom_metadata']['estimated_total_cost'] = sum(
                    o['cost_estimate'] for o in order['individual_orders']
                )
        
        else:
            # Single order optimization
            current_model = order['parameters']['model']
            if current_model not in optimized_models:
                order['parameters']['model'] = optimized_models[0]
            
            order['parameters']['quality'] = target_quality
            order['metadata']['cost_tier'] = target_tier
            
            # Recalculate cost
            template_data = self._find_template(order['template_used'])
            if template_data:
                template_data['cost_tier'] = target_tier
                order['cost_estimate'] = self._estimate_cost(
                    template_data, order['quantity']
                )
        
        return order
    
    def convert_order_to_csv_format(self, order: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert structured order to CSV format for processing
        
        Args:
            order: Structured order from create_order_* methods
            
        Returns:
            List of CSV records
        """
        csv_records = []
        
        if 'individual_orders' in order:
            # Batch or custom order
            for individual_order in order['individual_orders']:
                csv_record = {
                    'prompt': individual_order['full_prompt'],
                    'style': individual_order['parameters']['style'],
                    'quality': individual_order['parameters']['quality'],
                    'aspect_ratio': individual_order['parameters']['aspect_ratio'],
                    'model': individual_order['parameters']['model'],
                    'template_used': individual_order['template_used'],
                    'scene_description': individual_order['scene_description'],
                    'category': individual_order['metadata']['category'],
                    'cost_tier': individual_order['metadata']['cost_tier'],
                    'estimated_cost': individual_order['cost_estimate']
                }
                csv_records.append(csv_record)
        else:
            # Single order
            csv_record = {
                'prompt': order['full_prompt'],
                'style': order['parameters']['style'],
                'quality': order['parameters']['quality'],
                'aspect_ratio': order['parameters']['aspect_ratio'],
                'model': order['parameters']['model'],
                'template_used': order['template_used'],
                'scene_description': order['scene_description'],
                'category': order['metadata']['category'],
                'cost_tier': order['metadata']['cost_tier'],
                'estimated_cost': order['cost_estimate']
            }
            csv_records.append(csv_record)
        
        return csv_records
    
    def _find_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Find template by name across all categories"""
        templates = self.templates.get('image_order_templates', {})
        
        for category, category_templates in templates.items():
            if template_name in category_templates:
                return category_templates[template_name]
        
        return None
    
    def _estimate_cost(self, template_data: Dict[str, Any], quantity: int) -> float:
        """Estimate cost for template and quantity"""
        cost_tier = template_data.get('cost_tier', 'standard')
        cost_tiers = self.templates.get('cost_tiers', {})
        
        if cost_tier in cost_tiers:
            cost_per_image = cost_tiers[cost_tier]['cost_per_image']
        else:
            cost_per_image = 0.040  # Default cost
        
        return cost_per_image * quantity
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available templates by category"""
        templates = self.templates.get('image_order_templates', {})
        return {
            category: list(category_templates.keys())
            for category, category_templates in templates.items()
        }
    
    def list_batch_presets(self) -> Dict[str, str]:
        """List all available batch presets with descriptions"""
        presets = self.templates.get('batch_order_presets', {})
        return {
            name: preset['description']
            for name, preset in presets.items()
        }
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific template"""
        template_data = self._find_template(template_name)
        if not template_data:
            return None
        
        return {
            'name': template_name,
            'description': template_data.get('description', ''),
            'category': template_data.get('category', 'general'),
            'cost_tier': template_data.get('cost_tier', 'standard'),
            'preferred_model': template_data.get('model_preference', 'o1'),
            'default_style': template_data.get('style', 'vivid'),
            'default_quality': template_data.get('quality', 'standard'),
            'default_aspect_ratio': template_data.get('aspect_ratio', '1:1'),
            'estimated_cost_per_image': self._estimate_cost(template_data, 1)
        }


def create_order_processor(templates_file: Optional[Path] = None) -> ImageOrderProcessor:
    """Convenience function to create order processor"""
    return ImageOrderProcessor(templates_file)