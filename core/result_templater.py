"""
Result Templater for Image Generation Responses

Applies Jinja2 templates to format and process image generation results,
providing flexible output formatting and conditional logic.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ImageResultTemplater:
    """Templater for formatting image generation results"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize result templater
        
        Args:
            template_dir: Directory containing result templates
        """
        self.template_dir = template_dir or Path(__file__).parent.parent / 'templates' / 'results'
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['json_pretty'] = self._json_pretty_filter
        self.env.filters['format_cost'] = self._format_cost_filter
        self.env.filters['format_duration'] = self._format_duration_filter
        self.env.filters['extract_filename'] = self._extract_filename_filter
        self.env.filters['success_rate'] = self._success_rate_filter
        
        # Add custom functions
        self.env.globals['now'] = datetime.now
        self.env.globals['format_timestamp'] = self._format_timestamp
        
        logger.info(f"Result templater initialized with templates from {self.template_dir}")
    
    def render_result(
        self,
        template_name: str,
        result_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render image generation result using specified template
        
        Args:
            template_name: Name of template file (without .j2 extension)
            result_data: Image generation result data
            context: Additional context variables
            
        Returns:
            Rendered template as string
        """
        template_file = f"{template_name}.j2"
        
        try:
            template = self.env.get_template(template_file)
        except Exception as e:
            logger.error(f"Failed to load template {template_file}: {e}")
            # Fallback to default template
            return self._render_default_template(result_data, context)
        
        # Prepare template context
        template_context = {
            'result': result_data,
            'context': context or {},
            'timestamp': datetime.now(),
            'batch_info': self._extract_batch_info(result_data)
        }
        
        try:
            rendered = template.render(**template_context)
            logger.debug(f"Successfully rendered result with template {template_name}")
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            return self._render_default_template(result_data, context)
    
    def render_batch_summary(
        self,
        template_name: str,
        batch_results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render batch summary using specified template
        
        Args:
            template_name: Name of template file
            batch_results: List of individual image generation results
            summary_stats: Batch statistics
            context: Additional context variables
            
        Returns:
            Rendered batch summary
        """
        template_file = f"{template_name}.j2"
        
        try:
            template = self.env.get_template(template_file)
        except Exception as e:
            logger.error(f"Failed to load batch template {template_file}: {e}")
            return self._render_default_batch_template(batch_results, summary_stats)
        
        # Prepare template context
        template_context = {
            'batch_results': batch_results,
            'summary': summary_stats,
            'context': context or {},
            'timestamp': datetime.now(),
            'total_images': len([r for r in batch_results if r.get('success')]),
            'failed_images': len([r for r in batch_results if not r.get('success')]),
            'models_used': list(set(r.get('model', 'unknown') for r in batch_results))
        }
        
        try:
            rendered = template.render(**template_context)
            logger.debug(f"Successfully rendered batch summary with template {template_name}")
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render batch template {template_name}: {e}")
            return self._render_default_batch_template(batch_results, summary_stats)
    
    def create_default_templates(self):
        """Create default result templates"""
        
        # Single result template
        single_result_template = """
{# Single Image Generation Result Template #}
{% if result.success %}
✅ **Image Generated Successfully**

**Model:** {{ result.model }}
**Prompt:** {{ result.original_prompt }}
{% if result.revised_prompt %}**Revised Prompt:** {{ result.revised_prompt }}{% endif %}

**Generated Images:**
{% for image in result.images %}
- Image {{ loop.index }}: {{ image.url if image.url else '[Base64 Data Available]' }}
  {% if image.revised_prompt %}*Revised: {{ image.revised_prompt }}*{% endif %}
{% endfor %}

**Generation Details:**
- Cost: ${{ result.cost | format_cost }}
- Processing Time: {{ result.processing_time | format_duration }}
- Timestamp: {{ result.timestamp | format_timestamp }}
{% if result.parameters %}
- Parameters: {{ result.parameters | json_pretty }}
{% endif %}

{% else %}
❌ **Image Generation Failed**

**Model:** {{ result.model }}
**Prompt:** {{ result.original_prompt }}
**Error:** {{ result.error }}
**Timestamp:** {{ result.timestamp | format_timestamp }}
{% endif %}

---
Generated at {{ timestamp | format_timestamp }}
        """
        
        # Batch summary template
        batch_summary_template = """
{# Batch Image Generation Summary Template #}
# 🎨 Image Generation Batch Summary

**Batch Completed:** {{ timestamp | format_timestamp }}
**Total Requests:** {{ batch_results | length }}
**Successful:** {{ total_images }}
**Failed:** {{ failed_images }}
**Success Rate:** {{ (total_images, failed_images) | success_rate }}%

## 📊 Statistics
- **Total Cost:** ${{ summary.total_cost | format_cost }}
- **Average Cost per Image:** ${{ (summary.total_cost / total_images) | format_cost if total_images > 0 else 0 }}
- **Total Processing Time:** {{ summary.total_processing_time | format_duration }}
- **Models Used:** {{ models_used | join(', ') }}

## 🤖 Model Performance
{% for model in models_used %}
{% set model_results = batch_results | selectattr('model', 'equalto', model) | list %}
{% set model_success = model_results | selectattr('success', 'equalto', true) | list | length %}
{% set model_total = model_results | length %}
- **{{ model }}:** {{ model_success }}/{{ model_total }} ({{ (model_success / model_total * 100) | round(1) }}% success)
{% endfor %}

## 🖼️ Generated Images
{% for result in batch_results %}
{% if result.success %}
### Image {{ loop.index }}
- **Prompt:** {{ result.original_prompt }}
- **Model:** {{ result.model }}
- **Cost:** ${{ result.cost | format_cost }}
{% for image in result.images %}
- **URL:** {{ image.url if image.url else '[Base64 Available]' }}
{% endfor %}
{% endif %}
{% endfor %}

{% if failed_images > 0 %}
## ❌ Failed Generations
{% for result in batch_results %}
{% if not result.success %}
### Failed Request {{ loop.index }}
- **Prompt:** {{ result.original_prompt }}
- **Model:** {{ result.model }}
- **Error:** {{ result.error }}
{% endif %}
{% endfor %}
{% endif %}

---
*Generated by Image Batch Processor at {{ timestamp | format_timestamp }}*
        """
        
        # JSON output template
        json_output_template = """
{# JSON Output Template for API/Integration Use #}
{
  "batch_summary": {
    "timestamp": "{{ timestamp | format_timestamp }}",
    "total_requests": {{ batch_results | length }},
    "successful_images": {{ total_images }},
    "failed_requests": {{ failed_images }},
    "success_rate": {{ (total_images, failed_images) | success_rate }},
    "total_cost": {{ summary.total_cost }},
    "models_used": {{ models_used | tojson }},
    "processing_time": {{ summary.total_processing_time }}
  },
  "results": [
    {% for result in batch_results %}
    {
      "success": {{ result.success | tojson }},
      "model": {{ result.model | tojson }},
      "original_prompt": {{ result.original_prompt | tojson }},
      {% if result.success %}
      "images": {{ result.images | tojson }},
      "cost": {{ result.cost }},
      "processing_time": {{ result.processing_time }}
      {% else %}
      "error": {{ result.error | tojson }}
      {% endif %}
    }{% if not loop.last %},{% endif %}
    {% endfor %}
  ]
}
        """
        
        # HTML report template  
        html_report_template = """
{# HTML Report Template #}
<!DOCTYPE html>
<html>
<head>
    <title>Image Generation Batch Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-box { background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .image-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
        .success { border-left: 4px solid #4CAF50; }
        .failure { border-left: 4px solid #f44336; }
        .prompt { font-style: italic; color: #666; margin: 10px 0; }
        .metadata { font-size: 0.9em; color: #888; }
        img { max-width: 100%; height: auto; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Image Generation Batch Report</h1>
        <p><strong>Generated:</strong> {{ timestamp | format_timestamp }}</p>
        <p><strong>Batch ID:</strong> {{ context.batch_id or 'N/A' }}</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <h3>{{ total_images }}</h3>
            <p>Successful Images</p>
        </div>
        <div class="stat-box">
            <h3>{{ failed_images }}</h3>
            <p>Failed Requests</p>
        </div>
        <div class="stat-box">
            <h3>{{ (total_images, failed_images) | success_rate }}%</h3>
            <p>Success Rate</p>
        </div>
        <div class="stat-box">
            <h3>${{ summary.total_cost | format_cost }}</h3>
            <p>Total Cost</p>
        </div>
    </div>

    <h2>Generated Images</h2>
    <div class="image-grid">
        {% for result in batch_results %}
        <div class="image-card {% if result.success %}success{% else %}failure{% endif %}">
            {% if result.success %}
                <h4>✅ Image {{ loop.index }}</h4>
                <div class="prompt">{{ result.original_prompt }}</div>
                {% for image in result.images %}
                {% if image.url %}
                <img src="{{ image.url }}" alt="Generated Image" />
                {% endif %}
                {% endfor %}
                <div class="metadata">
                    Model: {{ result.model }} | Cost: ${{ result.cost | format_cost }} | 
                    Time: {{ result.processing_time | format_duration }}
                </div>
            {% else %}
                <h4>❌ Failed Request {{ loop.index }}</h4>
                <div class="prompt">{{ result.original_prompt }}</div>
                <div style="color: #f44336;">{{ result.error }}</div>
                <div class="metadata">Model: {{ result.model }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
</body>
</html>
        """
        
        # Write templates to files
        templates = {
            'single_result.j2': single_result_template,
            'batch_summary.j2': batch_summary_template,
            'json_output.j2': json_output_template,
            'html_report.j2': html_report_template
        }
        
        for filename, content in templates.items():
            template_path = self.template_dir / filename
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        logger.info(f"Created {len(templates)} default templates in {self.template_dir}")
    
    def _json_pretty_filter(self, value: Any) -> str:
        """Format value as pretty JSON"""
        try:
            return json.dumps(value, indent=2, default=str)
        except:
            return str(value)
    
    def _format_cost_filter(self, value: float) -> str:
        """Format cost value"""
        try:
            return f"{float(value):.4f}"
        except:
            return "0.0000"
    
    def _format_duration_filter(self, value: Union[int, float]) -> str:
        """Format duration in seconds"""
        try:
            seconds = float(value)
            if seconds < 60:
                return f"{seconds:.1f}s"
            else:
                minutes = seconds // 60
                remaining_seconds = seconds % 60
                return f"{int(minutes)}m {remaining_seconds:.1f}s"
        except:
            return "0.0s"
    
    def _extract_filename_filter(self, url: str) -> str:
        """Extract filename from URL"""
        try:
            return Path(url).name
        except:
            return url
    
    def _success_rate_filter(self, value: tuple) -> float:
        """Calculate success rate from (success_count, failure_count)"""
        try:
            success, failure = value
            total = success + failure
            return round((success / total * 100) if total > 0 else 0, 1)
        except:
            return 0.0
    
    def _format_timestamp(self, dt: datetime) -> str:
        """Format timestamp for display"""
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def _extract_batch_info(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract batch-level information from result data"""
        return {
            'batch_id': result_data.get('batch_id'),
            'model': result_data.get('model'),
            'provider': result_data.get('provider'),
            'timestamp': result_data.get('timestamp')
        }
    
    def _render_default_template(self, result_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Fallback template when main template fails"""
        return f"""
IMAGE GENERATION RESULT
======================
Success: {result_data.get('success', False)}
Model: {result_data.get('model', 'unknown')}
Prompt: {result_data.get('original_prompt', 'N/A')}
{'Error: ' + result_data.get('error', '') if not result_data.get('success') else ''}
Timestamp: {datetime.now()}
        """.strip()
    
    def _render_default_batch_template(self, batch_results: List[Dict[str, Any]], summary_stats: Dict[str, Any]) -> str:
        """Fallback batch template when main template fails"""
        successful = len([r for r in batch_results if r.get('success')])
        total = len(batch_results)
        
        return f"""
BATCH GENERATION SUMMARY
========================
Total: {total}
Successful: {successful}
Failed: {total - successful}
Success Rate: {(successful/total*100) if total > 0 else 0:.1f}%
Total Cost: ${summary_stats.get('total_cost', 0):.4f}
Timestamp: {datetime.now()}
        """.strip()
    
    def list_templates(self) -> List[str]:
        """List available templates"""
        templates = []
        for template_file in self.template_dir.glob('*.j2'):
            templates.append(template_file.stem)
        return sorted(templates)


# Convenience function
def create_result_templater(template_dir: Optional[Path] = None) -> ImageResultTemplater:
    """Create result templater and ensure default templates exist"""
    templater = ImageResultTemplater(template_dir)
    
    # Create default templates if none exist
    if not any(templater.template_dir.glob('*.j2')):
        templater.create_default_templates()
    
    return templater