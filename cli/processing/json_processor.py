#!/usr/bin/env python3
"""
Conductor - Single JSON Request Processor
© 2025 QUANTUM ENCODING LTD
Contact: info@quantumencoding.io
Website: https://quantumencoding.io

Universal AI Conductor for single JSON requests with full multi-provider support.

This module processes individual JSON files containing one request/prompt using the
harvesting engine's flexible provider system and standardized model abstractions.

Usage:
    ./json_processor.py request.json --model gpt-5 --template advice
    ./json_processor.py prompt.json --model gemini-2.5-pro --template research  
    ./json_processor.py data.json --model gpt-1 --template code_assist
    ./json_processor.py - --model deepseek-chat --template quick < request.json
"""

import click
import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.templater import Templater
from providers.provider_factory import ProviderFactory
from utils.output_manager import OutputManager
from utils.progress_tracker import ProgressTracker

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Templates for AI tasks (using same as summon.py)
AI_TEMPLATES = {
    'advice': {
        'name': 'Strategic Advisor',
        'template': '''You are a wise strategic advisor with deep business and life experience.

**Situation:** {request}

**Advisory Approach:**
- Consider multiple perspectives and stakeholders
- Analyze risks and opportunities
- Provide both short-term and long-term views
- Include practical next steps
- Consider ethical implications
- Draw from relevant frameworks or case studies

**Structured Advice:**
1. **Situation Analysis:** Current state assessment
2. **Key Considerations:** Important factors to weigh
3. **Options:** Different paths forward with pros/cons
4. **Recommendation:** Your advised course of action
5. **Next Steps:** Concrete actions to take''',
        'category': 'advisory',
        'model_preference': 'gpt-5'
    },
    
    'research': {
        'name': 'Research Assistant',
        'template': '''You are a thorough research analyst. Conduct comprehensive analysis on this topic.

**Research Query:** {request}

**Research Methodology:**
- Provide well-sourced information
- Consider multiple viewpoints
- Include recent developments
- Identify gaps in knowledge
- Suggest further research directions
- Maintain objectivity

**Research Report:**
1. **Executive Summary:** Key findings
2. **Background:** Context and current state
3. **Key Findings:** Major insights and data
4. **Analysis:** Interpretation and implications
5. **Recommendations:** Suggested actions
6. **Further Research:** Areas needing more investigation''',
        'category': 'research',
        'model_preference': 'gemini-2.5-pro'
    },
    
    'code_assist': {
        'name': 'Code Assistant',
        'template': '''You are a senior software engineer and code mentor. Help with this coding task.

**Request:** {request}

**Guidelines:**
- Write clean, efficient, well-documented code
- Follow best practices and design patterns
- Include error handling where appropriate
- Explain your approach and decisions
- Suggest optimizations or improvements
- Consider scalability and maintainability

**Provide:**
1. Complete working code
2. Explanation of approach
3. Best practices applied
4. Potential improvements''',
        'category': 'development',
        'model_preference': 'vtx-2'
    },
    
    'explain': {
        'name': 'Concept Explainer',
        'template': '''You are an expert educator. Explain this concept clearly and comprehensively.

**Concept to Explain:** {request}

**Teaching Approach:**
- Start with fundamentals
- Build complexity gradually
- Use analogies and examples
- Anticipate common misconceptions
- Provide multiple explanations for different learning styles
- Include practical applications

**Explanation Structure:**
1. **Simple Definition:** What it is in plain terms
2. **Core Concepts:** Fundamental principles
3. **How It Works:** Mechanisms and processes
4. **Examples:** Real-world applications
5. **Common Misconceptions:** What people get wrong
6. **Advanced Insights:** Deeper understanding''',
        'category': 'education',
        'model_preference': 'vtx-2'
    },
    
    'quick': {
        'name': 'Quick Response',
        'template': '''{request}

Please provide a helpful, concise response.''',
        'category': 'general',
        'model_preference': 'goo-2'
    },
    
    'writing': {
        'name': 'Writing Assistant',
        'template': '''You are a skilled writer and editor. Help with this writing task.

**Writing Request:** {request}

**Writing Principles:**
- Clarity and concision
- Appropriate tone and style
- Strong structure and flow
- Engaging and persuasive content
- Proper grammar and mechanics
- Target audience consideration

**Deliver:**
1. **Well-crafted content** that meets the requirements
2. **Style notes** explaining choices made
3. **Alternative versions** if helpful
4. **Improvement suggestions** for future writing''',
        'category': 'creative',
        'model_preference': 'gpt-5'
    }
}

class Conductor:
    """Single JSON request processor using harvesting engine infrastructure."""
    
    def __init__(self):
        # Initialize harvesting engine components
        # Use project root config directory
        project_root = Path(__file__).resolve().parent.parent.parent
        config_dir = project_root / 'config'
        self.provider_factory = ProviderFactory(config_dir)
        self.templater = Templater(template_dir=project_root / 'templates')
        self.output_manager = OutputManager()

        click.echo(f"🎯 Conductor initialized with providers: {', '.join(self.provider_factory.list_providers())}")
    
    async def process_json_request(
        self,
        json_file: str,
        template_name: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        save_format: str = 'json',
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process single JSON request using specified template and model."""
        
        start_time = datetime.now()
        
        # Load JSON data
        json_data = self._load_json_file(json_file)
        if verbose:
            click.echo(f"📄 Loaded JSON: {json.dumps(json_data, indent=2)[:200]}...")
        
        # Get template configuration
        if template_name in AI_TEMPLATES:
            template_config = AI_TEMPLATES[template_name]
            template_content = template_config['template']
        else:
            # Try to load from file system templates
            try:
                template_obj = self.templater.load_template(template_name)
                template_content = template_obj.content
                template_config = {'name': template_name, 'category': 'custom'}
            except Exception as e:
                raise ValueError(f"Template '{template_name}' not found in built-in templates or file system: {e}")
        
        # Extract request from JSON (flexible field mapping)
        request = self._extract_request_content(json_data)
        
        # Render template
        rendered_prompt = template_content.format(request=request)
        
        if verbose:
            click.echo(f"🎯 Template: {template_config['name']}")
            click.echo(f"🤖 Model: {model}")
            click.echo(f"📝 Rendered prompt: {rendered_prompt[:200]}...")
        
        # Get provider and process request
        provider = self.provider_factory.get_provider(model)
        response = await provider.complete(rendered_prompt, model)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Structure result
        result = {
            'template_used': template_config['name'],
            'model_used': model,
            'original_json': json_data,
            'extracted_request': request,
            'rendered_prompt': rendered_prompt,
            'response': response,
            'processing_time': duration,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'category': template_config.get('category', 'general'),
                'temperature': temperature,
                'max_tokens': max_tokens,
                'conductor_processing': True
            }
        }
        
        # Save result using output manager
        saved_files = self.output_manager.save_response(
            result=result,
            template_name=template_name,
            model_used=model,
            auto_save=True
        )
        
        if saved_files:
            session_dir = saved_files['json'].parent
            click.echo(f"💾 Saved to: {session_dir.name}")
            formats = list(saved_files.keys())
            if 'code' in formats:
                click.echo(f"📁 Formats: {', '.join(formats)} (detected code content)")
            else:
                click.echo(f"📁 Formats: {', '.join(formats)}")
        
        return result
    
    def _load_json_file(self, json_file: str) -> Dict[str, Any]:
        """Load JSON from file or stdin."""
        try:
            if json_file == '-':
                # Read from stdin
                data = json.load(sys.stdin)
            else:
                # Read from file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValueError("JSON input must contain an object/dictionary")
            
            return data
            
        except Exception as e:
            source = "stdin" if json_file == '-' else f"file {json_file}"
            raise ValueError(f"Failed to load JSON from {source}: {e}")
    
    def _extract_request_content(self, json_data: Dict[str, Any]) -> str:
        """Extract the main request content from JSON with flexible field mapping."""
        
        # Try common field names in order of preference
        field_candidates = [
            'request', 'prompt', 'content', 'text', 'input', 'query', 
            'question', 'message', 'description', 'task'
        ]
        
        for field in field_candidates:
            if field in json_data:
                return str(json_data[field])
        
        # If no standard field found, look for any string value
        for key, value in json_data.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                return value
        
        # Last resort: convert entire JSON to string
        return json.dumps(json_data, indent=2)

def display_templates():
    """Display available built-in templates."""
    click.echo(click.style("\n🎯 Available Conductor Templates", fg='cyan', bold=True))
    
    categories = {}
    for name, config in AI_TEMPLATES.items():
        category = config['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((name, config))
    
    for category, templates in categories.items():
        click.echo(f"\n📁 {category.title()}")
        for name, config in templates:
            click.echo(f"  {name:<15} - {config['name']}")
            click.echo(f"                 Preferred: {config.get('model_preference', 'gpt-5')}")

def display_template_info(template_name: str):
    """Display detailed information about a template."""
    if template_name not in AI_TEMPLATES:
        click.echo(f"❌ Template '{template_name}' not found")
        return
    
    config = AI_TEMPLATES[template_name]
    
    click.echo(click.style(f"\n📋 Template: {template_name}", fg='cyan', bold=True))
    click.echo(f"Name: {config['name']}")
    click.echo(f"Category: {config['category']}")
    click.echo(f"Preferred Model: {config.get('model_preference', 'gpt-5')}")
    
    click.echo("\nTemplate Content:")
    click.echo("-" * 40)
    click.echo(config['template'][:300] + "..." if len(config['template']) > 300 else config['template'])

@click.command()
@click.argument('json_file', required=False)
@click.option('--template', '-t', default='quick',
              help='Template to use: advice, research, code_assist, explain, quick, writing')
@click.option('--model', '-m', default='vtx-2',
              help='Model: gpt-5, vtx-2, gemini-2.5-pro, goo-2, gpt-1, gpt-2, deepseek-chat, ds-2')
@click.option('--temperature', type=float, default=0.7,
              help='Temperature for generation (0.0-2.0)')
@click.option('--max-tokens', type=int, default=4096,
              help='Maximum tokens in response')
@click.option('--format', 'save_format', default='json',
              type=click.Choice(['json', 'markdown', 'text']),
              help='Save format for results')
@click.option('--list-templates', is_flag=True,
              help='List available templates')
@click.option('--template-info', help='Show info about specific template')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def conductor(
    json_file: Optional[str],
    template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    save_format: str,
    list_templates: bool,
    template_info: Optional[str],
    verbose: bool
):
    """
    🎯 Conductor - Single JSON Request Processor
    
    Process individual JSON files using the harvesting engine's flexible provider system.
    
    Examples:
        # Process JSON file with advice template
        ./json_processor.py request.json --template advice --model gpt-5
        
        # Research query with Google model
        ./json_processor.py query.json --template research --model gemini-2.5-pro
        
        # Code assistance with Claude
        ./json_processor.py code_request.json --template code_assist --model vtx-2
        
        # Quick response from stdin
        echo '{"request": "What is AI?"}' | ./json_processor.py - --template quick --model goo-2
    """
    
    # Handle utility commands
    if list_templates:
        display_templates()
        return
    
    if template_info:
        display_template_info(template_info)
        return
    
    # Validate required arguments
    if not json_file:
        click.echo("❌ JSON file required (use - for stdin)")
        click.echo("Use --list-templates to see available templates")
        sys.exit(1)
    
    if template not in AI_TEMPLATES:
        click.echo(f"❌ Unknown template: {template}")
        click.echo(f"Available templates: {', '.join(AI_TEMPLATES.keys())}")
        click.echo("Use --list-templates for details")
        sys.exit(1)
    
    # Show header
    click.echo(click.style(f"🎯 Conductor Processing", fg='cyan', bold=True))
    click.echo(f"📋 Template: {template}")
    click.echo(f"🤖 Model: {model}")
    click.echo(f"📄 Input: {json_file}")
    
    # Process the request
    async def execute():
        try:
            conductor_instance = Conductor()
            
            result = await conductor_instance.process_json_request(
                json_file=json_file,
                template_name=template,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                save_format=save_format,
                verbose=verbose
            )
            
            # Display result
            click.echo(f"\n✨ {result['template_used']} Result:")
            click.echo("=" * 50)
            click.echo(result['response'])
            click.echo("=" * 50)
            click.echo(f"Model: {result['model_used']} | Time: {result['processing_time']:.1f}s")
                
        except KeyboardInterrupt:
            click.echo(click.style("\n⚠️ Processing interrupted by user", fg='yellow'))
            sys.exit(1)
        except Exception as e:
            click.echo(click.style(f"\n❌ Processing failed: {e}", fg='red'))
            if verbose:
                logger.exception("Detailed error:")
            sys.exit(1)

    # Run the processing with cleanup
    async def run_with_cleanup():
        """Run processing and ensure cleanup"""
        try:
            await execute()
        finally:
            # Always clean up provider sessions
            try:
                if hasattr(conductor_instance, 'provider_factory'):
                    for provider_instance in conductor_instance.provider_factory.provider_instances.values():
                        if hasattr(provider_instance, 'close'):
                            await provider_instance.close()
            except Exception as e:
                logger.debug(f"Error during cleanup: {e}")

    asyncio.run(run_with_cleanup())

if __name__ == '__main__':
    conductor()