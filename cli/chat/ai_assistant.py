#!/usr/bin/env python3
"""
Sacred Wrapper - Universal AI Assistant

A unified orchestrator for all AI providers with intelligent model routing,
context management, and battle-tested reliability.
"""

import click
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvester_sdk.sdk import HarvesterSDK
from utils.output_paths import generate_cli_output_directory

def get_valid_models() -> List[str]:
    """Get list of valid models from providers configuration"""
    try:
        config_path = Path(__file__).parent / 'config' / 'providers.yaml'
        with open(config_path, 'r') as f:
            providers_config = yaml.safe_load(f)
        
        valid_models = []
        for provider_name, provider_config in providers_config.get('providers', {}).items():
            if 'aliases' in provider_config:
                valid_models.extend(provider_config['aliases'].keys())
        
        return sorted(valid_models)
    except Exception as e:
        return ['gemini-2.5-pro', 'gemini-2.5-flash', 'gpt-5-nano', 'claude-4-5-sonnet']

@click.command()
@click.option('--prompt', '-p', help='Text prompt for the AI')
@click.option('--file', '-f', help='Read prompt from file')
@click.option('--model', '-m', default='gemini-2.5-flash', help='AI model to use')
@click.option('--output', '-o', help='Custom output directory (uses sovereign structure by default)')
@click.option('--temperature', '-t', default=0.7, help='Creativity level (0.0-2.0)')
@click.option('--max-tokens', default=4000, help='Maximum response length')
@click.option('--system', '-s', help='System prompt/context')
@click.option('--json', 'output_json', is_flag=True, help='Output response as JSON')
@click.option('--save', is_flag=True, help='Save conversation to file')
@click.option('--list-models', is_flag=True, help='List available models')
def main(prompt, file, model, output, temperature, max_tokens, system, output_json, save, list_models):
    """
    Sacred Wrapper - Universal AI Assistant
    
    Chat with any AI provider through a unified interface with intelligent routing.
    
    Examples:
        # Quick chat
        ai-assistant -p "Explain quantum computing" -m gemini-2.5-flash
        
        # Read prompt from file
        ai-assistant -f prompt.txt -m gpt-5-nano --save
        
        # JSON output with system context
        ai-assistant -p "Analyze this data" -s "You are a data scientist" --json
        
        # Creative writing with high temperature
        ai-assistant -p "Write a story" -t 1.5 -m claude-3-5-sonnet
    """
    
    # List models and exit
    if list_models:
        click.echo("📋 Available Models:")
        models = get_valid_models()
        for model_name in models:
            click.echo(f"  • {model_name}")
        return 0
    
    # Get prompt from file or command line
    if file:
        file_path = Path(file)
        if not file_path.exists():
            click.echo(f"❌ File not found: {file}")
            return 1
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    
    if not prompt:
        click.echo("❌ No prompt provided. Use -p or -f option.")
        return 1
    
    # Generate sovereign output directory for saved conversations
    if save and not output:
        output = generate_cli_output_directory("ai_assistant", prompt[:50])
    
    click.echo(f"🤖 Sacred Wrapper SDK - AI Assistant")
    click.echo(f"🧠 Model: {model}")
    click.echo(f"🌡️  Temperature: {temperature}")
    click.echo(f"📊 Max tokens: {max_tokens}")
    if system:
        click.echo(f"⚙️  System: {system[:60]}...")
    if save:
        click.echo(f"📂 Output: {output}")
    
    try:
        # Initialize SDK
        sdk = HarvesterSDK()
        
        # Prepare parameters
        params = {
            'prompt': prompt,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        if system:
            params['system'] = system
        
        click.echo("🧠 Thinking...")
        
        # Generate response
        response = sdk.generate_text(**params)
        
        # Prepare conversation data
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'parameters': {
                'temperature': temperature,
                'max_tokens': max_tokens
            },
            'system': system,
            'prompt': prompt,
            'response': response
        }
        
        # Output response
        if output_json:
            click.echo(json.dumps(conversation, indent=2, ensure_ascii=False))
        else:
            click.echo("\n" + "="*60)
            click.echo("🤖 AI Response:")
            click.echo("="*60)
            click.echo(response)
            click.echo("="*60)
        
        # Save conversation if requested
        if save:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            conversation_file = output_path / f"conversation_{timestamp}.json"
            
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
            
            click.echo(f"💾 Conversation saved: {conversation_file}")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        return 1

if __name__ == '__main__':
    main()
