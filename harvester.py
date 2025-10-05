#!/usr/bin/env python3
"""
Harvester SDK - Unified CLI
© 2025 QUANTUM ENCODING LTD
Contact: info@quantumencoding.io
Website: https://quantumencoding.io

The Master Conductor for all AI processing capabilities

This is the central command interface for the entire SDK, providing unified access to:
- Batch text processing from CSV
- Directory processing with templates
- Image generation and processing
- Interactive chat interfaces
- Live search capabilities

Usage:
    harvester batch --provider openai /path/to/data.csv
    harvester process --template refactor --model gpt-5 /path/to/code
    harvester image --provider dalle3 "A beautiful sunset"
    harvester chat --provider grok
    harvester search "latest AI news" --provider grok
"""

import click
import sys
import os
import asyncio
import subprocess
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

@click.group()
@click.version_option(version='1.0.0', prog_name='Harvester SDK by Quantum Encoding Ltd')
def cli():
    """
    🚀 Harvester SDK - Unified CLI
    
    © 2025 QUANTUM ENCODING LTD
    📧 Contact: info@quantumencoding.io
    🌐 Website: https://quantumencoding.io
    
    The Master Conductor for all AI processing capabilities.
    Use 'harvester COMMAND --help' for more information on each command.
    """
    pass

@cli.command('batch')
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--provider', '-p', help='AI provider to use')
@click.option('--model', '-m', help='Model to use (or group like grp-fast, grp-quality)')
@click.option('--template', '-t', default='default', help='Template to apply')
@click.option('--output', '-o', help='Output directory')
@click.option('--parallel', default=5, help='Number of parallel workers')
@click.option('--image', is_flag=True, help='Process as image generation batch')
def batch_command(csv_file, provider, model, template, output, parallel, image):
    """Process CSV batch with AI providers (text or image)"""
    if image:
        click.echo("🎨 Batch image processing")
        # Route to image batch processor
        cmd = [
            sys.executable,
            'cli/image/linear_vertex_image.py',  # Linear/anti-fragile image processor
            csv_file
        ]
        if model:
            cmd.extend(['--model', model])
        if output:
            cmd.extend(['--output', output])
    else:
        click.echo("📝 Batch text processing from CSV")
        # Use csv_processor for CSV batch processing
        cmd = [
            sys.executable,
            'cli/processing/csv_processor.py',
            'process',
            csv_file
        ]
        if model:
            cmd.extend(['--model', model])
        if template:
            cmd.extend(['--template', template])
        if output:
            cmd.extend(['--output', output])
    
    subprocess.run(cmd)

@cli.command('process')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--template', '-t', required=True, help='Template number or name to use')
@click.option('--model', '-m', default='gemini-2.5-flash', help='Model to use (or "all" for multi-provider)')
@click.option('--parallel', '-p', default=20, help='Number of parallel workers')
@click.option('--pattern', default='**/*', help='File pattern to match')
@click.option('--output', '-o', help='Output directory')
@click.option('--max-files', default=100, help='Maximum files to process')
def process_command(directory, template, model, parallel, pattern, output, max_files):
    """Process directory with templates (formerly batch_code)"""
    click.echo(f"📁 Processing directory: {directory}")
    click.echo(f"📋 Template: {template}")
    
    # Check for --model all flag
    if model.lower() == 'all':
        click.echo("🌌 GALACTIC FEDERATION MODE ACTIVATED")
        click.echo("⚡ Processing with ALL providers in parallel!")
    
    click.echo(f"🤖 Model: {model}")
    click.echo(f"⚡ Parallel workers: {parallel}")
    
    # Route to parallel_template_cli (Crown Jewel processor)
    template_with_ext = f'{template}.j2' if not template.endswith('.j2') else template

    cmd = [
        sys.executable,
        'parallel_template_cli.py',
        '--source', directory,
        '--template', template_with_ext,
        '--model', model,
        '--workers', str(parallel),
        '--file-pattern', pattern,
        '--max-files', str(max_files)
    ]

    if output:
        cmd.extend(['--output', output])

    subprocess.run(cmd)

@cli.command('image')
@click.argument('prompt', required=False)
@click.option('--model', '-m', default='nano-banana', help='Image model (dalle-3, nano-banana, imagen-4-ultra, imagen-4-fast, gpt-image, grok-image)')
@click.option('--template', '-t', help='Style template (cosmic_duck, professional, etc.)')
@click.option('--size', default='1024x1024', help='Image size (1024x1024, 1792x1024, etc.)')
@click.option('--save', '-s', help='Save image to specific file path')
@click.option('--batch', type=click.Path(exists=True), help='Process batch from CSV file')
def image_command(prompt, model, template, size, save, batch):
    """Generate images with AI models
    
    Two ways to generate images:

    1. Single image with command args:
        harvester image "two ducks and a swan" --model dalle-3
        harvester image "cosmic scene" --model nano-banana --template cosmic_duck
        harvester image "professional headshot" --model imagen-4-ultra
        harvester image "futuristic city" --model grok-image

    2. Batch generation from CSV:
        harvester image --batch images.csv

        CSV format: prompt,model,template,size,save
        Example row: "two ducks,dalle-3,cosmic_duck,1024x1024,duck.png"

    Available models:
        • nano-banana       - Gemini 2.5 Flash Image (fastest, default)
        • dalle-3           - DALL-E 3 (OpenAI)
        • imagen-4-ultra    - Imagen 4 Ultra (highest quality, needs GCP)
        • imagen-4-fast     - Imagen 4 Fast (needs GCP)
        • imagen-4          - Imagen 4 Standard (needs GCP)
        • gpt-image         - GPT Image (OpenAI)
        • grok-image        - Grok 2 Image (xAI)
    """
    
    # Validate that either prompt or batch is provided
    if not batch and not prompt:
        click.echo("❌ Either provide a prompt or use --batch with a CSV file")
        click.echo("\nExamples:")
        click.echo("  harvester image 'two ducks and a swan' --model dalle-3")
        click.echo("  harvester image --batch images.csv")
        return

    # Apply template to prompt if specified
    if template and prompt:
        if template == 'cosmic_duck':
            prompt = f"{prompt}, cosmic nebula background, ethereal lighting, space fantasy art style"
        elif template == 'professional':
            prompt = f"{prompt}, professional photography, clean background, high quality"
        elif template == 'artistic':
            prompt = f"{prompt}, digital art, vibrant colors, artistic composition"
        # Add more templates as needed
    
    if batch:
        click.echo(f"🎨 Batch image generation from: {batch}")
        click.echo("📋 CSV should have columns: prompt, model, template (optional), size (optional), filename (optional)")
        click.echo("   Example: 'two ducks,dalle-3,cosmic_duck,1024x1024,duck_image.png'")
        click.echo()
        
        # Check if this is a vertex/imagen batch that should use vertex_batch_ultra.py
        try:
            import pandas as pd
            df = pd.read_csv(batch)
            models_in_batch = df['model'].str.lower() if 'model' in df.columns else ['dalle-3']
            
            # If any imagen models detected, offer vertex batch processing
            imagen_models = [m for m in models_in_batch if 'imagen' in str(m)]
            if imagen_models:
                vertex_script = Path(__file__).parent.parent / 'knowledge-query-system' / 'harvesting_engine' / 'vertex_batch_ultra.py'
                if vertex_script.exists():
                    click.echo(f"🚀 Detected Imagen models: {set(imagen_models)}")
                    click.echo(f"💡 For optimized Imagen processing, consider using:")
                    click.echo(f"   python {vertex_script} process {batch} --concurrency 15")
                    click.echo()
        except Exception:
            pass
        
        cmd = [sys.executable, 'cli/image/image_cli.py', 'batch', batch]
        # Let the image_cli handle the model mapping and templates
    else:
        click.echo(f"🎨 Generating image with {model}: {prompt[:50]}...")

        # Map friendly names to canonical aliases (pass through if already correct)
        model_mapping = {
            # OpenAI
            'dall-e-3': 'dalle-3',
            # Google GenAI (Nano Banana!)
            'gemini-flash-image': 'nano-banana',
            'gemini-2.5-flash-image': 'nano-banana',
            'genai-flash-img': 'nano-banana',  # Old alias
            # Google Vertex (Imagen 4) - already clean, pass through
            # xAI - already clean, pass through
            # GPT Image variations
            'gpt-image-1': 'gpt-image',
        }

        mapped_model = model_mapping.get(model.lower(), model)

        cmd = [
            sys.executable, 'cli/image/image_cli.py',
            '--prompt', prompt,
            '--model', mapped_model,
            '--size', size
        ]

        if save:
            cmd.extend(['--output', str(Path(save).parent)])

        # Apply style/quality based on template
        if template == 'cosmic_duck':
            cmd.extend(['--style', 'vivid', '--quality', 'hd'])
        elif template == 'professional':
            cmd.extend(['--style', 'natural', '--quality', 'hd'])
        elif template == 'artistic':
            cmd.extend(['--style', 'vivid'])
    
    subprocess.run(cmd)

@cli.command('templates')
@click.option('--type', '-t', default='image', type=click.Choice(['image', 'text', 'batch']), help='Template type')
@click.option('--category', '-c', help='Template category (blog, product, universal, etc.)')
@click.option('--list', '-l', is_flag=True, help='List available templates')
@click.option('--copy', help='Copy template to current directory')
def templates_command(type, category, list, copy):
    """Manage batch processing templates
    
    Examples:
        harvester templates --list                    # List all templates
        harvester templates --copy image_batch_blog   # Copy blog template
        harvester templates --type image --category blog  # Show blog image template
    """
    templates_dir = Path(__file__).parent / 'templates'
    
    if list:
        click.echo("📋 Available Templates:")
        click.echo()
        
        # Image batch templates
        click.echo("🎨 Image Batch Templates:")
        for template_file in templates_dir.glob('image_batch_*.csv'):
            category_name = template_file.stem.replace('image_batch_', '')
            click.echo(f"   • image_batch_{category_name}")
            
            # Show first few rows as preview
            try:
                import pandas as pd
                df = pd.read_csv(template_file)
                click.echo(f"     Columns: {', '.join(df.columns)}")
                click.echo(f"     Examples: {len(df)} rows")
                click.echo(f"     Usage: harvester templates --copy image_batch_{category_name}")
            except:
                pass
            click.echo()
        
        click.echo("💡 Create your own prompts:")
        click.echo("   1. Copy a template: harvester templates --copy image_batch_blog")
        click.echo("   2. Edit the CSV with your prompts")  
        click.echo("   3. Process: harvester image --batch your_prompts.csv")
        click.echo()
        click.echo("🤖 Or ask any AI to fill the template:")
        click.echo("   'Hey Claude/Gemini, create 50 blog header prompts using this CSV format'")
        return
    
    if copy:
        template_file = templates_dir / f"{copy}.csv"
        if template_file.exists():
            import shutil
            dest = Path.cwd() / f"{copy}.csv"
            shutil.copy2(template_file, dest)
            click.echo(f"✅ Copied template to: {dest}")
            click.echo(f"📝 Edit the CSV and run: harvester image --batch {dest}")
        else:
            click.echo(f"❌ Template '{copy}' not found. Use --list to see available templates")
        return
    
    if type == 'image' and category:
        template_file = templates_dir / f"image_batch_{category}.csv"
        if template_file.exists():
            click.echo(f"📋 Template: image_batch_{category}")
            click.echo(f"📁 File: {template_file}")
            click.echo()
            
            # Show template content
            try:
                with open(template_file, 'r') as f:
                    lines = f.readlines()[:6]  # Show header + 5 examples
                    for i, line in enumerate(lines):
                        if i == 0:
                            click.echo(f"Header: {line.strip()}")
                        else:
                            click.echo(f"Row {i}: {line.strip()[:80]}...")
            except:
                pass
            
            click.echo(f"\n💡 Copy with: harvester templates --copy image_batch_{category}")
        else:
            click.echo(f"❌ Template 'image_batch_{category}' not found")

@cli.command('chat')
@click.option('--provider', '-p', default='grok', help='Chat provider (grok, gemini, openai, claude, deepseek)')
@click.option('--model', '-m', help='Specific model to use')
@click.option('--search', is_flag=True, help='Enable web search (Grok, Gemini, OpenAI, Claude)')
@click.option('--fetch', is_flag=True, help='Enable web fetch (Claude)')
@click.option('--functions', is_flag=True, help='Enable function calling (Grok, Gemini, OpenAI, Claude, DeepSeek)')
@click.option('--memory', is_flag=True, help='Enable memory tool (Claude)')
@click.option('--code', is_flag=True, help='Enable code execution (Claude)')
def chat_command(provider, model, search, fetch, functions, memory, code):
    """Start interactive streaming chat with AI provider

    Examples:
        harvester chat                                    # Grok-4 (default)
        harvester chat --provider gemini                  # Gemini 2.5 Flash
        harvester chat --provider openai                  # GPT-5
        harvester chat --provider claude                  # Claude Sonnet 4
        harvester chat --provider deepseek                # DeepSeek Chat

        harvester chat --search --functions                       # Grok with search + functions
        harvester chat --provider gemini --search --functions     # Gemini with search + functions
        harvester chat --provider openai --search --functions     # GPT with search + functions
        harvester chat --provider claude --search --functions     # Claude with search + functions
        harvester chat --provider claude --fetch                  # Claude with web fetch
        harvester chat --provider claude --memory                 # Claude with memory
        harvester chat --provider claude --code                   # Claude with code execution
        harvester chat --provider claude --search --fetch --functions --memory --code  # Claude with ALL tools!
        harvester chat --provider deepseek --functions            # DeepSeek with function calling
    """

    # Route to specific chat implementation
    if provider == 'grok':
        click.echo("💬 Starting Grok chat...")
        cmd = [sys.executable, 'cli/chat/grok_chat.py']
        if model:
            cmd.extend(['--model', model])
        if search:
            cmd.append('--search')
        if functions:
            cmd.append('--functions')

    elif provider == 'gemini':
        click.echo("💬 Starting Gemini chat...")
        cmd = [sys.executable, 'cli/chat/gemini_chat.py']
        if model:
            cmd.extend(['--model', model])
        if search:
            cmd.append('--search')
        if functions:
            cmd.append('--functions')

    elif provider in ['openai', 'gpt']:
        click.echo("💬 Starting OpenAI (GPT) chat...")
        cmd = [sys.executable, 'cli/chat/openai_chat.py']
        if model:
            cmd.extend(['--model', model])
        if search:
            cmd.append('--search')
        if functions:
            cmd.append('--functions')

    elif provider in ['claude', 'anthropic']:
        click.echo("💬 Starting Claude chat...")
        cmd = [sys.executable, 'cli/chat/claude_chat.py']
        if model:
            cmd.extend(['--model', model])
        if search:
            cmd.append('--search')
        if fetch:
            cmd.append('--fetch')
        if functions:
            cmd.append('--functions')
        if memory:
            cmd.append('--memory')
        if code:
            cmd.append('--code')

    elif provider == 'deepseek':
        click.echo("💬 Starting DeepSeek chat...")
        cmd = [sys.executable, 'cli/chat/deepseek_chat.py']
        if model:
            # Map model names for DeepSeek
            if model in ['chat', 'deepseek-chat']:
                cmd.extend(['--model', 'chat'])
            elif model in ['reasoner', 'deepseek-reasoner']:
                cmd.extend(['--model', 'reasoner'])
        if functions:
            cmd.append('--functions')

    else:
        click.echo(f"❌ Unknown provider: {provider}")
        click.echo("Available providers: grok, gemini, openai, claude, deepseek")
        return

    subprocess.run(cmd)

@cli.command('message')
@click.option('--model', '-m', default='gemini-2.5-flash', help='Model to use for conversation')
@click.option('--system', '-s', help='System prompt/context')
@click.option('--temperature', '-t', default=0.7, help='Response creativity (0.0-2.0)')
@click.option('--save', is_flag=True, help='Save conversation history to file')
@click.option('--max-tokens', default=4000, help='Maximum response length')
def message_command(model, system, temperature, save, max_tokens):
    """Start turn-based conversation with AI (non-streaming)"""
    import json
    from datetime import datetime
    
    click.echo("💬 Harvester SDK - Turn-Based Conversation")
    click.echo("© 2025 QUANTUM ENCODING LTD | info@quantumencoding.io")
    click.echo(f"🤖 Model: {model}")
    click.echo(f"🌡️  Temperature: {temperature}")
    if system:
        click.echo(f"⚙️  System: {system}")
    click.echo("Type 'exit', 'quit', or press Ctrl+C to end conversation")
    click.echo("=" * 60)
    
    # Initialize components
    from providers.provider_factory import ProviderFactory
    
    provider_factory = ProviderFactory()
    conversation_history = []
    
    # Add system message if provided
    if system:
        conversation_history.append({"role": "system", "content": system})
    
    try:
        provider = provider_factory.get_provider(model)
        click.echo(f"✅ Connected to {model}")
    except Exception as e:
        click.echo(f"❌ Error connecting to {model}: {e}")
        return
    
    conversation_count = 0
    
    # Enable readline for better input handling (paste support)
    try:
        import readline
        # Set up readline for better paste handling
        readline.set_startup_hook(None)
    except ImportError:
        pass  # readline not available on Windows
    
    try:
        while True:
            # Get user input with proper editing support
            try:
                click.echo("\n👤 You: ", nl=False)
                sys.stdout.flush()
                
                # Enhanced input with better editing support
                def get_input_with_editing():
                    import select
                    
                    # Read first line with full readline editing support
                    first_line = input()
                    if not first_line.strip():
                        return ""
                    
                    lines = [first_line]
                    
                    # Check for pasted multi-line content
                    try:
                        while hasattr(select, 'select') and select.select([sys.stdin], [], [], 0)[0]:
                            additional_line = input()
                            lines.append(additional_line)
                    except (EOFError, OSError, AttributeError):
                        pass
                    
                    # If single line, return as-is (full editing was available)
                    if len(lines) == 1:
                        return first_line
                    
                    # Check if any line is an exit command
                    for line in lines:
                        if line.strip().lower() in ['exit', 'quit']:
                            return line.strip()
                    
                    # Multi-line detected - offer editing options
                    combined = '\n'.join(lines)
                    click.echo(f"\n📋 Multi-line input detected ({len(lines)} lines)")
                    
                    # Show preview
                    preview_lines = lines[:2]
                    for i, line in enumerate(preview_lines, 1):
                        display_line = line[:50] + "..." if len(line) > 50 else line
                        click.echo(f"   {i}: {display_line}")
                    if len(lines) > 2:
                        click.echo(f"   ... and {len(lines) - 2} more lines")
                    
                    # Simple choice: use as-is or re-enter
                    click.echo("\nOptions:")
                    click.echo("  [Enter] - Use this text")
                    click.echo("  [e] - Re-enter text with full editing")
                    choice = input("Choice: ").strip().lower()
                    
                    if choice == 'e':
                        # Let them re-enter with a pre-filled readline buffer
                        click.echo("\n✏️  Enter your text (with full backspace/editing support):")
                        # Pre-fill the readline buffer with the combined text (spaces instead of newlines)
                        combined_single_line = combined.replace('\n', ' ')
                        return input(f"Re-edit: ")
                    else:
                        return combined
                
                user_input = get_input_with_editing().strip()
                
                # Handle empty input
                if not user_input:
                    continue
                    
            except (EOFError, KeyboardInterrupt):
                break
            
            if user_input.lower() in ['exit', 'quit']:
                break

            # Handle chat commands
            if user_input.startswith('/'):
                command_parts = user_input[1:].strip().split(maxsplit=1)
                command = command_parts[0].lower()
                args = command_parts[1] if len(command_parts) > 1 else ""

                if command == 'model':
                    if args:
                        # Switch model
                        new_model = args.strip()
                        try:
                            provider = provider_factory.get_provider(new_model)
                            model = new_model
                            click.echo(f"✅ Switched to {model}")
                            continue
                        except Exception as e:
                            click.echo(f"❌ Error switching to {new_model}: {e}")
                            continue
                    else:
                        # Show available models
                        click.echo(f"\n🤖 Current model: {model}")
                        click.echo("\n📋 Available models:")
                        click.echo("\nOpenAI:")
                        click.echo("  • gpt-5, gpt-5-mini, gpt-5-nano")
                        click.echo("\nAnthropic:")
                        click.echo("  • opus-4-1, sonnet-4-5, sonnet-4, haiku-3-5")
                        click.echo("\nGoogle GenAI:")
                        click.echo("  • gemini-pro-2-5, gemini-flash-2-5, gemini-flash-lite-2-5")
                        click.echo("\nGoogle Vertex:")
                        click.echo("  • vtx-gemini-pro, vtx-gemini-flash")
                        click.echo("  • vtx-opus, vtx-sonnet, vtx-haiku")
                        click.echo("\nxAI:")
                        click.echo("  • grok-4, grok-4-fast-r, grok-4-fast, grok-code")
                        click.echo("  • grok-3, grok-3-mini")
                        click.echo("\nDeepSeek:")
                        click.echo("  • deepseek-chat, deepseek-reasoner")
                        click.echo("\nUsage: /model <model-name>")
                        continue

                elif command == 'help':
                    click.echo("\n📋 Available Commands:")
                    click.echo("  /model [name]  - Show or switch model")
                    click.echo("  /temp <0-2>    - Set temperature")
                    click.echo("  /system <msg>  - Set system prompt")
                    click.echo("  /clear         - Clear conversation history")
                    click.echo("  /save          - Save conversation now")
                    click.echo("  /help          - Show this help")
                    click.echo("  exit, quit     - End conversation")
                    continue

                elif command == 'temp':
                    try:
                        new_temp = float(args)
                        if 0 <= new_temp <= 2:
                            temperature = new_temp
                            click.echo(f"✅ Temperature set to {temperature}")
                        else:
                            click.echo("❌ Temperature must be between 0 and 2")
                    except ValueError:
                        click.echo("❌ Invalid temperature value")
                    continue

                elif command == 'system':
                    if args:
                        # Update system prompt
                        if conversation_history and conversation_history[0]["role"] == "system":
                            conversation_history[0]["content"] = args
                        else:
                            conversation_history.insert(0, {"role": "system", "content": args})
                        system = args
                        click.echo(f"✅ System prompt updated")
                    else:
                        click.echo("❌ Usage: /system <message>")
                    continue

                elif command == 'clear':
                    conversation_history = []
                    if system:
                        conversation_history.append({"role": "system", "content": system})
                    conversation_count = 0
                    click.echo("✅ Conversation history cleared")
                    continue

                elif command == 'save':
                    if conversation_count > 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        conv_file = f"conversation_{model.replace('/', '_').replace('-', '_')}_{timestamp}.json"
                        with open(conv_file, 'w') as f:
                            json.dump({
                                "model": model,
                                "timestamp": timestamp,
                                "temperature": temperature,
                                "system": system,
                                "conversation": conversation_history
                            }, f, indent=2)
                        click.echo(f"✅ Conversation saved to {conv_file}")
                    else:
                        click.echo("❌ No conversation to save")
                    continue

                else:
                    click.echo(f"❌ Unknown command: /{command}")
                    click.echo("Type /help for available commands")
                    continue

            conversation_count += 1
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Show thinking indicator
            click.echo("🤔 Thinking...")
            
            try:
                # Use direct provider approach with model parameter
                response = asyncio.run(provider.complete(user_input, model))
                
                if response:
                    # Response is typically a string from complete method
                    assistant_message = str(response)
                    
                    # Display response with nice formatting
                    click.echo(f"\n🤖 {model}:")
                    click.echo("-" * 50)
                    click.echo(assistant_message)
                    click.echo("-" * 50)
                    
                    # Add assistant response to history
                    conversation_history.append({"role": "assistant", "content": assistant_message})
                    
                    # Save conversation periodically if requested
                    if save and conversation_count % 5 == 0:  # Save every 5 exchanges
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        conv_file = f"conversation_{model.replace('/', '_').replace('-', '_')}_{timestamp}.json"
                        with open(conv_file, 'w') as f:
                            json.dump({
                                "model": model,
                                "timestamp": timestamp,
                                "temperature": temperature,
                                "system": system,
                                "conversation": conversation_history
                            }, f, indent=2)
                        click.echo(f"💾 Conversation saved to {conv_file}")
                
                else:
                    click.echo("❌ No response received")
                    
            except Exception as e:
                click.echo(f"❌ Error getting response: {e}")
                continue
    
    except KeyboardInterrupt:
        pass

    # Cleanup provider sessions
    async def cleanup_sessions():
        """Close all provider sessions to prevent warnings"""
        try:
            # Close the current provider
            if hasattr(provider, 'close'):
                await provider.close()
            elif hasattr(provider, 'session') and provider.session:
                await provider.session.close()
            elif hasattr(provider, '_session') and provider._session:
                await provider._session.close()

            # Close all provider factory instances
            if hasattr(provider_factory, 'provider_instances'):
                for instance in provider_factory.provider_instances.values():
                    try:
                        if hasattr(instance, 'close'):
                            await instance.close()
                        elif hasattr(instance, 'session') and instance.session:
                            await instance.session.close()
                        elif hasattr(instance, '_session') and instance._session:
                            await instance._session.close()
                    except Exception:
                        pass
        except Exception:
            pass

    # Run cleanup
    try:
        asyncio.run(cleanup_sessions())
    except Exception:
        pass

    # Final save if requested
    if save and conversation_count > 0:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        conv_file = f"conversation_{model.replace('/', '_').replace('-', '_')}_{timestamp}_final.json"
        with open(conv_file, 'w') as f:
            json.dump({
                "model": model,
                "timestamp": timestamp,
                "temperature": temperature,
                "system": system,
                "total_exchanges": conversation_count,
                "conversation": conversation_history
            }, f, indent=2)
        click.echo(f"\n💾 Final conversation saved to {conv_file}")

    click.echo(f"\n👋 Conversation ended. Total exchanges: {conversation_count}")
    click.echo("Thank you for using Harvester SDK!")

@cli.command('computer')
@click.argument('task')
@click.option('--environment', '-e', type=click.Choice(['browser', 'docker']), default='browser',
              help='Environment to use (browser or docker)')
@click.option('--url', '-u', help='Initial URL to navigate to (browser only)')
@click.option('--headed', is_flag=True, help='Run browser in headed mode (visible window)')
@click.option('--cdp', help='Connect to Chrome DevTools Protocol URL (e.g., http://localhost:9222)')
@click.option('--width', type=int, default=1024, help='Display width')
@click.option('--height', type=int, default=768, help='Display height')
def computer_command(task, environment, url, headed, cdp, width, height):
    """GPT Computer Use - AI agent that controls browser/computer

    Examples:
        harvester computer "Search for OpenAI news on bing.com"
        harvester computer --url https://bing.com "Search for latest AI news"
        harvester computer --environment docker "Open Firefox and browse web"
        harvester computer "Book a flight from NYC to SF on kayak.com"
    """
    click.echo("🤖 Starting GPT Computer Use Agent...")
    cmd = [
        sys.executable,
        'cli/chat/gpt_computer_use.py',
        task,
        '--environment', environment,
        '--width', str(width),
        '--height', str(height)
    ]

    if url:
        cmd.extend(['--url', url])

    if headed:
        cmd.append('--headed')

    if cdp:
        cmd.extend(['--cdp', cdp])

    subprocess.run(cmd)

@cli.command('search')
@click.argument('query')
@click.option('--provider', '-p', default='grok', help='Search provider (currently only grok)')
@click.option('--model', '-m', default='grok-4', help='Model to use')
@click.option('--sources', multiple=True, default=['web', 'x', 'news'], help='Search sources')
@click.option('--country', help='Country code for results')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--save', '-s', help='Save results to file')
def search_command(query, provider, model, sources, country, format, save):
    """Search the web with AI-enhanced results"""
    if provider != 'grok':
        click.echo(f"⚠️  Search is currently only available with Grok provider")
        return

    click.echo(f"🔍 Searching: {query}")
    cmd = [
        sys.executable,
        'cli/chat/grok_search.py',
        query,
        '--model', model,
        '--format', format
    ]

    if sources:
        cmd.extend(['--sources'] + list(sources))
    if country:
        cmd.extend(['--country', country])
    if save:
        cmd.extend(['--save', save])

    subprocess.run(cmd)

@cli.command('structured')
@click.argument('prompt')
@click.option('--schema', '-s', help='Schema type: person, review, meeting, code, analysis', default='analysis')
@click.option('--model', '-m', default='gemini-2.5-flash', help='Model to use')
@click.option('--output', '-o', help='Output file to save structured JSON')
def structured_command(prompt, schema, model, output):
    """Generate structured output with schema validation"""
    click.echo(f"🎯 Structured Output Generation")
    click.echo(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    click.echo(f"📋 Schema: {schema}")
    click.echo(f"🤖 Model: {model}")
    click.echo()

    # Import SDK and Pydantic
    from harvester_sdk.sdk import HarvesterSDK
    import asyncio
    import json
    from typing import List, Optional
    from pydantic import BaseModel, Field

    # Define Pydantic schemas for provider-aware structured output
    class PersonSchema(BaseModel):
        name: str
        age: Optional[int] = None
        email: Optional[str] = None

    class ReviewSchema(BaseModel):
        rating: int = Field(..., ge=1, le=5)
        summary: str
        pros: Optional[List[str]] = []
        cons: Optional[List[str]] = []

    class MeetingSchema(BaseModel):
        title: str
        date: Optional[str] = None
        attendees: Optional[List[str]] = []
        action_items: Optional[List[str]] = []

    class CodeSchema(BaseModel):
        language: str
        description: str
        key_features: Optional[List[str]] = []
        use_cases: Optional[List[str]] = []

    class AnalysisSchema(BaseModel):
        summary: str
        key_points: List[str]
        recommendations: Optional[List[str]] = []

    # Map schema names to Pydantic classes
    pydantic_schemas = {
        'person': PersonSchema,
        'review': ReviewSchema,
        'meeting': MeetingSchema,
        'code': CodeSchema,
        'analysis': AnalysisSchema
    }

    async def generate():
        try:
            # Get Pydantic schema class
            schema_class = pydantic_schemas.get(schema, AnalysisSchema)

            # Use SDK's provider-aware structured output
            sdk = HarvesterSDK()
            click.echo("🔄 Generating provider-aware structured output...")
            click.echo(f"📦 Provider: {sdk.provider_factory.get_provider(model).__class__.__name__}")

            # Use the SDK's generate_structured method (provider-aware!)
            result = await sdk.generate_structured(
                prompt=prompt,
                schema_class=schema_class,
                model=model,
                max_tokens=2000
            )

            click.echo(f"✅ Success! (validated in {result.validation_attempts} attempt(s))")
            click.echo()
            click.echo("📄 Structured Output:")

            # Convert Pydantic model to dict for display
            result_dict = result.parsed_data.model_dump()
            click.echo(json.dumps(result_dict, indent=2))

            if output:
                with open(output, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                click.echo(f"\n💾 Saved to: {output}")

        except Exception as e:
            click.echo(f"❌ Error: {e}")

    asyncio.run(generate())

@cli.command('functions')
@click.argument('function_name', required=False)
@click.option('--list', 'list_functions', is_flag=True, help='List available functions')
@click.option('--args', help='Function arguments as JSON string')
@click.option('--file', help='Read function arguments from JSON file')
def functions_command(function_name, list_functions, args, file):
    """Execute functions and tools"""
    click.echo("🔧 FUNCTION CALLING & TOOL USE")
    click.echo("=" * 60)
    
    
    # Import SDK for function calling
    import asyncio
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harvester_sdk.sdk import HarvesterSDK
    
    async def run_function_command():
        sdk = HarvesterSDK()
        
        if list_functions:
            # List available functions
            try:
                functions = sdk.list_available_functions()
            except Exception as e:
                click.echo(f"❌ Error listing functions: {e}")
                return
            
            if not functions:
                click.echo("ℹ️  No functions available. Install dependencies:")
                click.echo("     pip install requests aiohttp")
                return

            click.echo(f"📋 Available Functions (Open Source - All Enabled):")
            click.echo()

            for name, info in functions.items():
                click.echo(f"🔧 {name}")
                click.echo(f"   📝 {info['description']}")
                click.echo(f"   📂 Category: {info['category']}")
                click.echo(f"   🔒 Security: {info['security_level']}")

                if info.get('parameters_schema'):
                    params = info['parameters_schema'].get('properties', {})
                    required_params = info['parameters_schema'].get('required', [])
                    if params:
                        click.echo("   📋 Parameters:")
                        for param_name, param_info in params.items():
                            req_marker = " (required)" if param_name in required_params else ""
                            param_type = param_info.get('type', 'any')
                            click.echo(f"     • {param_name}: {param_type}{req_marker}")
                            if param_info.get('description'):
                                click.echo(f"       └─ {param_info['description']}")
                click.echo()
            
            return
        
        if not function_name:
            click.echo("❌ Please specify a function name or use --list to see available functions")
            click.echo("Example: harvester functions read_file --args '{\"file_path\": \"/path/to/file.txt\"}'")
            return
        
        # Parse arguments
        arguments = {}
        if args:
            try:
                arguments = json.loads(args)
            except json.JSONDecodeError as e:
                click.echo(f"❌ Invalid JSON in --args: {e}")
                return
        elif file:
            try:
                with open(file, 'r') as f:
                    arguments = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                click.echo(f"❌ Error reading arguments file: {e}")
                return
        
        # Execute function
        click.echo(f"🔧 Executing: {function_name}")
        click.echo(f"📋 Arguments: {arguments}")
        click.echo()
        
        try:
            result = await sdk.call_function(function_name, arguments)
            
            if result.success:
                click.echo("✅ Function executed successfully!")
                click.echo(f"📤 Result: {result.result}")
                
                if result.metadata:
                    click.echo(f"📊 Metadata: {result.metadata}")
            else:
                click.echo(f"❌ Function failed: {result.error}")
                
        except Exception as e:
            click.echo(f"❌ Error executing function: {e}")
    
    # Run the async function
    asyncio.run(run_function_command())

@cli.command('status')
@click.option('--job-id', help='Check specific job status')
@click.option('--all', is_flag=True, help='Show all jobs')
def status_command(job_id, all):
    """Check batch job status"""
    click.echo("📊 Checking batch status...")
    cmd = [sys.executable, 'cli/batch/batch_status.py']
    
    if job_id:
        cmd.extend(['--job', job_id])
    elif all:
        cmd.append('--all')
    
    subprocess.run(cmd)

@cli.command('json')
@click.argument('json_file', type=click.Path(exists=True))
@click.option('--model', '-m', default='vtx-1', help='Model to use')
@click.option('--template', '-t', default='advice', help='Template to apply')
@click.option('--output', '-o', help='Output file')
def json_command(json_file, model, template, output):
    """Process single JSON request with AI"""
    click.echo(f"📄 Processing JSON request: {json_file}")
    cmd = [
        sys.executable,
        'cli/processing/json_processor.py',
        json_file,
        '--model', model,
        '--template', template
    ]
    
    if output:
        cmd.extend(['--output', output])
    
    subprocess.run(cmd)

@cli.command('convert')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output CSV file')
@click.option('--format', '-f', help='Input format (auto-detected if not specified)')
def convert_command(input_file, output, format):
    """Convert any file format to CSV for batch processing"""
    click.echo(f"🔄 Converting {input_file} to CSV...")
    cmd = [
        sys.executable,
        'cli/processing/csv_processor.py',
        'convert',
        input_file
    ]
    
    if output:
        cmd.extend(['--output', output])
    if format:
        cmd.extend(['--format', format])
    
    subprocess.run(cmd)

@cli.command('list-models')
@click.option('--provider', '-p', help='Filter by provider')
@click.option('--groups', is_flag=True, help='Show model groups')
def list_models_command(provider, groups):
    """List available models and providers"""
    click.echo("🤖 Available Models and Providers\n")
    
    # Import provider factory to list models
    from providers.provider_factory import ProviderFactory
    factory = ProviderFactory()
    
    if groups:
        click.echo("Model Groups:")
        click.echo("  grp-fast     : Fast, efficient models")
        click.echo("  grp-quality  : High quality, slower models")
        click.echo("  grp-code     : Optimized for code generation")
        click.echo("  all          : All available models\n")
    
    click.echo("Providers and Models:")
    for provider_name in factory.list_providers():
        if provider and provider not in provider_name:
            continue
        click.echo(f"\n  {provider_name}:")
        models = factory.list_models()
        provider_models = [m for m in models if provider_name in m.lower() or m.startswith(provider_name[:3])]
        for model in provider_models[:5]:  # Show first 5 models per provider
            click.echo(f"    - {model}")

@cli.command('config')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--set-key', nargs=2, help='Set API key (provider, key)')
@click.option('--test', help='Test provider configuration')
def config_command(show, set_key, test):
    """Manage SDK configuration and API keys"""
    if show:
        click.echo("📋 Current Configuration:\n")
        env_vars = {
            'ANTHROPIC_API_KEY': '✓' if os.getenv('ANTHROPIC_API_KEY') else '✗',
            'OPENAI_API_KEY': '✓' if os.getenv('OPENAI_API_KEY') else '✗',
            'GEMINI_API_KEY': '✓' if os.getenv('GEMINI_API_KEY') else '✗',
            'XAI_API_KEY': '✓' if os.getenv('XAI_API_KEY') else '✗',
            'DEEPSEEK_API_KEY': '✓' if os.getenv('DEEPSEEK_API_KEY') else '✗',
        }
        for key, status in env_vars.items():
            click.echo(f"  {key}: {status}")
    
    elif set_key:
        provider, key = set_key
        env_var = f"{provider.upper()}_API_KEY"
        click.echo(f"Setting {env_var}...")
        # Note: This only sets for current session
        os.environ[env_var] = key
        click.echo(f"✓ {env_var} set for this session")
        click.echo("To persist, add to your .env file or shell profile")
    
    elif test:
        click.echo(f"Testing {test} provider...")
        # Could implement provider test here
        click.echo("Provider test functionality coming soon!")

@cli.command('agent-grok')
@click.argument('task', required=True)
@click.option('--files', '-f', multiple=True, help='Files to include as context')
@click.option('--project-structure', '-p', help='Project structure description or file')
@click.option('--task-type', '-t', default='general',
              type=click.Choice(['general', 'debugging', 'refactoring', 'feature']),
              help='Type of coding task')
@click.option('--max-iterations', '-i', default=100, help='Maximum iterations for agentic workflow')
@click.option('--show-reasoning', is_flag=True, help='Display reasoning traces')
@click.option('--output', '-o', help='Save result to file')
def grok_code_command(task, files, project_structure, task_type, max_iterations, show_reasoning, output):
    """
    🤖 Grok Code Agent - Agentic coding assistant

    Powered by grok-code-fast-1 with:
    - Streaming reasoning traces
    - Native tool calling (file ops, search, execution)
    - Context-aware prompting with XML/Markdown
    - Cache optimization for fast iterations

    Examples:
        # Simple task
        harvester grok-code "Add error handling to sql.ts"

        # With context files
        harvester grok-code "Refactor authentication" -f auth.ts -f db.ts

        # Debugging with project structure
        harvester grok-code "Fix memory leak" -p structure.txt -t debugging

        # Feature development with reasoning display
        harvester grok-code "Add rate limiting" -t feature --show-reasoning
    """
    from harvester_agents.grok_code_agent import GrokCodeAgent

    click.echo("🤖 Grok Code Agent")
    click.echo(f"📋 Task: {task}")
    click.echo(f"🎯 Type: {task_type}")
    click.echo()

    # Build context
    context = {}

    # Add files to context
    if files:
        context['files'] = {}
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    context['files'][file_path] = f.read()
                click.echo(f"📄 Loaded: {file_path}")
            except Exception as e:
                click.echo(f"⚠️  Could not load {file_path}: {e}")

    # Add project structure
    if project_structure:
        try:
            if os.path.isfile(project_structure):
                with open(project_structure, 'r') as f:
                    context['project_structure'] = f.read()
            else:
                context['project_structure'] = project_structure
        except Exception as e:
            click.echo(f"⚠️  Could not load project structure: {e}")

    async def run_agent():
        agent = GrokCodeAgent(max_iterations=max_iterations)

        click.echo("🚀 Starting agentic workflow...")
        click.echo()

        # Track tool calls in real-time
        def on_tool_call(tool_name: str, args: dict):
            click.echo(f"🔧 {tool_name}({', '.join(f'{k}={v}' for k, v in args.items())})")

        # Execute task
        result = await agent.execute_task(
            description=task,
            context=context if context else None,
            task_type=task_type
        )

        # Display results
        click.echo(f"📊 Status: {result.status}")
        click.echo(f"🔄 Iterations: {result.iterations}")
        click.echo()

        # Show reasoning traces
        if show_reasoning and result.reasoning_traces:
            click.echo("💭 Reasoning Traces:")
            for i, trace in enumerate(result.reasoning_traces, 1):
                click.echo(f"\n--- Step {i} ---")
                click.echo(trace.content[:500] + "..." if len(trace.content) > 500 else trace.content)
            click.echo()

        # Show tool calls
        if result.tool_calls:
            click.echo(f"🔧 Tool Calls: {len(result.tool_calls)}")
            for tool_call in result.tool_calls:
                click.echo(f"  - {tool_call.name}({', '.join(f'{k}={v}' for k, v in tool_call.arguments.items())})")
            click.echo()

        # Show result
        click.echo("✅ Result:")
        click.echo(result.result)

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                output_data = {
                    'task': task,
                    'task_type': task_type,
                    'status': result.status,
                    'iterations': result.iterations,
                    'result': result.result,
                    'reasoning_traces': [
                        {'step': i, 'content': t.content}
                        for i, t in enumerate(result.reasoning_traces, 1)
                    ] if show_reasoning else [],
                    'tool_calls': [
                        {
                            'name': tc.name,
                            'arguments': tc.arguments,
                            'result': tc.result
                        }
                        for tc in result.tool_calls
                    ]
                }
                json.dump(output_data, f, indent=2)
            click.echo(f"\n💾 Saved to: {output}")

    # Run async agent
    asyncio.run(run_agent())

@cli.command('agent-claude')
@click.argument('task', required=True)
@click.option('--files', '-f', multiple=True, help='Files to include as context')
@click.option('--project-structure', '-p', help='Project structure description or file')
@click.option('--task-type', '-t', default='general',
              type=click.Choice(['general', 'debugging', 'refactoring', 'feature']),
              help='Type of coding task')
@click.option('--max-iterations', '-i', default=100, help='Maximum iterations')
@click.option('--model', '-m', default='claude-sonnet-4-5', help='Claude model to use')
@click.option('--output', '-o', help='Save result to file')
def claude_code_command(task, files, project_structure, task_type, max_iterations, model, output):
    """
    🤖 Claude Code Agent - Professional agentic assistant

    Built on Anthropic's official Claude Agent SDK with:
    - Production-tested agent loop
    - Built-in context compaction and cache optimization
    - Proper verification and error handling
    - Subagents for parallel work
    - MCP integration for external services
    - Custom harvester tools (query other providers, JSON tools)

    Examples:
        # Simple task
        harvester claude-code "Add error handling to auth.py"

        # With context files
        harvester claude-code "Refactor database layer" -f db.py -f models.py

        # Feature development
        harvester claude-code "Implement rate limiting API" -t feature

        # Different Claude model
        harvester claude-code "Debug memory leak" -m claude-opus-4 -t debugging
    """
    from harvester_agents.claude_code_agent import ClaudeCodeAgent

    click.echo("🤖 Claude Code Agent")
    click.echo(f"📋 Task: {task}")
    click.echo(f"🎯 Type: {task_type}")
    click.echo(f"🧠 Model: {model}")
    click.echo()

    # Build context
    context = {}

    # Add files to context
    if files:
        context['files'] = {}
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    context['files'][file_path] = f.read()
                click.echo(f"📄 Loaded: {file_path}")
            except Exception as e:
                click.echo(f"⚠️  Could not load {file_path}: {e}")

    # Add project structure
    if project_structure:
        try:
            if os.path.isfile(project_structure):
                with open(project_structure, 'r') as f:
                    context['project_structure'] = f.read()
            else:
                context['project_structure'] = project_structure
        except Exception as e:
            click.echo(f"⚠️  Could not load project structure: {e}")

    async def run_agent():
        agent = ClaudeCodeAgent(
            model=model,
            max_iterations=max_iterations
        )

        click.echo("🚀 Starting Claude Agent SDK workflow...")
        click.echo()

        # Execute task (SDK handles everything)
        result = await agent.execute_task(
            description=task,
            context=context if context else None,
            task_type=task_type,
            show_progress=True
        )

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                output_data = {
                    'task': task,
                    'task_type': task_type,
                    'model': model,
                    'status': result.status,
                    'iterations': result.iterations,
                    'result': result.result,
                    'messages': result.messages
                }
                json.dump(output_data, f, indent=2)
            click.echo(f"\n💾 Saved to: {output}")

    # Run async agent
    asyncio.run(run_agent())

@cli.command('agent-openai')
@click.argument('task', required=True)
@click.option('--model', '-m', default='gpt-4o', help='OpenAI model to use (gpt-4o, gpt-4o-mini, o1)')
@click.option('--temperature', '-t', type=float, help='Sampling temperature (0.0-2.0)')
@click.option('--output', '-o', help='Save result to file')
def openai_agent_command(task, model, temperature, output):
    """
    🤖 OpenAI Code Agent - GPT-powered agentic coding assistant

    Built on OpenAI's Agents SDK with:
    - File reading, writing, and editing capabilities
    - Shell command execution
    - Advanced reasoning models (o1, o3-mini)
    - Multi-step planning and execution

    Examples:
        # Create a Python script
        harvester agent-openai "Create a hello world Python script"

        # Advanced reasoning with o1
        harvester agent-openai "Refactor auth.py for better error handling" -m o1

        # Complex task
        harvester agent-openai "Add logging to all functions in utils.py"

        # Save result to file
        harvester agent-openai "Analyze this codebase" -o analysis.txt
    """
    from harvester_agents.openai_code_agent import OpenAICodeAgent

    click.echo("🤖 OpenAI Code Agent")
    click.echo(f"📋 Task: {task}")
    click.echo(f"🧠 Model: {model}")
    if temperature is not None:
        click.echo(f"🌡️  Temperature: {temperature}")
    click.echo()

    try:
        # Create code agent
        agent = OpenAICodeAgent(
            model=model,
            temperature=temperature,
        )

        # Run agent (sync wrapper for async execution)
        result = agent.run_sync(task, show_progress=True)

        # Display result
        click.echo()
        click.echo("📤 Result:")
        click.echo("-" * 60)
        click.echo(result['result'])
        click.echo("-" * 60)

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                output_data = {
                    'task': task,
                    'model': model,
                    'temperature': temperature,
                    'result': result
                }
                json.dump(output_data, f, indent=2)
            click.echo(f"\n💾 Saved to: {output}")

    except ImportError as e:
        click.echo(f"❌ Error: OpenAI Agents SDK not installed")
        click.echo(f"   Install with: pip install 'harvester-sdk[computer]'")
        click.echo(f"   Details: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command('agent-gpt5')
@click.argument('task', required=True)
@click.option('--model', '-m', default='gpt-5', help='GPT-5 model (gpt-5, gpt-5-mini, gpt-5-nano)')
@click.option('--reasoning', '-r', default='medium',
              type=click.Choice(['minimal', 'low', 'medium', 'high']),
              help='Reasoning effort (minimal=fastest, high=most thorough)')
@click.option('--verbosity', '-v', default='medium',
              type=click.Choice(['low', 'medium', 'high']),
              help='Output verbosity (low=concise, high=detailed)')
@click.option('--output', '-o', help='Save result to file')
def gpt5_agent_command(task, model, reasoning, verbosity, output):
    """
    🧠 GPT-5 Code Agent - Advanced reasoning for coding and agentic tasks

    Built on GPT-5 with:
    - Configurable reasoning effort (minimal to high)
    - Verbosity control for output length
    - Custom tools with freeform text inputs
    - File operations and shell execution
    - Preambles for transparent tool-calling

    Reasoning effort guide:
        minimal - Fastest time-to-first-token, best for simple tasks
        low     - Quick reasoning, good for straightforward coding
        medium  - Balanced reasoning (default), good for most tasks
        high    - Thorough reasoning, best for complex multi-step tasks

    Verbosity guide:
        low    - Concise responses, minimal commentary
        medium - Balanced explanations (default)
        high   - Detailed explanations and documentation

    Examples:
        # Quick task with minimal reasoning
        harvester agent-gpt5 "Create a hello world script" -r minimal -v low

        # Standard coding task
        harvester agent-gpt5 "Add error handling to auth.py"

        # Complex refactoring with high reasoning
        harvester agent-gpt5 "Refactor entire codebase for async/await" -r high -v high

        # Cost-optimized with gpt-5-mini
        harvester agent-gpt5 "Fix bugs in utils.py" -m gpt-5-mini

        # High-throughput classification with nano
        harvester agent-gpt5 "Classify all files by type" -m gpt-5-nano -r minimal
    """
    from harvester_agents.gpt5_code_agent import GPT5CodeAgent

    click.echo("🧠 GPT-5 Code Agent")
    click.echo(f"📋 Task: {task}")
    click.echo(f"🤖 Model: {model}")
    click.echo(f"💭 Reasoning: {reasoning}")
    click.echo(f"📝 Verbosity: {verbosity}")
    click.echo()

    try:
        # Create GPT-5 code agent
        agent = GPT5CodeAgent(
            model=model,
            reasoning_effort=reasoning,
            verbosity=verbosity,
        )

        # Run agent
        result = agent.execute_task(task, show_progress=True)

        # Display result
        click.echo()
        click.echo("📤 Result:")
        click.echo("-" * 60)
        click.echo(result['result'])
        click.echo("-" * 60)

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                output_data = {
                    'task': task,
                    'model': model,
                    'reasoning_effort': reasoning,
                    'verbosity': verbosity,
                    'result': result
                }
                json.dump(output_data, f, indent=2)
            click.echo(f"\n💾 Saved to: {output}")

    except ImportError as e:
        click.echo(f"❌ Error: OpenAI SDK not installed")
        click.echo(f"   Install with: pip install 'harvester-sdk[computer]'")
        click.echo(f"   Details: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command('code-interpreter')
@click.argument('task', required=True)
@click.option('--model', '-m', default='gpt-4.1', help='Model to use (gpt-4.1, gpt-5, o3, o4-mini)')
@click.option('--upload', '-u', multiple=True, help='Upload file(s) to container')
@click.option('--download-all', '-d', is_flag=True, help='Download all generated files')
@click.option('--output-dir', '-o', default='./output', help='Directory for downloaded files')
@click.option('--container-id', '-c', help='Use existing container ID')
def code_interpreter_command(task, model, upload, download_all, output_dir, container_id):
    """
    🐍 Code Interpreter Agent - Python code execution in sandboxed containers

    Allows models to write and run Python to solve complex problems:
    - Data analysis and visualization
    - Mathematical computations
    - File processing and transformation
    - Iterative problem solving

    Examples:
        # Solve a math problem
        harvester code-interpreter "Solve the equation 3x + 11 = 14"

        # Data analysis
        harvester code-interpreter "Analyze data.csv and create a histogram" -u data.csv

        # Image processing
        harvester code-interpreter "Resize image.png to 800x600" -u image.png -d

        # Generate visualization
        harvester code-interpreter "Create a sine wave plot from 0 to 2π" -d

        # Use specific model
        harvester code-interpreter "Calculate fibonacci(100)" -m gpt-5
    """
    from harvester_agents.code_interpreter_agent import CodeInterpreterAgent
    from pathlib import Path

    click.echo("🐍 Code Interpreter Agent")
    click.echo(f"📋 Task: {task}")
    click.echo(f"🤖 Model: {model}")
    if upload:
        click.echo(f"📁 Uploading {len(upload)} file(s)")
    click.echo()

    try:
        # Determine container mode
        if container_id:
            container_mode = "explicit"
        else:
            container_mode = "auto"

        # Create agent
        agent = CodeInterpreterAgent(
            model=model,
            container_mode=container_mode,
            container_id=container_id
        )

        # Upload files if specified
        uploaded_file_ids = []
        if upload:
            for filepath in upload:
                if not Path(filepath).exists():
                    click.echo(f"❌ File not found: {filepath}")
                    sys.exit(1)
                file_id = agent.upload_file(filepath)
                uploaded_file_ids.append(file_id)
                click.echo(f"✓ Uploaded: {filepath} → {file_id}")

        # Add uploaded files to agent
        if uploaded_file_ids:
            agent.file_ids.extend(uploaded_file_ids)

        # Run task
        result = agent.execute_task(task, show_progress=True)

        # Display result
        click.echo()
        click.echo("📤 Result:")
        click.echo("-" * 60)
        click.echo(result['result'])
        click.echo("-" * 60)

        # Download generated files if requested
        if download_all and result['generated_files']:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            click.echo()
            click.echo(f"📥 Downloading {len(result['generated_files'])} file(s) to {output_dir}/")

            for file_info in result['generated_files']:
                filename = file_info['filename']
                file_id = file_info['file_id']
                save_path = output_path / filename

                agent.download_file(file_id, str(save_path))
                click.echo(f"✓ Downloaded: {filename}")

        # Show container info
        if result['container_id']:
            click.echo()
            click.echo(f"🗂️  Container ID: {result['container_id']}")
            click.echo("   (Container expires after 20 minutes of inactivity)")

    except ImportError as e:
        click.echo(f"❌ Error: OpenAI SDK not installed")
        click.echo(f"   Install with: pip install 'harvester-sdk[computer]'")
        click.echo(f"   Details: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command('image-gen')
@click.argument('prompt', required=True)
@click.option('--model', '-m', default='gpt-5', help='Model to use (gpt-4o, gpt-4.1, gpt-5, o3)')
@click.option('--output', '-o', default='generated.png', help='Output file path')
@click.option('--size', '-s', default='auto', help='Image size (e.g., 1024x1024, auto)')
@click.option('--quality', '-q', default='auto',
              type=click.Choice(['low', 'medium', 'high', 'auto']),
              help='Rendering quality')
@click.option('--format', '-f', default='png',
              type=click.Choice(['png', 'jpeg', 'webp']),
              help='Output format')
@click.option('--input-image', '-i', help='Input image for editing')
@click.option('--edit', '-e', is_flag=True, help='Edit the previously generated image')
def image_gen_command(prompt, model, output, size, quality, format, input_image, edit):
    """
    🎨 Image Generation Agent - AI-powered image creation and editing

    Generate and edit images using GPT Image model with automatic prompt optimization.

    Examples:
        # Generate an image
        harvester image-gen "A gray tabby cat hugging an otter with an orange scarf"

        # Specify output path and quality
        harvester image-gen "Sunset over mountains" -o sunset.png -q high

        # Edit an existing image
        harvester image-gen "Make it more colorful" -i input.jpg -o edited.png

        # Use specific model and size
        harvester image-gen "Abstract art" -m gpt-5 -s 1024x1536

        # Multi-turn editing (stores previous response)
        harvester image-gen "Draw a cat"
        harvester image-gen "Make it realistic" --edit
    """
    from harvester_agents.image_generation_agent import ImageGenerationAgent

    click.echo("🎨 Image Generation Agent")
    click.echo(f"📝 Prompt: {prompt}")
    click.echo(f"🤖 Model: {model}")
    if input_image:
        click.echo(f"🖼️  Input: {input_image}")
    click.echo()

    try:
        # Create agent
        agent = ImageGenerationAgent(
            model=model,
            size=size,
            quality=quality,
            format=format
        )

        # Generate or edit image
        if edit:
            result = agent.edit(
                edit_prompt=prompt,
                output_path=output,
                show_progress=True
            )
        elif input_image:
            result = agent.generate_from_file(
                prompt=prompt,
                input_image_path=input_image,
                output_path=output,
                show_progress=True
            )
        else:
            result = agent.generate(
                prompt=prompt,
                output_path=output,
                show_progress=True
            )

        # Display result
        if result['status'] == 'completed':
            click.echo()
            click.echo("📤 Result:")
            click.echo("-" * 60)
            if result.get('revised_prompt'):
                click.echo(f"✨ Revised prompt: {result['revised_prompt']}")
            if result.get('output_path'):
                click.echo(f"💾 Saved to: {result['output_path']}")
            click.echo(f"🆔 Response ID: {result['response_id']}")
            click.echo("-" * 60)
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except ImportError as e:
        click.echo(f"❌ Error: OpenAI SDK not installed")
        click.echo(f"   Install with: pip install 'harvester-sdk[computer]'")
        click.echo(f"   Details: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error generating image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    cli()