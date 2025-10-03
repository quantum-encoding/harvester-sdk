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
                        click.echo("  • gpt-5, gpt-5-nano, gpt-5-mini")
                        click.echo("\nAnthropic:")
                        click.echo("  • claude-opus-4-1-20250805")
                        click.echo("  • claude-sonnet-4-5-20250929")
                        click.echo("  • claude-sonnet-4-20250514")
                        click.echo("  • claude-3-5-haiku-20241022")
                        click.echo("\nGoogle:")
                        click.echo("  • gemini-2.5-pro, gemini-2.5-flash")
                        click.echo("  • gemini-2.5-flash-image, gemini-2.5-flash-lite")
                        click.echo("\nxAI:")
                        click.echo("  • grok-4-0709, grok-3, grok-3-mini")
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
@click.option('--width', type=int, default=1024, help='Display width')
@click.option('--height', type=int, default=768, help='Display height')
def computer_command(task, environment, url, width, height):
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

    # For now, show what would happen
    click.echo("\n🎯 Would generate structured output with schema validation")
    click.echo("📊 Would include automatic retry on validation failures")

    if output:
        click.echo(f"💾 Would save to: {output}")

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
                click.echo("ℹ️  Function calling is available at Professional tier and above")
                click.echo("📋 Basic functions available at Professional tier:")
                click.echo("  • read_file - Read contents of a file")
                click.echo("  • list_files - List files in a directory")
                click.echo("  • get_weather - Get current weather (mock)")
                click.echo()
                click.echo("🎯 Premium tier unlocks:")
                click.echo("  • write_file - Write content to files")
                click.echo("  • web_search - Search the web")
                click.echo("  • execute_code - Run code in sandbox")
                click.echo("  • analyze_image - Computer vision")
                click.echo("  • database_query - Query databases")
                click.echo()
                click.echo("🌟 Upgrade to Premium: https://quantumencoding.io/premium")
                return
            
            click.echo(f"📋 Available Functions ({arbiter.current_tier.upper()} tier):")
            click.echo()
            
            for name, info in functions.items():
                click.echo(f"🔧 {name}")
                click.echo(f"   📝 {info['description']}")
                click.echo(f"   📂 Category: {info['category']}")
                click.echo(f"   🔒 Security: {info['security_level']}")
                
                if info['parameters']:
                    click.echo("   📋 Parameters:")
                    for param_name, param_info in info['parameters'].items():
                        required = " (required)" if param_info.get('required', False) else ""
                        click.echo(f"     • {param_name}: {param_info.get('type', 'any')}{required}")
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

if __name__ == '__main__':
    cli()