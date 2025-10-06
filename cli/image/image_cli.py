#!/usr/bin/env python3
"""
Sacred Wrapper - Universal Image Generation CLI

A unified interface for all image generation providers with smart model routing,
legacy compatibility, and battle-tested reliability.
"""

import click
import json
import base64
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent SDK directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from providers.provider_factory import ProviderFactory
from utils.output_paths import generate_cli_output_directory

# Legacy model compatibility mapping
IMAGE_MODEL_ALIASES = {
    'o1-img': 'dalle-3',           # Old alias -> DALL-E 3
    'o2-img': 'dalle-3',           # Old alias -> DALL-E 3
    'g1-img': 'goo-1-img',         # Old alias -> Imagen 3 (goo-1-img)
    'g2-img': 'goo-4-img',         # Old alias -> Imagen 4 Ultra (goo-4-img)
    'g3-img': 'goo-5-img',         # Old alias -> Imagen 4 Fast (goo-5-img)
    'imagen-3': 'goo-1-img',       # Direct name -> alias
    'imagen-3-fast': 'goo-3-img',  # Direct name -> alias
    'imagen-4': 'goo-4-img',       # Direct name -> alias
    'imagen-4-ultra': 'goo-4-img', # Direct name -> alias
    'imagen-4-fast': 'goo-5-img',  # Direct name -> alias
}

@click.command()
@click.option('--prompt', '-p', required=True, help='Image generation prompt')
@click.option('--model', '-m', default='nano-banana', help='Model: dalle-3, nano-banana, imagen-4-ultra, imagen-4-fast, gpt-image, grok-image')
@click.option('--output', '-o', help='Custom output directory (uses sovereign structure by default)')
@click.option('--size', '-s', default='1024x1024', help='Image size (1024x1024, 1792x1024, 1024x1792)')
@click.option('--quality', '-q', default='hd', help='Image quality (standard, hd)')
@click.option('--style', default='vivid', help='Image style (vivid, natural)')
@click.option('--aspect-ratio', '-a', help='Aspect ratio for Imagen/GenAI models (16:9, 4:3, 9:16, 1:1, etc.)')
@click.option('--save-metadata', is_flag=True, help='Save generation metadata as JSON')
def main(prompt, model, output, size, quality, style, aspect_ratio, save_metadata):
    """
    Sacred Wrapper - Universal Image Generation CLI
    
    Generate images using various AI providers with unified interface.
    
    Examples:
        # Generate with Gemini Flash Image (default, fastest)
        image-cli -p "A serene mountain landscape" -m goo-2-img

        # Generate with DALL-E 3
        image-cli -p "A serene mountain landscape" -m dalle-3

        # High quality with Imagen 4 Ultra
        image-cli -p "Portrait of a wise sage" -m goo-4-img -a 9:16

        # Fast generation with Imagen 4 Fast
        image-cli -p "Futuristic cityscape" -m goo-5-img -a 16:9

        # Save with metadata
        image-cli -p "Abstract art" -m gpt-image-1 --save-metadata
    """
    
    # Handle legacy model aliases
    if model in IMAGE_MODEL_ALIASES:
        click.echo(f"🔄 Converting legacy alias '{model}' -> '{IMAGE_MODEL_ALIASES[model]}'")
        model = IMAGE_MODEL_ALIASES[model]
    
    # Generate output directory in ~/harvester-sdk/images/
    if not output:
        home = Path.home()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        folder_name = f"{safe_prompt.replace(' ', '_')}_{timestamp}"
        output = home / "harvester-sdk" / "images" / folder_name
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"🎨 Sacred Wrapper SDK - Image Generation")
    click.echo(f"🤖 Model: {model}")
    click.echo(f"📝 Prompt: {prompt}")
    click.echo(f"📐 Size: {size}")
    click.echo(f"💎 Quality: {quality}")
    click.echo(f"🎭 Style: {style}")
    if aspect_ratio:
        click.echo(f"📏 Aspect Ratio: {aspect_ratio}")
    click.echo(f"📂 Output: {output_path}")
    
    try:
        # Initialize provider - use SDK config directory
        config_dir = Path(__file__).parent.parent.parent / 'config'
        provider_factory = ProviderFactory(config_dir)
        provider = provider_factory.get_provider(model)
        
        # Prepare parameters based on provider
        params = {
            'prompt': prompt,
            'model': model
        }
        
        # Provider-specific parameters
        if 'dalle' in model or 'gpt-image' in model:
            # OpenAI DALL-E / GPT Image parameters
            params.update({
                'size': size,
                'quality': quality,
                'style': style
            })
        elif 'nano-banana' in model or 'gemini-2.5-flash-image' in model:
            # Google GenAI (Nano Banana - Gemini Flash Image) parameters
            if aspect_ratio:
                params['aspect_ratio'] = aspect_ratio
            params.update({
                'safety_filter_level': 'block_some',
                'person_generation': 'allow_adult'
            })
        elif 'imagen-' in model:
            # Google Vertex AI Imagen parameters (goo-4-img, goo-5-img)
            # Default to 1:1 if no aspect ratio specified
            params['aspect_ratio'] = aspect_ratio if aspect_ratio else '1:1'
            params.update({
                'safety_setting': 'block_few',  # Match Vertex SDK param name
                'person_generation': 'allow_all',  # Match your working script
                'add_watermark': True,
                'sample_count': 1,  # Number of images (1-4 supported)
                'sample_image_size': '1K',  # Image quality: 1K or 2K
                'enhance_prompt': True,  # Let Imagen enhance the prompt
                'language': 'en'  # Language code
            })
        elif 'grok' in model:
            # xAI Grok Image parameters (OpenAI-compatible format)
            params.update({
                'size': size,
                'quality': quality,
                'style': style
            })
        
        click.echo("🎨 Generating image...")

        # Generate image (handle both sync and async)
        result = provider.generate_image(**params)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        # Close any open aiohttp sessions
        try:
            if hasattr(provider, 'session') and provider.session:
                asyncio.run(provider.session.close())
            if hasattr(provider, '_session') and provider._session:
                asyncio.run(provider._session.close())
            # Also try the close method if available
            if hasattr(provider, 'close'):
                asyncio.run(provider.close())
        except:
            pass

        # Parse result
        if isinstance(result, str):
            result = json.loads(result)
        
        # Process and save image
        if result.get('images') and len(result['images']) > 0:
            image_data = result['images'][0]
            
            if 'b64_json' in image_data:
                # Decode and save image
                image_bytes = base64.b64decode(image_data['b64_json'])
                
                # Generate filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"image_{timestamp}.png"
                filepath = output_path / filename
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                click.echo(f"✅ Image saved: {filepath}")
                
                # Save metadata if requested
                if save_metadata:
                    metadata = {
                        'prompt': prompt,
                        'model': model,
                        'parameters': params,
                        'filename': filename,
                        'timestamp': datetime.now().isoformat(),
                        'safety_rating': image_data.get('safety_rating', 'unknown'),
                        'revised_prompt': image_data.get('revised_prompt', prompt)
                    }
                    
                    metadata_file = output_path / f"metadata_{timestamp}.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    click.echo(f"📄 Metadata saved: {metadata_file}")
                
                return 0
            else:
                click.echo("❌ No image data in response")
                return 1
        else:
            click.echo("❌ No images in response")
            return 1
            
    except Exception as e:
        click.echo(f"❌ Error generating image: {str(e)}")
        return 1

if __name__ == '__main__':
    main()