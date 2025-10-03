#!/usr/bin/env python3
"""
Automated Batch Vertex AI Image Processor
"""
import click
import asyncio
import json
import base64
import csv
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict, Any
import time

# Add parent SDK directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from providers.provider_factory import ProviderFactory


class BatchVertexProcessor:
    """Batch processor using style-vertex's proven approach"""
    
    def __init__(self, model_alias: str = 'goo-4-img', output_dir: str = 'batch_output',
                 requests_per_minute: int = 6, aspect_ratio: str = None):
        self.model_alias = model_alias
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.requests_per_minute = requests_per_minute
        self.aspect_ratio_override = aspect_ratio  # CLI override for aspect ratio

        # Calculate delay between requests (in seconds)
        self.request_delay = 60.0 / requests_per_minute
        self.last_request_time = 0

        # Initialize provider factory - use SDK config directory
        config_dir = Path(__file__).parent.parent.parent / 'config'
        self.provider_factory = ProviderFactory(config_dir)
        self.provider = self.provider_factory.get_provider(model_alias)
        
        # Stats
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def process_csv(self, csv_path: Path, max_concurrent: int = 5):
        """Process CSV file with image prompts"""
        self.stats['start_time'] = datetime.now()
        
        # Load prompts from CSV
        prompts = self.load_csv(csv_path)
        self.stats['total'] = len(prompts)
        
        click.echo(f"📋 Loaded {len(prompts)} prompts from {csv_path}")
        click.echo(f"🤖 Using model: {self.model_alias}")
        click.echo(f"📂 Output directory: {self.output_dir}")
        click.echo(f"⏱️  Rate limit: {self.requests_per_minute} requests/minute")
        click.echo(f"⏳ Delay between requests: {self.request_delay:.1f} seconds")
        if self.aspect_ratio_override:
            click.echo(f"📐 Aspect ratio override: {self.aspect_ratio_override}")
        
        # Calculate estimated time
        estimated_minutes = len(prompts) / self.requests_per_minute
        click.echo(f"⏰ Estimated time: {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)")
        
        # Process with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = []
        for idx, prompt_data in enumerate(prompts):
            task = self.process_single_prompt(idx, prompt_data, semaphore)
            tasks.append(task)
        
        # Process all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stats['end_time'] = datetime.now()
        
        # Display summary
        self.display_summary()
        
        # Save results report
        self.save_results_report(prompts, results)
        
        # Close provider session properly
        if hasattr(self.provider, 'close'):
            await self.provider.close()
    
    def load_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load prompts from CSV file"""
        prompts = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row)
        
        return prompts
    
    async def process_single_prompt(self, idx: int, prompt_data: Dict[str, Any], semaphore: asyncio.Semaphore):
        """Process a single prompt with rate limiting and automatic retry on 429"""
        async with semaphore:
            retry_count = 0
            max_retries = 10  # Keep trying up to 10 times for quota errors
            
            while retry_count <= max_retries:
                # Enforce rate limit
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_delay:
                    await asyncio.sleep(self.request_delay - time_since_last)
                
                self.last_request_time = time.time()
                
                try:
                    # Extract prompt and parameters
                    prompt = prompt_data.get('prompt', '')
                    if not prompt:
                        raise ValueError("Empty prompt")
                    
                    # Build parameters (following style-vertex approach)
                    # Use CLI override if provided, otherwise use CSV value, otherwise default
                    aspect_ratio = self.aspect_ratio_override or prompt_data.get('aspect_ratio', '16:9')
                    
                    params = {
                        'prompt': prompt,
                        'model': self.model_alias,
                        'aspect_ratio': aspect_ratio,
                    }
                    
                    # Add style if specified
                    if 'style' in prompt_data:
                        # Map style to Vertex AI parameters
                        style_mapping = {
                            'vivid': {'safety_filter_level': 'block_few', 'person_generation': 'allow_all'},
                            'natural': {'safety_filter_level': 'block_some', 'person_generation': 'allow_adult'},
                            'photorealistic': {'safety_filter_level': 'block_some', 'person_generation': 'allow_adult'}
                        }
                        if prompt_data['style'] in style_mapping:
                            params.update(style_mapping[prompt_data['style']])
                    
                    # For Imagen 4.0 models
                    if self.model_alias in ['goo-4-img', 'goo-5-img']:
                        params['add_watermark'] = True
                        params['number_of_images'] = 1
                    
                    click.echo(f"[{idx+1}/{self.stats['total']}] Processing: {prompt[:60]}...")
                    
                    # Generate image
                    result = await self.provider.generate_image(**params)
                    
                    # Parse result
                    if isinstance(result, str):
                        result = json.loads(result)
                    
                    # Save image
                    if result.get('images') and len(result['images']) > 0:
                        image_data = result['images'][0]
                        
                        if 'b64_json' in image_data:
                            # Decode and save image
                            image_bytes = base64.b64decode(image_data['b64_json'])
                            
                            # Generate filename
                            batch_id = prompt_data.get('batch_id', f'img_{idx:04d}')
                            filename = f"{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            filepath = self.output_dir / filename
                            
                            with open(filepath, 'wb') as f:
                                f.write(image_bytes)
                            
                            # Save metadata
                            metadata = {
                                'index': idx,
                                'prompt': prompt,
                                'prompt_data': prompt_data,
                                'model': self.model_alias,
                                'filename': filename,
                                'timestamp': datetime.now().isoformat(),
                                'safety_rating': image_data.get('safety_rating', 'unknown')
                            }
                            
                            metadata_file = self.output_dir / f"{batch_id}_metadata.json"
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            self.stats['successful'] += 1
                            click.echo(f"  ✅ Saved: {filename}")
                            
                            return {'success': True, 'filename': filename, 'index': idx}
                        else:
                            raise ValueError("No image data in response")
                    else:
                        raise ValueError("No images in response")
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check if it's a quota error (429)
                    if "429" in error_msg:
                        retry_count += 1
                        if retry_count <= max_retries:
                            # Calculate backoff time - exponential backoff with jitter
                            backoff_time = min(30 * (2 ** (retry_count - 1)), 300)  # Max 5 minutes
                            click.echo(f"  ⏸️  Quota exceeded - retry {retry_count}/{max_retries} in {backoff_time}s...")
                            await asyncio.sleep(backoff_time)
                            continue  # Retry the same prompt
                        else:
                            click.echo(f"  ❌ Failed after {max_retries} retries: {error_msg}")
                    
                    # Check if it's a timeout error (503)
                    elif "503" in error_msg:
                        retry_count += 1
                        if retry_count <= 3:  # Less retries for timeouts
                            backoff_time = 10 * retry_count
                            click.echo(f"  ⏸️  Timeout - retry {retry_count}/3 in {backoff_time}s...")
                            await asyncio.sleep(backoff_time)
                            continue  # Retry the same prompt
                        else:
                            click.echo(f"  ❌ Failed after 3 timeout retries: {error_msg}")
                    
                    else:
                        # Other errors - don't retry
                        click.echo(f"  ❌ Failed: {error_msg}")
                    
                    # If we get here, we've exhausted retries or hit a non-retryable error
                    self.stats['failed'] += 1
                    return {'success': False, 'error': error_msg, 'index': idx}
            
            # Should never reach here, but just in case
            self.stats['failed'] += 1
            return {'success': False, 'error': 'Unknown error', 'index': idx}
    
    def display_summary(self):
        """Display processing summary"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        click.echo("\n" + "="*60)
        click.echo("✨ Batch Processing Complete!")
        click.echo("="*60)
        click.echo(f"Total prompts: {self.stats['total']}")
        click.echo(f"✅ Successful: {self.stats['successful']}")
        click.echo(f"❌ Failed: {self.stats['failed']}")
        click.echo(f"⏱️  Duration: {duration:.1f} seconds")
        click.echo(f"📊 Success rate: {self.stats['successful']/self.stats['total']*100:.1f}%")
        click.echo(f"🚀 Processing rate: {self.stats['total']/duration:.1f} prompts/second")
    
    def save_results_report(self, prompts: List[Dict], results: List[Dict]):
        """Save detailed results report"""
        # Convert datetime objects to strings for JSON serialization
        summary_stats = self.stats.copy()
        if summary_stats.get('start_time'):
            summary_stats['start_time'] = summary_stats['start_time'].isoformat()
        if summary_stats.get('end_time'):
            summary_stats['end_time'] = summary_stats['end_time'].isoformat()
        
        report = {
            'summary': summary_stats,
            'model': self.model_alias,
            'results': []
        }
        
        for prompt_data, result in zip(prompts, results):
            if isinstance(result, dict):
                report['results'].append({
                    'prompt': prompt_data.get('prompt', ''),
                    'batch_id': prompt_data.get('batch_id', ''),
                    'success': result.get('success', False),
                    'filename': result.get('filename', ''),
                    'error': result.get('error', '')
                })
        
        report_file = self.output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        click.echo(f"\n📄 Report saved: {report_file}")


@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--model', '-m', default='goo-4-img', help='Model: goo-4-img (Ultra), goo-5-img (Fast)')
@click.option('--output', '-o', default='batch_output', help='Output directory')
@click.option('--concurrent', '-c', default=1, help='Max concurrent requests (default: 1 for rate limiting)')
@click.option('--rpm', '-r', default=6, help='Requests per minute (default: 6 for startup accounts)')
@click.option('--aspect-ratio', '-a', type=click.Choice(['16:9', '4:3', '9:16', '1:1', '3:4']), 
              help='Override aspect ratio for all images (default: use CSV values)')
def batch_process(csv_file, model, output, concurrent, rpm, aspect_ratio):
    """
    Automated Batch Vertex AI Image Processor
    
    Process CSV files with image prompts using Imagen 4.0 models.
    
    Examples:
        # Default: 6 requests/minute with 1 concurrent
        python cli/image/linear_vertex_image.py blog_images.csv --model goo-4-img

        # Custom rate limit: 10 requests/minute
        python cli/image/linear_vertex_image.py blog_images.csv --rpm 10

        # Override aspect ratio for all images
        python cli/image/linear_vertex_image.py blog_images.csv --aspect-ratio 16:9

        # Faster model with portrait orientation
        python cli/image/linear_vertex_image.py blog_images.csv --model goo-5-img --rpm 20 -a 9:16
    """
    processor = BatchVertexProcessor(
        model_alias=model, 
        output_dir=output, 
        requests_per_minute=rpm,
        aspect_ratio=aspect_ratio
    )
    asyncio.run(processor.process_csv(Path(csv_file), max_concurrent=concurrent))


def main():
    """Entry point for CLI"""
    batch_process()

if __name__ == '__main__':
    main()
