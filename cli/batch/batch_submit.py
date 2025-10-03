#!/usr/bin/env python3
"""
Unified Batch Job Submission CLI

Submit massive batch jobs to OpenAI or Anthropic for 50% cost savings.
Supports CSV and JSON input with automatic provider routing.

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import click
import json
import csv
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict

# Add parent SDK directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from processors.batch_submitter import UnifiedBatchSubmitter
from utils.output_paths import generate_cli_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prompts_from_csv(file_path: Path) -> List[Dict]:
    """
    Load prompts from CSV file
    
    Expected columns: prompt, model (optional), custom_id (optional)
    """
    prompts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompt_data = {
                "prompt": row.get("prompt", ""),
                "custom_id": row.get("custom_id", f"request-{idx}"),
                "model": row.get("model", "gpt-5-nano"),
            }
            
            # Add optional parameters if present
            if "temperature" in row:
                prompt_data["temperature"] = float(row["temperature"])
            if "max_tokens" in row:
                prompt_data["max_tokens"] = int(row["max_tokens"])
            
            prompts.append(prompt_data)
    
    return prompts


def load_prompts_from_json(file_path: Path) -> List[Dict]:
    """
    Load prompts from JSON file
    
    Expected format: List of dictionaries with 'prompt' key
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "prompts" in data:
        return data["prompts"]
    else:
        raise ValueError("JSON must be a list or dict with 'prompts' key")


async def submit_batch_async(
    input_file: Path,
    model: str,
    provider: str,
    output_dir: Path,
    priority: str,
    wait: bool,
    check_interval: int
):
    """
    Submit batch job and optionally wait for results
    """
    
    # Load prompts based on file type
    file_ext = input_file.suffix.lower()
    
    if file_ext == '.csv':
        prompts = load_prompts_from_csv(input_file)
    elif file_ext == '.json':
        prompts = load_prompts_from_json(input_file)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Use .csv or .json")
    
    logger.info(f"📄 Loaded {len(prompts)} prompts from {input_file}")
    
    # Override model if specified
    if model:
        for prompt in prompts:
            prompt["model"] = model
    
    # Initialize batch submitter
    submitter = UnifiedBatchSubmitter()
    
    # Prepare kwargs based on provider
    kwargs = {}
    if provider == "anthropic":
        kwargs["processing_priority"] = priority
    elif provider == "openai":
        kwargs["completion_window"] = "24h"
    
    # Submit batch
    logger.info(f"🚀 Submitting batch to {provider or 'auto-detected provider'}...")
    
    try:
        batch_response = await submitter.submit_batch(
            requests=prompts,
            provider=provider,
            **kwargs
        )
        
        # Extract batch ID (different key names per provider)
        batch_id = batch_response.get("id") or batch_response.get("batch_id")
        
        # Save batch metadata
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = output_dir / f"batch_{batch_id}_metadata.json"
        
        metadata = {
            "batch_id": batch_id,
            "provider": provider or submitter.get_provider_for_model(prompts[0]["model"]),
            "submitted_at": datetime.now().isoformat(),
            "total_requests": len(prompts),
            "input_file": str(input_file),
            "model": model or "mixed",
            "batch_response": batch_response
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Batch submitted successfully!")
        logger.info(f"📋 Batch ID: {batch_id}")
        logger.info(f"📄 Metadata saved: {metadata_file}")
        
        # Wait for completion if requested
        if wait:
            logger.info(f"⏳ Waiting for batch completion (checking every {check_interval}s)...")
            
            detected_provider = metadata["provider"]
            results = await submitter.wait_for_completion(
                batch_id=batch_id,
                provider=detected_provider,
                check_interval=check_interval
            )
            
            # Save results
            results_file = output_dir / f"batch_{batch_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Batch completed!")
            logger.info(f"📊 Total results: {len(results)}")
            logger.info(f"📄 Results saved: {results_file}")
            
            # Also save as CSV for easier analysis
            csv_file = output_dir / f"batch_{batch_id}_results.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if results:
                    # Extract fields from first result
                    fieldnames = ["custom_id", "response", "error"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in results:
                        row = {
                            "custom_id": result.get("custom_id", ""),
                            "response": json.dumps(result.get("response", {})),
                            "error": result.get("error", "")
                        }
                        writer.writerow(row)
            
            logger.info(f"📄 CSV results saved: {csv_file}")
        
        return batch_id
        
    except Exception as e:
        logger.error(f"❌ Batch submission failed: {str(e)}")
        raise


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--model', '-m', help='Model to use (overrides file values)')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'gemini']), 
              help='Provider to use (auto-detected if not specified)')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--priority', type=click.Choice(['default', 'urgent']), 
              default='default', help='Processing priority (Anthropic only, urgent = 2x cost)')
@click.option('--wait', '-w', is_flag=True, 
              help='Wait for batch to complete and download results')
@click.option('--check-interval', default=60, 
              help='Seconds between status checks when waiting (default: 60)')
def main(input_file, model, provider, output, priority, wait, check_interval):
    """
    Submit batch jobs to OpenAI or Anthropic for 50% cost savings.
    
    Supports CSV and JSON input files with automatic provider routing.
    
    CSV Format:
        prompt,model,custom_id,temperature,max_tokens
        "What is AI?",gpt-5-nano,req-001,0.7,1000
        "Explain quantum computing",claude-3-5-sonnet,req-002,0.5,2000
    
    JSON Format:
        [
            {"prompt": "What is AI?", "model": "gpt-5-nano"},
            {"prompt": "Explain quantum", "model": "claude-3-5-sonnet"}
        ]
    
    Examples:
        # Submit OpenAI batch and wait for results
        batch-submit prompts.csv --provider openai --wait
        
        # Submit Anthropic batch with urgent priority
        batch-submit prompts.json --provider anthropic --priority urgent
        
        # Auto-detect provider from model names
        batch-submit mixed_prompts.csv --wait
        
        # Override all models to use GPT-4
        batch-submit prompts.csv --model gpt-5 --wait
    """
    
    input_path = Path(input_file)
    
    # Generate output directory
    if not output:
        output = generate_cli_output_directory("batch_submit", input_path.stem)
    output_dir = Path(output)
    
    click.echo(f"🔥 Sacred Wrapper SDK - Batch Job Submitter")
    click.echo(f"📄 Input file: {input_path}")
    click.echo(f"📂 Output directory: {output_dir}")
    
    if model:
        click.echo(f"🤖 Model override: {model}")
    if provider:
        click.echo(f"🏢 Provider: {provider}")
    if priority != 'default':
        click.echo(f"⚡ Priority: {priority}")
    if wait:
        click.echo(f"⏳ Will wait for completion")
    
    # Run async submission
    try:
        batch_id = asyncio.run(submit_batch_async(
            input_file=input_path,
            model=model,
            provider=provider,
            output_dir=output_dir,
            priority=priority,
            wait=wait,
            check_interval=check_interval
        ))
        
        if not wait:
            click.echo(f"\n✅ Batch submitted successfully!")
            click.echo(f"📋 Batch ID: {batch_id}")
            click.echo(f"\nTo check status later, use:")
            click.echo(f"  batch-status {batch_id} --provider {provider or 'auto'}")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        return 1


if __name__ == '__main__':
    main()