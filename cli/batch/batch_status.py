#!/usr/bin/env python3
"""
Batch Job Status Checker and Result Retriever

Monitor batch jobs and retrieve results from OpenAI or Anthropic.

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import click
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from processors.batch_submitter import UnifiedBatchSubmitter
from utils.output_paths import generate_cli_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_batch_status(batch_id: str, provider: str):
    """Check the status of a batch job"""
    
    submitter = UnifiedBatchSubmitter()
    
    try:
        status = await submitter.check_status(batch_id, provider)
        
        # Display status based on provider format
        if provider == "openai":
            click.echo(f"\n📊 Batch Status: {status['status']}")
            click.echo(f"📋 Batch ID: {batch_id}")
            click.echo(f"🏢 Provider: OpenAI")
            click.echo(f"⏰ Created: {status.get('created_at', 'N/A')}")
            
            if status.get("request_counts"):
                counts = status["request_counts"]
                total = counts.get('total', 0)
                completed = counts.get('completed', 0)
                failed = counts.get('failed', 0)
                
                click.echo(f"\n📈 Progress:")
                click.echo(f"   Total: {total}")
                click.echo(f"   Completed: {completed} ({completed/total*100:.1f}%)" if total > 0 else "   Completed: 0")
                click.echo(f"   Failed: {failed}")
                
                # Progress bar
                if total > 0:
                    progress = completed / total
                    bar_length = 30
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    click.echo(f"   [{bar}] {progress*100:.1f}%")
        
        elif provider == "anthropic":
            click.echo(f"\n📊 Batch Status: {status['processing_status']}")
            click.echo(f"📋 Batch ID: {batch_id}")
            click.echo(f"🏢 Provider: Anthropic")
            click.echo(f"⏰ Created: {status.get('created_at', 'N/A')}")
            click.echo(f"⏰ Expires: {status.get('expires_at', 'N/A')}")
            
            if status.get("request_counts"):
                counts = status["request_counts"]
                total = sum(counts.values())
                succeeded = counts.get('succeeded', 0)
                processing = counts.get('processing', 0)
                errored = counts.get('errored', 0)
                expired = counts.get('expired', 0)
                
                click.echo(f"\n📈 Progress:")
                click.echo(f"   Total: {total}")
                click.echo(f"   Processing: {processing}")
                click.echo(f"   Succeeded: {succeeded} ({succeeded/total*100:.1f}%)" if total > 0 else "   Succeeded: 0")
                click.echo(f"   Errored: {errored}")
                click.echo(f"   Expired: {expired}")
                
                # Progress bar
                if total > 0:
                    progress = succeeded / total
                    bar_length = 30
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    click.echo(f"   [{bar}] {progress*100:.1f}%")
        
        elif provider == "gemini":
            state = status.get("state", "UNKNOWN")
            click.echo(f"\n📊 Batch Status: {state}")
            click.echo(f"📋 Batch ID: {batch_id}")
            click.echo(f"🏢 Provider: Google Gemini")
            click.echo(f"⏰ Created: {status.get('createTime', 'N/A')}")
            click.echo(f"⏰ Updated: {status.get('updateTime', 'N/A')}")
            
            if "completedTaskCount" in status:
                completed = status.get("completedTaskCount", 0)
                total = status.get("taskCount", 0)
                
                click.echo(f"\n📈 Progress:")
                click.echo(f"   Total tasks: {total}")
                click.echo(f"   Completed: {completed} ({completed/total*100:.1f}%)" if total > 0 else "   Completed: 0")
                
                # Progress bar
                if total > 0:
                    progress = completed / total
                    bar_length = 30
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    click.echo(f"   [{bar}] {progress*100:.1f}%")
            
            if "error" in status:
                click.echo(f"\n❌ Error: {status['error']}")
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Failed to check status: {str(e)}")
        raise


async def retrieve_batch_results(batch_id: str, provider: str, output_dir: Path):
    """Retrieve results from a completed batch"""
    
    submitter = UnifiedBatchSubmitter()
    
    try:
        # First check status
        status = await check_batch_status(batch_id, provider)
        
        # Check if complete
        is_complete = False
        if provider == "openai":
            is_complete = status["status"] == "completed"
        elif provider == "anthropic":
            is_complete = status["processing_status"] == "ended"
        elif provider == "gemini":
            is_complete = status.get("state") == "JOB_STATE_SUCCEEDED"
        
        if not is_complete:
            click.echo(f"\n⏳ Batch is not yet complete. Current status: {status.get('status') or status.get('processing_status')}")
            return None
        
        # Retrieve results
        click.echo(f"\n📥 Retrieving results...")
        results = await submitter.retrieve_results(batch_id, provider)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON format
        results_file = output_dir / f"batch_{batch_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"✅ Results retrieved successfully!")
        click.echo(f"📊 Total results: {len(results)}")
        click.echo(f"📄 Results saved: {results_file}")
        
        # Also save summary
        summary_file = output_dir / f"batch_{batch_id}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Batch ID: {batch_id}\n")
            f.write(f"Provider: {provider}\n")
            f.write(f"Retrieved at: {datetime.now().isoformat()}\n")
            f.write(f"Total results: {len(results)}\n\n")
            
            # Count successes and failures
            successful = sum(1 for r in results if not r.get("error"))
            failed = len(results) - successful
            
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            
            if failed > 0:
                f.write(f"\nFailed requests:\n")
                for r in results:
                    if r.get("error"):
                        f.write(f"  - {r.get('custom_id', 'unknown')}: {r['error']}\n")
        
        click.echo(f"📄 Summary saved: {summary_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve results: {str(e)}")
        raise


@click.command()
@click.argument('batch_id')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'gemini']), required=True,
              help='Provider that processed the batch')
@click.option('--retrieve', '-r', is_flag=True,
              help='Retrieve results if batch is complete')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--wait', '-w', is_flag=True,
              help='Wait for completion and retrieve results')
@click.option('--check-interval', default=60,
              help='Seconds between checks when waiting (default: 60)')
def main(batch_id, provider, retrieve, output, wait, check_interval):
    """
    Check batch job status and retrieve results.
    
    Examples:
        # Check status only
        batch-status batch_abc123 --provider openai
        
        # Retrieve results if complete
        batch-status batch_abc123 --provider anthropic --retrieve
        
        # Wait for completion and retrieve
        batch-status batch_abc123 --provider openai --wait
    """
    
    # Generate output directory if retrieving
    if retrieve or wait:
        if not output:
            output = generate_cli_output_directory("batch_results", batch_id[:8])
        output_dir = Path(output)
    else:
        output_dir = None
    
    click.echo(f"🔥 Sacred Wrapper SDK - Batch Status Checker")
    click.echo(f"📋 Checking batch: {batch_id}")
    
    async def run_async():
        if wait:
            # Wait for completion
            click.echo(f"⏳ Waiting for batch completion...")
            
            submitter = UnifiedBatchSubmitter()
            results = await submitter.wait_for_completion(
                batch_id=batch_id,
                provider=provider,
                check_interval=check_interval
            )
            
            # Save results
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / f"batch_{batch_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            click.echo(f"✅ Batch completed and results retrieved!")
            click.echo(f"📊 Total results: {len(results)}")
            click.echo(f"📄 Results saved: {results_file}")
            
        else:
            # Just check status
            status = await check_batch_status(batch_id, provider)
            
            # Retrieve if requested and complete
            if retrieve:
                is_complete = False
                if provider == "openai":
                    is_complete = status["status"] == "completed"
                elif provider == "anthropic":
                    is_complete = status["processing_status"] == "ended"
                
                if is_complete:
                    await retrieve_batch_results(batch_id, provider, output_dir)
                else:
                    click.echo(f"\n❌ Cannot retrieve - batch is not complete")
    
    try:
        asyncio.run(run_async())
        return 0
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        return 1


if __name__ == '__main__':
    main()