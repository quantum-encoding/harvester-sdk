#!/usr/bin/env python3
"""
Enhanced Template-Based Code Transformation Engine - UNIFIED EDITION

© 2025 QUANTUM ENCODING LTD
Contact: info@quantumencoding.io
Website: https://quantumencoding.io

Reforged with the Crown Jewel ParallelProcessor for military-grade parallelism.
Transform entire codebases with impossible speed and reliability.

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import os
import json
import csv
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import click
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvester_sdk.sdk import HarvesterSDK
from core.templater import Templater
from core.result_templater import ImageResultTemplater
from utils.output_manager import OutputManager
from utils.output_paths import generate_cli_output_directory

# THE CROWN JEWEL - Unified Parallel Processing
from processors.parallel import CodeRefactoringProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedCodeTransformer:
    """
    Code Transformation with the Crown Jewel ParallelProcessor
    
    This is the Great Unification in action - sequential bottlenecks eliminated forever.
    """
    
    def __init__(self, model: str = 'gemini-2.5-flash', template: str = 'documentation.j2'):
        self.model = model
        self.template = template
        self.sdk = HarvesterSDK()
        self.template_processor = Templater()
        self.result_templater = ImageResultTemplater()
        
        # Initialize the Crown Jewel
        self.processor = CodeRefactoringProcessor(
            max_workers=15,  # Optimized for code workloads
            rate_limit_per_minute=60,
            retry_attempts=5,
            backoff_multiplier=1.5
        )
    
    async def transform_file(self, operation: dict) -> dict:
        """
        Transform a single file using AI-powered templates
        
        This is now an async operation that can run in parallel with perfect coordination.
        """
        file_path = Path(operation['file_path'])
        template_variables = operation.get('template_vars', {})
        source_root = Path(operation.get('source_root', file_path.parent))
        output_root = Path(operation.get('output_root', '.'))
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Prepare context - MUST match template variables!
            context = {
                'code': content,  # Template expects 'code' not 'file_content'!
                'filename': file_path.name,  # Template expects 'filename' not 'file_name'!
                'language': file_path.suffix[1:] if file_path.suffix else 'text',  # Get language from extension
                'file_path': str(file_path),  # Extra context
                'source_directory': str(file_path.parent),  # Extra context
                **template_variables
            }
            
            # Generate AI response using the SDK
            prompt = self.template_processor.render(self.template, context)
            response = await self.sdk.async_generate_text(
                prompt=prompt,
                model=self.model
            )
            
            # Save the transformed content directly as a file
            # Preserve directory structure relative to source
            relative_path = file_path.relative_to(source_root)
            output_file = output_root / relative_path
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the transformed content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            
            return {
                'source_file': str(file_path),
                'output_file': str(output_file),
                'template': self.template,
                'model': self.model,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error transforming {file_path}: {str(e)}")
            raise
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on common patterns"""
        ignore_patterns = [
            # Version control
            '.git', '.svn', '.hg',
            # Dependencies
            'node_modules', 'venv', 'env', '.venv', '.env',
            # Build artifacts  
            'build', 'dist', 'target', '.next', '.nuxt',
            # Cache and temp
            '__pycache__', '.pytest_cache', '.mypy_cache',
            '.cache', 'tmp', 'temp',
            # IDE files
            '.vscode', '.idea', '*.swp', '*.swo',
            # OS files
            '.DS_Store', 'Thumbs.db',
            # Package files
            '*.egg-info', '.tox'
        ]
        
        path_str = str(file_path)
        file_name = file_path.name
        
        # Check if file or any parent directory matches ignore patterns
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True
        
        # Skip binary files by extension
        binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.obj',
            '.jpg', '.jpeg', '.png', '.gif', '.ico', '.svg',
            '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar'
        }
        
        if file_path.suffix.lower() in binary_extensions:
            return True
            
        return False

    async def transform_codebase(self, 
                                source_path: Path,
                                file_pattern: str,
                                max_files: int,
                                output_path: Path,
                                template_vars: dict = None) -> dict:
        """
        Transform an entire codebase using the Crown Jewel ParallelProcessor
        
        This replaces the old sequential loop with military-grade parallelism.
        """
        # Find files to process with proper filtering
        all_files = source_path.glob(file_pattern)
        files = []
        
        for file_path in all_files:
            # Skip directories and ignored files
            if file_path.is_file() and not self._should_ignore_file(file_path):
                files.append(file_path)
                if len(files) >= max_files:
                    break
        
        if not files:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return {'status': 'no_files', 'pattern': file_pattern}
        
        logger.info(f"🎯 Found {len(files)} files to transform")
        
        # Prepare operations for parallel processing
        operations = [
            {
                'file_path': str(file_path),
                'template_vars': template_vars or {},
                'source_root': str(source_path),
                'output_root': str(output_path),
                'index': idx
            }
            for idx, file_path in enumerate(files)
        ]
        
        # Progress callback for real-time updates
        async def progress_callback(progress: dict):
            if progress['completed'] % 5 == 0 or progress['completed'] == progress['total']:
                click.echo(f"⚡ Progress: {progress['completed']}/{progress['total']} "
                          f"({progress['progress_percent']:.1f}%)")
        
        # EXECUTE WITH THE CROWN JEWEL
        logger.info("🔥 Initiating parallel transformation with Crown Jewel processor...")
        
        batch_result = await self.processor.execute_batch(
            operations=operations,
            operation_handler=self.transform_file,
            progress_callback=progress_callback
        )
        
        # Save a summary of the transformation
        output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        summary_file = output_path / 'transformation_summary.json'
        
        # Create a cleaner summary for the JSON file
        summary = {
            'status': 'completed',
            'source_directory': str(source_path),
            'output_directory': str(output_path),
            'template': self.template,
            'model': self.model,
            'total_files': batch_result['total_operations'],
            'successful': batch_result['successful_operations'],
            'failed': batch_result['failed_operations'],
            'success_rate': batch_result['success_rate'],
            'duration_seconds': batch_result['duration_seconds'],
            'throughput_per_second': batch_result['throughput_per_second'],
            'files_processed': [
                {
                    'source': result['result']['source_file'],
                    'output': result['result']['output_file']
                }
                for result in batch_result['individual_results']
                if isinstance(result, dict) and result.get('status') == 'success'
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Display performance metrics
        metrics = self.processor.get_performance_metrics()
        logger.info(f"✨ Transformation Complete!")
        logger.info(f"📊 Success Rate: {metrics['efficiency_ratio']*100:.1f}%")
        logger.info(f"⚡ Average Response Time: {metrics['average_response_time']:.2f}s")
        logger.info(f"🚀 Throughput: {batch_result['throughput_per_second']:.1f} ops/second")
        
        return batch_result


@click.command()
@click.option('--source', '-s', required=True, help='Source directory to process')
@click.option('--template', '-t', default='documentation.j2', help='Template to use for transformation')
@click.option('--model', '-m', default='gemini-2.5-flash', help='AI model to use')
@click.option('--output', '-o', help='Custom output directory')
@click.option('--file-pattern', '-p', default='**/*', help='File pattern to match (glob syntax)')
@click.option('--max-files', default=50, help='Maximum number of files to process')
@click.option('--template-vars', help='JSON string of template variables')
@click.option('--dry-run', is_flag=True, help='Show what would be processed without executing')
@click.option('--workers', '-w', default=15, help='Number of parallel workers')
@click.option('--rpm', default=60, help='Requests per minute rate limit')
def main(source, template, model, output, file_pattern, max_files, template_vars, dry_run, workers, rpm):
    """
    Enhanced Template-Based Code Transformation Engine - UNIFIED EDITION
    
    Transform codebases using AI-powered templates with the Crown Jewel ParallelProcessor.
    Now with military-grade parallelism, intelligent rate limiting, and automatic retries.
    
    Examples:
        # Transform ALL files with 15 parallel workers (default)
        batch-code -s ./my_project -t documentation.j2
        
        # Process only Python files with custom parallelism
        batch-code -s ./src -t refactor.j2 -p "**/*.py" -w 20 --rpm 120
        
        # Dry run to see what would be processed
        batch-code -s ./code -t review.j2 --dry-run
    """
    
    # Generate sovereign output directory
    if not output:
        output = generate_cli_output_directory("batch_code", source)
    
    output_path = Path(output)
    source_path = Path(source)
    
    click.echo(f"🔥 Sacred Wrapper SDK - Code Transformation Engine (UNIFIED)")
    click.echo(f"⚡ Powered by the Crown Jewel ParallelProcessor")
    click.echo(f"📁 Source: {source}")
    click.echo(f"📋 Template: {template}")
    click.echo(f"🤖 Model: {model}")
    click.echo(f"📂 Output: {output_path}")
    click.echo(f"🔍 Pattern: {file_pattern}")
    click.echo(f"📊 Max files: {max_files}")
    click.echo(f"👥 Workers: {workers}")
    click.echo(f"⏱️  Rate limit: {rpm} requests/minute")
    
    if dry_run:
        click.echo("\n🌊 DRY RUN MODE - Showing files that would be processed:")
        files = list(source_path.glob(file_pattern))[:max_files]
        for file_path in files:
            click.echo(f"  - {file_path}")
        click.echo(f"\n📊 Total: {len(files)} files")
        return
    
    # Parse template variables
    template_variables = {}
    if template_vars:
        try:
            template_variables = json.loads(template_vars)
        except json.JSONDecodeError as e:
            click.echo(f"❌ Invalid template variables JSON: {e}")
            return 1
    
    # Verify source exists
    if not source_path.exists():
        click.echo(f"❌ Source path does not exist: {source}")
        return 1
    
    # Create async transformer with custom worker configuration
    transformer = UnifiedCodeTransformer(model=model, template=template)
    transformer.processor.max_workers = workers
    transformer.processor.rate_limit_per_minute = rpm
    
    # Run the async transformation
    click.echo("\n🚀 Initiating parallel transformation...")
    
    try:
        # Execute with the Crown Jewel
        result = asyncio.run(transformer.transform_codebase(
            source_path=source_path,
            file_pattern=file_pattern,
            max_files=max_files,
            output_path=output_path,
            template_vars=template_variables
        ))
        
        # Display final summary
        click.echo(f"\n{'='*60}")
        
        # Check if any files were processed
        if result.get('status') == 'no_files':
            click.echo("❌ NO FILES FOUND!")
            click.echo(f"{'='*60}")
            click.echo(f"🔍 Pattern searched: {result.get('pattern', 'unknown')}")
            click.echo(f"📁 Directory: {source}")
            click.echo("💡 Try adjusting the --file-pattern parameter")
            return 1
        
        click.echo("✨ THE GREAT UNIFICATION IS COMPLETE!")
        click.echo(f"{'='*60}")
        click.echo(f"📊 Total files: {result['total_operations']}")
        click.echo(f"✅ Successful: {result['successful_operations']}")
        click.echo(f"❌ Failed: {result['failed_operations']}")
        click.echo(f"🎯 Success rate: {result['success_rate']*100:.1f}%")
        click.echo(f"⚡ Throughput: {result['throughput_per_second']:.1f} files/second")
        click.echo(f"⏱️  Total time: {result['duration_seconds']:.1f} seconds")
        click.echo(f"📂 Results saved to: {output_path}")
        
        return 0 if result['failed_operations'] == 0 else 1
        
    except Exception as e:
        click.echo(f"❌ Critical error: {str(e)}")
        logger.exception("Transformation failed")
        return 1


if __name__ == '__main__':
    main()
