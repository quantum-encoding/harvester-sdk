#!/usr/bin/env python3
"""
CSV Processor - Universal AI Conductor & CSV Processor
© 2025 QUANTUM ENCODING LTD
Contact: info@quantumencoding.io
Website: https://quantumencoding.io

The ultimate AI processing pipeline: Convert any format → CSV → Batch AI Processing

This module provides the complete workflow for processing any data format through
the harvesting engine's flexible provider system with full CSV batch capabilities.

Usage:
    ./csv_processor.py convert input.txt                    # Convert to CSV
    ./csv_processor.py process data.csv --template advice   # Process CSV with AI
    ./csv_processor.py pipeline input.txt --template research --model grp-quality  # Full pipeline
    ./csv_processor.py --batch data.csv --template research --model gpt-5  # Batch flag
"""

import click
import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.batch_processor import BatchProcessor, BatchJob
from providers.provider_factory import ProviderFactory
from utils.progress_tracker import ProgressTracker
from utils.output_manager import OutputManager

# Import our new converter
from converter import UniversalConverter

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Mothership:
    """Universal AI conductor integrating conversion and CSV processing."""
    
    def __init__(self):
        # Initialize harvesting engine components
        config_dir = Path(__file__).parent / 'config'
        self.provider_factory = ProviderFactory(config_dir)
        self.output_manager = OutputManager()
        self.converter = UniversalConverter()
        
        # Load model groups
        self.model_groups = self._load_model_groups()
        
        click.echo(f"🚀 Mothership initialized")
        click.echo(f"🤖 Available providers: {', '.join(self.provider_factory.list_providers())}")
    
    def _load_model_groups(self) -> Dict[str, List[str]]:
        """Load model groups from config"""
        try:
            import yaml
            config_dir = Path(__file__).parent / 'config'
            providers_path = config_dir / 'providers.yaml'
            
            with open(providers_path, 'r') as f:
                providers_config = yaml.safe_load(f)
            
            return providers_config.get('groups', {})
        except Exception as e:
            logger.warning(f"Error loading model groups: {e}")
            # Return empty groups - they'll be loaded from config
            return {}
    
    def _resolve_models(self, model_input: str) -> List[str]:
        """Resolve model input to list of actual models"""
        
        # Handle 'all' - return all available models from groups
        if model_input == 'all':
            return self.model_groups.get('all', [])
        
        # Handle group selection (grp-fast, grp-quality, etc.)
        if model_input.startswith('grp-'):
            return self.model_groups.get(model_input, [model_input])
        
        # Handle comma-separated models
        if ',' in model_input:
            return [m.strip() for m in model_input.split(',')]
        
        # Single model
        return [model_input]
    
    async def convert_only(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        template_name: Optional[str] = None,
        strategy: str = "auto",
        preview: bool = False
    ) -> Dict[str, Any]:
        """Convert any format to CSV only."""
        
        # Generate output path if not provided
        if not output_path:
            input_file = Path(input_path)
            output_path = f"{input_file.stem}_converted.csv"
        
        if preview:
            click.echo(f"🔍 Would convert: {input_path} → {output_path}")
            return {"preview": True, "input_file": input_path, "output_file": output_path}
        
        # Convert using our converter
        report = await self.converter.convert_to_csv(
            input_path,
            output_path,
            template_name=template_name,
            conversion_strategy=strategy
        )
        
        click.echo(f"✅ Conversion complete: {output_path}")
        return report
    
    async def process_csv(
        self,
        csv_file: str,
        template_name: str,
        models: List[str],
        max_concurrent: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        chunk_size: int = 50,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process CSV file with AI models."""
        
        start_time = datetime.now()
        
        # Load CSV records
        records = await self._load_csv_records(csv_file)
        click.echo(f"📊 Loaded {len(records)} records from CSV")
        
        # Determine processing method based on size
        if len(records) > 500:
            click.echo(f"🚀 Large dataset detected, using chunked processing")
            return await self._process_large_csv(
                records, template_name, models, chunk_size, max_concurrent, verbose
            )
        else:
            click.echo(f"🚀 Processing {len(records)} records with {len(models)} models")
            return await self._process_standard_csv(
                records, template_name, models, max_concurrent, temperature, max_tokens, verbose
            )
    
    async def _load_csv_records(self, csv_file: str) -> List[Dict[str, Any]]:
        """Load records from CSV file."""
        records = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records
    
    async def _process_standard_csv(
        self,
        records: List[Dict[str, Any]],
        template_name: str,
        models: List[str],
        max_concurrent: int,
        temperature: float,
        max_tokens: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """Process CSV with standard batch processing."""
        
        # Simple template mapping
        templates = {
            'advice': 'You are a strategic advisor. Provide guidance on: {request}',
            'research': 'You are a research analyst. Analyze: {request}',
            'explain': 'You are an expert educator. Explain clearly: {request}',
            'code_assist': 'You are a senior software engineer. Help with: {request}',
            'quick': '{request}\n\nPlease provide a helpful, concise response.',
            'writing': 'You are a skilled writer. Help with this writing task: {request}'
        }
        
        template_text = templates.get(template_name, '{request}\n\nPlease provide a comprehensive response.')
        
        # Create batch jobs for all records across all models
        jobs = []
        job_id = 0
        
        for model in models:
            for i, record in enumerate(records):
                # Extract request content from CSV record
                request = self._extract_request_from_record(record)
                
                # Render template
                rendered_prompt = template_text.format(request=request)
                
                jobs.append(BatchJob(
                    id=f"csv_{job_id}_{model}",
                    prompt=rendered_prompt,
                    model=model,
                    metadata={
                        'record_index': i,
                        'original_record': record,
                        'template': template_name,
                        'csv_model': model
                    }
                ))
                job_id += 1
        
        # Process batch using harvesting engine
        batch_processor = BatchProcessor(self.provider_factory, {
            'max_concurrent': max_concurrent,
            'output_dir': Path.cwd() / 'temp'
        })
        tracker = ProgressTracker()
        
        batch_result = await batch_processor.process_batch(jobs, tracker)
        
        # Organize results
        results_by_record = {}
        successful_jobs = 0
        failed_jobs = 0
        
        for job in batch_result.results:
            record_index = job.metadata.get('record_index', 0)
            model = job.metadata.get('csv_model', 'unknown')
            
            if record_index not in results_by_record:
                results_by_record[record_index] = {
                    'original_record': job.metadata.get('original_record', {}),
                    'results_by_model': {}
                }
            
            if job.status == 'completed' and job.result:
                results_by_record[record_index]['results_by_model'][model] = {
                    'success': True,
                    'response': job.result,
                    'metadata': job.metadata
                }
                successful_jobs += 1
            else:
                results_by_record[record_index]['results_by_model'][model] = {
                    'success': False,
                    'error': getattr(job, 'error', 'Unknown error'),
                    'metadata': job.metadata
                }
                failed_jobs += 1
        
        # Structure final result
        result = {
            'template_used': template_name,
            'models_used': models,
            'total_records': len(records),
            'total_jobs': len(jobs),
            'successful_jobs': successful_jobs,
            'failed_jobs': failed_jobs,
            'results_by_record': results_by_record,
            'metadata': {
                'mothership_csv_processing': True,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'max_concurrent': max_concurrent
            }
        }
        
        # Save results
        saved_files = self.output_manager.save_response(
            result=result,
            template_name=f"csv_{template_name}",
            model_used=f"multi_model_{len(models)}",
            auto_save=True
        )
        
        if saved_files:
            session_dir = saved_files['json'].parent
            click.echo(f"💾 CSV processing results saved to: {session_dir.name}")
        
        return result
    
    async def _process_large_csv(
        self,
        records: List[Dict[str, Any]],
        template_name: str,
        models: List[str],
        chunk_size: int,
        max_concurrent: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """Process large CSV with chunked processing."""
        
        # For large datasets, process in chunks
        total_chunks = (len(records) + chunk_size - 1) // chunk_size
        click.echo(f"📊 Processing {len(records)} records in {total_chunks} chunks")
        
        all_results = []
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(records))
            chunk_records = records[start_idx:end_idx]
            
            click.echo(f"🔄 Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_records)} records)")
            
            # Process chunk using standard method
            chunk_result = await self._process_standard_csv(
                chunk_records, template_name, models, max_concurrent, 0.7, 4096, verbose
            )
            
            all_results.append(chunk_result)
        
        # Combine results
        combined_result = {
            'template_used': template_name,
            'models_used': models,
            'total_records': len(records),
            'total_chunks': total_chunks,
            'chunk_results': all_results,
            'metadata': {
                'mothership_large_csv_processing': True,
                'chunk_size': chunk_size
            }
        }
        
        return combined_result
    
    def _extract_request_from_record(self, record: Dict[str, Any]) -> str:
        """Extract the main request content from CSV record."""
        
        # Try common field names in order of preference
        field_candidates = [
            'request', 'prompt', 'content', 'text', 'input', 'query', 
            'question', 'message', 'description', 'task', 'original_prompt'
        ]
        
        for field in field_candidates:
            if field in record and record[field]:
                return str(record[field])
        
        # If no standard field found, combine all non-empty values
        values = []
        for key, value in record.items():
            if value and str(value).strip():
                values.append(f"{key}: {value}")
        
        return '; '.join(values) if values else str(record)
    
    async def full_pipeline(
        self,
        input_path: str,
        template_name: str,
        models: List[str],
        conversion_strategy: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """Complete pipeline: convert → process with AI."""
        
        click.echo(f"🚀 Starting full pipeline: {input_path}")
        
        # Step 1: Convert to CSV
        click.echo(f"📄 Step 1: Converting to CSV...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"{Path(input_path).stem}_{timestamp}.csv"
        
        convert_result = await self.convert_only(
            input_path, csv_file, template_name, conversion_strategy
        )
        
        # Step 2: Process CSV
        click.echo(f"🤖 Step 2: Processing CSV with AI...")
        process_result = await self.process_csv(
            csv_file, template_name, models, **kwargs
        )
        
        # Combine results
        pipeline_result = {
            'pipeline_complete': True,
            'input_file': input_path,
            'csv_file': csv_file,
            'conversion_result': convert_result,
            'processing_result': process_result,
            'template_used': template_name,
            'models_used': models
        }
        
        click.echo(f"✅ Full pipeline complete!")
        click.echo(f"📊 Processed {convert_result.get('converted_records', 0)} records")
        click.echo(f"🤖 Used {len(models)} models")
        
        return pipeline_result

@click.command()
@click.argument('operation_or_path', required=False)
@click.argument('path_arg', required=False)
@click.option('--template', '-t', default='quick',
              help='Template: advice, research, explain, code_assist, quick, writing')
@click.option('--model', '-m', default='grp-fast',
              help='Model(s): single model, group (grp-fast, grp-quality), or comma-separated')
@click.option('--batch', is_flag=True,
              help='Process CSV file in batch mode (same as: csv_processor.py process file.csv)')
@click.option('--output', '-o', help='Output file path for conversion')
@click.option('--strategy', '-s', default='auto',
              type=click.Choice(['auto', 'template_based', 'line_prompts', 'structured_content']),
              help='Conversion strategy')
@click.option('--max-concurrent', type=int, default=5,
              help='Maximum concurrent requests')
@click.option('--chunk-size', type=int, default=50,
              help='Records per chunk for large datasets')
@click.option('--temperature', type=float, default=0.7,
              help='Temperature for generation')
@click.option('--max-tokens', type=int, default=4096,
              help='Maximum tokens per response')
@click.option('--preview', is_flag=True,
              help='Preview operation without execution')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def mothership(
    operation_or_path: Optional[str],
    path_arg: Optional[str],
    template: str,
    model: str,
    batch: bool,
    output: Optional[str],
    strategy: str,
    max_concurrent: int,
    chunk_size: int,
    temperature: float,
    max_tokens: int,
    preview: bool,
    verbose: bool
):
    """
    🚀 CSV Processor - Universal AI Conductor & CSV Processor
    
    The complete AI processing pipeline: Convert any format → CSV → Batch AI Processing
    
    OPERATIONS:
        convert <file>     Convert any format to CSV
        process <csv>      Process CSV file with AI models
        pipeline <file>    Full pipeline: convert → process
        
    EXAMPLES:
        # Convert text file to CSV
        ./csv_processor.py convert prompts.txt
        
        # Process CSV with AI (standard usage)
        ./csv_processor.py process data.csv --template advice --model gpt-5
        
        # Batch flag (same as process)
        ./csv_processor.py data.csv --batch --template research --model grp-quality
        
        # Full pipeline (convert → process)
        ./csv_processor.py pipeline input.txt --template research --model grp-quality
        
        # Multi-model comparison
        ./csv_processor.py process data.csv --template advice --model gpt-5,goo-1,gpt-1
    """
    
    click.echo(click.style("🚀 CSV Processor - Universal AI Conductor", fg='cyan', bold=True))
    click.echo(click.style("© 2025 QUANTUM ENCODING LTD | info@quantumencoding.io", fg='blue'))
    
    # Initialize mothership
    mothership_instance = Mothership()
    
    # Resolve models
    models = mothership_instance._resolve_models(model)
    
    # Parse operation and path
    if operation_or_path in ['convert', 'process', 'pipeline']:
        operation = operation_or_path
        file_path = path_arg
    else:
        # Determine operation from context
        if batch:
            operation = 'process'
            file_path = operation_or_path
        elif operation_or_path and Path(operation_or_path).suffix.lower() == '.csv':
            operation = 'process'
            file_path = operation_or_path
        else:
            operation = 'pipeline'  # Default to full pipeline
            file_path = operation_or_path
    
    # Validate file path
    if not file_path:
        click.echo("❌ File path required")
        click.echo("Examples:")
        click.echo("  ./csv_processor.py convert input.txt")
        click.echo("  ./csv_processor.py process data.csv --template advice")
        click.echo("  ./csv_processor.py --batch data.csv --template research")
        sys.exit(1)
    
    if not Path(file_path).exists():
        click.echo(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # Show operation info
    click.echo(f"📋 Operation: {operation}")
    click.echo(f"📄 File: {file_path}")
    click.echo(f"🎯 Template: {template}")
    click.echo(f"🤖 Models: {', '.join(models)}")
    
    # Execute operation
    async def execute():
        try:
            if operation == 'convert':
                result = await mothership_instance.convert_only(
                    file_path, output, template, strategy, preview
                )
                
                if not preview:
                    click.echo(f"✅ Conversion complete: {result['output_file']}")
                    click.echo(f"📊 Converted {result['converted_records']} records")
            
            elif operation == 'process':
                if not file_path.endswith('.csv'):
                    click.echo(f"❌ Process operation requires CSV file, got: {file_path}")
                    sys.exit(1)
                
                result = await mothership_instance.process_csv(
                    file_path, template, models, max_concurrent, 
                    temperature, max_tokens, chunk_size, verbose
                )
                
                click.echo(f"✅ CSV processing complete!")
                click.echo(f"📊 Records: {result['total_records']}")
                click.echo(f"🚀 Jobs: {result['total_jobs']}")
                click.echo(f"✅ Successful: {result['successful_jobs']}")
                click.echo(f"❌ Failed: {result['failed_jobs']}")
            
            elif operation == 'pipeline':
                result = await mothership_instance.full_pipeline(
                    file_path, template, models, strategy,
                    max_concurrent=max_concurrent, chunk_size=chunk_size,
                    temperature=temperature, max_tokens=max_tokens, verbose=verbose
                )
                
                click.echo(f"✅ Full pipeline complete!")
                
        except KeyboardInterrupt:
            click.echo(click.style("\n⚠️ Operation interrupted by user", fg='yellow'))
            sys.exit(1)
        except Exception as e:
            click.echo(click.style(f"\n❌ Operation failed: {e}", fg='red'))
            if verbose:
                logger.exception("Detailed error:")
            sys.exit(1)
    
    # Run the operation
    asyncio.run(execute())

if __name__ == '__main__':
    mothership()