"""
JSON to CSV Converter for Image Generation Orchestration

Transforms JSON input containing image prompts and parameters into CSV format
for batch processing with the harvesting engine.
"""
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ImageOrchestrationConverter:
    """Converts JSON orchestration requests to CSV for batch image generation"""
    
    def __init__(self):
        # Standard CSV columns for image generation
        self.standard_columns = [
            'prompt',           # Primary image prompt
            'negative_prompt',  # What to avoid (optional)
            'style',           # Image style (vivid, natural, etc.)
            'aspect_ratio',    # Image dimensions
            'quality',         # Image quality (standard, hd)
            'size',           # Explicit size (1024x1024, etc.)
            'model',          # Model preference (gpt-1, goo-1, etc.)
            'seed',           # Random seed for reproducibility
            'output_format',  # url or b64_json
            'safety_level',   # Safety filter level
            'person_generation', # Person generation policy
            'batch_id',       # Batch identifier
            'priority',       # Processing priority
            'metadata'        # Additional metadata as JSON string
        ]
    
    def convert_json_to_csv(
        self,
        input_json: Union[str, Path, Dict, List],
        output_csv: Union[str, Path],
        batch_id: str = None,
        default_model: str = "gpt-1",
        flatten_strategy: str = "expand"
    ) -> Dict[str, Any]:
        """
        Convert JSON input to CSV format for batch processing
        
        Args:
            input_json: JSON file path, JSON string, or Python dict/list
            output_csv: Output CSV file path
            batch_id: Optional batch identifier
            default_model: Default model if not specified in JSON
            flatten_strategy: How to handle arrays ('expand' or 'serialize')
            
        Returns:
            Conversion report with statistics and metadata
        """
        # Load JSON data
        if isinstance(input_json, (str, Path)):
            if str(input_json).endswith('.json'):
                with open(input_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # Assume JSON string
                data = json.loads(input_json)
        else:
            data = input_json
        
        # Generate batch ID if not provided
        if not batch_id:
            batch_id = f"img_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert to records
        records = self._json_to_records(data, batch_id, default_model, flatten_strategy)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(records)
        
        # Ensure all standard columns exist
        for col in self.standard_columns:
            if col not in df.columns:
                df[col] = None
        
        # Reorder columns
        column_order = [col for col in self.standard_columns if col in df.columns]
        extra_columns = [col for col in df.columns if col not in self.standard_columns]
        df = df[column_order + extra_columns]
        
        # Save to CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Generate report
        report = {
            'input_source': str(input_json) if isinstance(input_json, (str, Path)) else 'direct_data',
            'output_csv': str(output_path),
            'batch_id': batch_id,
            'total_records': len(records),
            'columns': list(df.columns),
            'conversion_timestamp': datetime.now().isoformat(),
            'models_detected': list(df['model'].dropna().unique()) if 'model' in df.columns else [],
            'sample_prompts': df['prompt'].head(3).tolist() if 'prompt' in df.columns else [],
            'statistics': {
                'unique_prompts': df['prompt'].nunique() if 'prompt' in df.columns else 0,
                'models_used': df['model'].nunique() if 'model' in df.columns else 0,
                'has_negative_prompts': df['negative_prompt'].notna().sum() if 'negative_prompt' in df.columns else 0,
                'has_seeds': df['seed'].notna().sum() if 'seed' in df.columns else 0
            }
        }
        
        logger.info(f"Converted JSON to CSV: {len(records)} records → {output_path}")
        return report
    
    def _json_to_records(
        self,
        data: Union[Dict, List],
        batch_id: str,
        default_model: str,
        flatten_strategy: str
    ) -> List[Dict[str, Any]]:
        """Convert JSON data structure to list of records"""
        
        records = []
        
        if isinstance(data, list):
            # Array of prompts/requests
            for i, item in enumerate(data):
                record = self._process_item(item, batch_id, default_model, i)
                if flatten_strategy == "expand" and isinstance(record, list):
                    records.extend(record)
                else:
                    records.append(record)
        
        elif isinstance(data, dict):
            # Check for different JSON structures
            
            # Structure 1: {"prompts": [...], "config": {...}}
            if 'prompts' in data:
                config = data.get('config', {})
                for i, prompt_item in enumerate(data['prompts']):
                    record = self._process_item(prompt_item, batch_id, default_model, i, global_config=config)
                    if flatten_strategy == "expand" and isinstance(record, list):
                        records.extend(record)
                    else:
                        records.append(record)
            
            # Structure 2: {"batch_config": {...}, "requests": [...]}
            elif 'requests' in data:
                batch_config = data.get('batch_config', {})
                for i, request in enumerate(data['requests']):
                    record = self._process_item(request, batch_id, default_model, i, global_config=batch_config)
                    if flatten_strategy == "expand" and isinstance(record, list):
                        records.extend(record)
                    else:
                        records.append(record)
            
            # Structure 3: {"models": ["gpt-1", "goo-1"], "prompts": [...]}
            elif 'models' in data and 'prompts' in data:
                models = data['models']
                prompts = data['prompts']
                config = {k: v for k, v in data.items() if k not in ['models', 'prompts']}
                
                # Cross product: each prompt with each model
                for i, prompt in enumerate(prompts):
                    for j, model in enumerate(models):
                        record = self._process_item(prompt, batch_id, model, i * len(models) + j, global_config=config)
                        records.append(record)
            
            # Structure 4: Single request object
            else:
                record = self._process_item(data, batch_id, default_model, 0)
                if isinstance(record, list):
                    records.extend(record)
                else:
                    records.append(record)
        
        return records
    
    def _process_item(
        self,
        item: Union[str, Dict],
        batch_id: str,
        default_model: str,
        index: int,
        global_config: Dict = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process a single item into a record"""
        
        if global_config is None:
            global_config = {}
        
        # Handle string prompts
        if isinstance(item, str):
            return {
                'prompt': item,
                'model': default_model,
                'batch_id': batch_id,
                'priority': index,
                **global_config
            }
        
        # Handle dict items
        if isinstance(item, dict):
            record = {
                'batch_id': batch_id,
                'priority': index
            }
            
            # Map common fields
            field_mapping = {
                'prompt': ['prompt', 'text', 'description', 'image_prompt'],
                'negative_prompt': ['negative_prompt', 'negative', 'avoid', 'exclude'],
                'style': ['style', 'image_style', 'art_style'],
                'aspect_ratio': ['aspect_ratio', 'ratio', 'dimensions'],
                'quality': ['quality', 'image_quality'],
                'size': ['size', 'image_size', 'resolution'],
                'model': ['model', 'provider', 'engine'],
                'seed': ['seed', 'random_seed'],
                'output_format': ['output_format', 'format', 'response_format'],
                'safety_level': ['safety_level', 'safety', 'content_filter'],
                'person_generation': ['person_generation', 'people', 'persons']
            }
            
            # Apply field mapping
            for target_field, source_fields in field_mapping.items():
                for source_field in source_fields:
                    if source_field in item:
                        record[target_field] = item[source_field]
                        break
            
            # Apply global config
            for key, value in global_config.items():
                if key not in record:
                    record[key] = value
            
            # Set defaults
            if 'model' not in record:
                record['model'] = default_model
            
            # Handle special cases
            
            # Multiple models - expand to multiple records
            if 'models' in item:
                models = item['models'] if isinstance(item['models'], list) else [item['models']]
                records = []
                for i, model in enumerate(models):
                    record_copy = record.copy()
                    record_copy['model'] = model
                    record_copy['priority'] = index * 1000 + i  # Maintain order
                    records.append(record_copy)
                return records
            
            # Serialize complex metadata
            metadata = {}
            for key, value in item.items():
                if key not in [field for fields in field_mapping.values() for field in fields]:
                    if key not in ['models']:  # Skip already processed fields
                        metadata[key] = value
            
            if metadata:
                record['metadata'] = json.dumps(metadata)
            
            return record
        
        # Fallback for other types
        return {
            'prompt': str(item),
            'model': default_model,
            'batch_id': batch_id,
            'priority': index
        }
    
    def create_example_json(self, output_path: Union[str, Path]) -> None:
        """Create example JSON input files showing different supported formats"""
        
        examples = {
            # Example 1: Simple prompt array
            'simple_prompts.json': [
                "A majestic mountain landscape at sunset",
                "A futuristic city with flying cars",
                "A cozy coffee shop in autumn"
            ],
            
            # Example 2: Detailed request objects
            'detailed_requests.json': {
                "batch_config": {
                    "quality": "hd",
                    "aspect_ratio": "16:9",
                    "output_format": "url"
                },
                "requests": [
                    {
                        "prompt": "A serene lake with mountains reflected in the water",
                        "style": "vivid",
                        "model": "gpt-1",
                        "seed": 12345
                    },
                    {
                        "prompt": "An abstract painting of emotions",
                        "style": "natural",
                        "model": "goo-1",
                        "negative_prompt": "realistic, photographic"
                    }
                ]
            },
            
            # Example 3: Multi-model processing
            'multi_model.json': {
                "models": ["gpt-1", "goo-1", "goo-2"],
                "prompts": [
                    "A magical forest with glowing trees",
                    "A steampunk robot in a Victorian setting"
                ],
                "quality": "standard",
                "aspect_ratio": "1:1"
            },
            
            # Example 4: Complex orchestration
            'orchestration.json': {
                "prompts": [
                    {
                        "prompt": "A dragon soaring over a medieval castle",
                        "models": ["gpt-1", "goo-1"],
                        "variations": {
                            "styles": ["vivid", "natural"],
                            "qualities": ["standard", "hd"]
                        }
                    }
                ],
                "config": {
                    "output_format": "b64_json",
                    "safety_level": "block_some"
                }
            }
        }
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in examples.items():
            file_path = output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(examples)} example JSON files in {output_dir}")


# CLI interface for the converter
def main():
    """Command line interface for JSON to CSV conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert JSON to CSV for image generation orchestration")
    parser.add_argument('input', help='Input JSON file or JSON string')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('--batch-id', help='Batch identifier')
    parser.add_argument('--model', default='gpt-1', help='Default model')
    parser.add_argument('--strategy', choices=['expand', 'serialize'], default='expand',
                       help='Strategy for handling arrays')
    parser.add_argument('--create-examples', help='Create example JSON files in specified directory')
    
    args = parser.parse_args()
    
    converter = ImageOrchestrationConverter()
    
    if args.create_examples:
        converter.create_example_json(args.create_examples)
        print(f"Example JSON files created in {args.create_examples}")
        return
    
    if not args.output:
        # Generate output filename based on input
        if args.input.endswith('.json'):
            args.output = args.input.replace('.json', '.csv')
        else:
            args.output = f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        report = converter.convert_json_to_csv(
            input_json=args.input,
            output_csv=args.output,
            batch_id=args.batch_id,
            default_model=args.model,
            flatten_strategy=args.strategy
        )
        
        print(f"✅ Conversion complete!")
        print(f"📄 Input: {report['input_source']}")
        print(f"📊 Output: {report['output_csv']}")
        print(f"🔢 Records: {report['total_records']}")
        print(f"🤖 Models: {', '.join(report['models_detected'])}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())