"""
CSV Scanner for Image Generation Batch Processing

Reads CSV files containing image prompts and adapts them for the harvesting engine.
"""
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Generator, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CSVImageRecord:
    """Represents a single image generation request from CSV"""
    prompt: str
    negative_prompt: Optional[str] = None
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    quality: Optional[str] = None
    size: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    output_format: Optional[str] = None
    safety_level: Optional[str] = None
    person_generation: Optional[str] = None
    batch_id: Optional[str] = None
    priority: Optional[int] = None
    metadata: Optional[str] = None
    row_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing"""
        return {
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'style': self.style,
            'aspect_ratio': self.aspect_ratio,
            'quality': self.quality,
            'size': self.size,
            'model': self.model,
            'seed': self.seed,
            'output_format': self.output_format,
            'safety_level': self.safety_level,
            'person_generation': self.person_generation,
            'batch_id': self.batch_id,
            'priority': self.priority,
            'metadata': self.metadata,
            'row_number': self.row_number
        }

@dataclass 
class CSVScanResult:
    """Adapts CSV records to look like file scan results for harvesting engine compatibility"""
    path: Path
    size: int
    language: str
    relative_path: str
    csv_records: List[CSVImageRecord]
    total_prompts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "language": self.language,
            "size": self.size,
            "total_prompts": self.total_prompts,
            "csv_records": [record.to_dict() for record in self.csv_records]
        }

class CSVScanner:
    """Scans CSV files containing image generation prompts"""
    
    # Required columns for image generation
    REQUIRED_COLUMNS = ['prompt']
    
    # Optional columns with their default values
    OPTIONAL_COLUMNS = {
        'negative_prompt': None,
        'style': 'vivid',
        'aspect_ratio': '1:1',
        'quality': 'standard',
        'size': '1024x1024',
        'model': 'o1',  # Default to OpenAI DALL-E 3
        'seed': None,
        'output_format': 'url',
        'safety_level': 'block_some',
        'person_generation': 'dont_allow',
        'batch_id': None,
        'priority': None,
        'metadata': None
    }
    
    def __init__(self, profile: Dict[str, Any]):
        """
        Initialize CSV Scanner with profile configuration
        
        Args:
            profile: Profile configuration (similar to code scanner)
        """
        self.profile = profile
        self.stats = {
            'total_files_scanned': 0,
            'total_prompts': 0,
            'total_size_bytes': 0,
            'by_model': {},
            'skipped_files': 0,
            'skipped_rows': 0
        }
        
        # CSV-specific settings
        self.max_file_size = profile.get('max_file_size', 50 * 1024 * 1024)  # 50MB
        self.min_file_size = profile.get('min_file_size', 1)
        self.max_prompts_per_file = profile.get('max_prompts_per_file', 10000)
        self.skip_empty_prompts = profile.get('skip_empty_prompts', True)
        
        logger.info(f"CSV Scanner initialized for image generation")
        
    def scan(self, path_input: Path) -> Generator[CSVScanResult, None, None]:
        """
        Scan CSV files for image generation prompts
        
        Args:
            path_input: Path to CSV file or directory containing CSV files
            
        Yields:
            CSVScanResult objects containing parsed image generation requests
        """
        path_input = Path(path_input).resolve()
        
        if path_input.is_file():
            # Single CSV file
            if path_input.suffix.lower() == '.csv':
                yield from self._process_csv_file(path_input)
            else:
                logger.warning(f"File {path_input} is not a CSV file")
        
        elif path_input.is_dir():
            # Directory - scan for CSV files
            logger.info(f"Scanning directory {path_input} for CSV files")
            for csv_file in path_input.glob('**/*.csv'):
                if self._should_process_file(csv_file):
                    yield from self._process_csv_file(csv_file)
        
        else:
            logger.error(f"Path {path_input} does not exist")
    
    def _process_csv_file(self, csv_path: Path) -> Generator[CSVScanResult, None, None]:
        """Process a single CSV file"""
        try:
            logger.info(f"Processing CSV file: {csv_path}")
            
            # Check file size
            file_size = csv_path.stat().st_size
            if not self._should_process_file(csv_path):
                return
            
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                logger.error(f"CSV file {csv_path} missing required columns: {missing_cols}")
                self.stats['skipped_files'] += 1
                return
            
            # Process records
            records = []
            skipped_rows = 0
            
            for idx, row in df.iterrows():
                try:
                    record = self._create_record_from_row(row, idx)
                    if record:
                        records.append(record)
                    else:
                        skipped_rows += 1
                except Exception as e:
                    logger.warning(f"Error processing row {idx} in {csv_path}: {e}")
                    skipped_rows += 1
            
            if records:
                # Create scan result
                relative_path = csv_path.name
                try:
                    # Try to get relative path if we're in a directory scan
                    if csv_path.parent != Path.cwd():
                        relative_path = str(csv_path.relative_to(csv_path.parent.parent))
                except:
                    relative_path = csv_path.name
                
                result = CSVScanResult(
                    path=csv_path,
                    size=file_size,
                    language='csv_image_prompts',
                    relative_path=relative_path,
                    csv_records=records,
                    total_prompts=len(records)
                )
                
                self._update_stats(result, skipped_rows)
                logger.info(f"Processed {len(records)} prompts from {csv_path} (skipped {skipped_rows} rows)")
                yield result
            else:
                logger.warning(f"No valid records found in {csv_path}")
                self.stats['skipped_files'] += 1
                
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_path}: {e}")
            self.stats['skipped_files'] += 1
    
    def _create_record_from_row(self, row: pd.Series, row_idx: int) -> Optional[CSVImageRecord]:
        """Create CSVImageRecord from pandas row"""
        # Check if prompt is valid
        prompt = str(row.get('prompt', '')).strip()
        if not prompt or prompt.lower() in ['nan', 'none', '']:
            if self.skip_empty_prompts:
                return None
            else:
                prompt = f"Default prompt for row {row_idx}"
        
        # Build record with optional fields
        record_data = {
            'prompt': prompt,
            'row_number': row_idx
        }
        
        # Add optional fields
        for col, default_value in self.OPTIONAL_COLUMNS.items():
            if col in row and pd.notna(row[col]):
                value = row[col]
                
                # Type conversion for specific fields
                if col == 'seed' and value:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        value = None
                elif col == 'priority' and value:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        value = row_idx  # Use row number as priority
                else:
                    value = str(value).strip() if value else default_value
                
                record_data[col] = value
            else:
                record_data[col] = default_value
        
        return CSVImageRecord(**record_data)
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if CSV file should be processed"""
        try:
            size = file_path.stat().st_size
            
            if size > self.max_file_size:
                logger.warning(f"Skipping {file_path}: size {size} exceeds limit {self.max_file_size}")
                return False
            
            if size < self.min_file_size:
                logger.warning(f"Skipping {file_path}: size {size} below minimum {self.min_file_size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False
    
    def _update_stats(self, result: CSVScanResult, skipped_rows: int):
        """Update scanning statistics"""
        self.stats['total_files_scanned'] += 1
        self.stats['total_prompts'] += result.total_prompts
        self.stats['total_size_bytes'] += result.size
        self.stats['skipped_rows'] += skipped_rows
        
        # Track by model
        for record in result.csv_records:
            model = record.model or 'unknown'
            self.stats['by_model'][model] = self.stats['by_model'].get(model, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanning statistics"""
        total_size_mb = self.stats['total_size_bytes'] / (1024 * 1024)
        
        return {
            'total_files_scanned': self.stats['total_files_scanned'],
            'total_prompts': self.stats['total_prompts'],
            'total_size_bytes': self.stats['total_size_bytes'],
            'total_size_mb': round(total_size_mb, 2),
            'by_model': self.stats['by_model'],
            'skipped_files': self.stats['skipped_files'],
            'skipped_rows': self.stats['skipped_rows'],
            'avg_prompts_per_file': round(self.stats['total_prompts'] / max(1, self.stats['total_files_scanned']), 2)
        }
    
    def validate_csv_format(self, csv_path: Path) -> Dict[str, Any]:
        """Validate CSV file format and return report"""
        report = {
            'valid': False,
            'file_path': str(csv_path),
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            if not csv_path.exists():
                report['issues'].append(f"File does not exist: {csv_path}")
                return report
            
            # Read file
            df = pd.read_csv(csv_path)
            
            # Check required columns
            missing_required = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_required:
                report['issues'].append(f"Missing required columns: {missing_required}")
            
            # Check for empty prompts
            if 'prompt' in df.columns:
                empty_prompts = df['prompt'].isna().sum() + (df['prompt'] == '').sum()
                if empty_prompts > 0:
                    report['warnings'].append(f"{empty_prompts} rows have empty prompts")
            
            # Statistics
            report['statistics'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'file_size_bytes': csv_path.stat().st_size,
                'estimated_prompts': len(df) - empty_prompts if 'prompt' in df.columns else 0
            }
            
            # Mark as valid if no critical issues
            if not report['issues']:
                report['valid'] = True
            
        except Exception as e:
            report['issues'].append(f"Error reading CSV file: {e}")
        
        return report