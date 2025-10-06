#!/usr/bin/env python3
"""
Universal Converter - Convert any format to CSV for batch AI processing

This module provides a universal converter that can transform various input formats
(TXT, JSON, JSONL, XML, etc.) into CSV format suitable for batch processing.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import re

logger = logging.getLogger(__name__)


class UniversalConverter:
    """Universal format converter for batch AI processing"""
    
    def __init__(self):
        """Initialize the converter with supported formats"""
        self.supported_formats = {
            '.txt': self._convert_txt_to_csv,
            '.json': self._convert_json_to_csv,
            '.jsonl': self._convert_jsonl_to_csv,
            '.xml': self._convert_xml_to_csv,
            '.md': self._convert_markdown_to_csv,
            '.log': self._convert_log_to_csv,
            '.tsv': self._convert_tsv_to_csv,
        }
        logger.info("🔄 UniversalConverter initialized")
    
    async def convert_to_csv(
        self,
        input_path: str,
        output_path: str,
        template_name: Optional[str] = None,
        conversion_strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Convert any supported format to CSV
        
        Args:
            input_path: Path to input file
            output_path: Path for output CSV
            template_name: Optional template name for context
            conversion_strategy: Conversion strategy (auto, lines, records, etc.)
            
        Returns:
            Conversion report with statistics
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Detect format
        file_extension = input_file.suffix.lower()
        
        # If already CSV, just copy or validate
        if file_extension == '.csv':
            return await self._validate_csv(input_path, output_path)
        
        # Get appropriate converter
        converter_func = self.supported_formats.get(file_extension)
        
        if not converter_func:
            # Try to treat as text file
            converter_func = self._convert_txt_to_csv
            logger.warning(f"Unknown format {file_extension}, treating as text")
        
        # Convert
        try:
            result = await converter_func(input_path, output_path, conversion_strategy)
            result['template_name'] = template_name
            result['format'] = file_extension
            
            logger.info(f"✅ Converted {input_path} → {output_path}")
            logger.info(f"📊 {result.get('rows', 0)} rows created")
            
            return result
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise
    
    async def _validate_csv(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Validate and copy CSV file"""
        try:
            df = pd.read_csv(input_path)
            
            # Ensure required columns exist
            if 'prompt' not in df.columns and 'request' not in df.columns:
                # Add prompt column if missing
                if len(df.columns) > 0:
                    df['prompt'] = df[df.columns[0]]
            
            # Save to output
            df.to_csv(output_path, index=False)
            
            return {
                'status': 'success',
                'rows': len(df),
                'columns': list(df.columns),
                'message': 'CSV validated and copied'
            }
        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            raise
    
    async def _convert_txt_to_csv(
        self, 
        input_path: str, 
        output_path: str,
        strategy: str = "lines"
    ) -> Dict[str, Any]:
        """Convert text file to CSV"""
        
        rows = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if strategy == "lines":
            # Each line becomes a row
            lines = content.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    rows.append({
                        'id': i + 1,
                        'prompt': line.strip(),
                        'source': Path(input_path).name
                    })
        
        elif strategy == "paragraphs":
            # Each paragraph becomes a row
            paragraphs = re.split(r'\n\s*\n', content)
            for i, para in enumerate(paragraphs):
                if para.strip():
                    rows.append({
                        'id': i + 1,
                        'prompt': para.strip(),
                        'source': Path(input_path).name
                    })
        
        else:  # strategy == "auto" or "whole"
            # Whole file as single row or smart detection
            if len(content) < 1000 or '\n' not in content:
                # Small file or single line - one row
                rows.append({
                    'id': 1,
                    'prompt': content.strip(),
                    'source': Path(input_path).name
                })
            else:
                # Use lines strategy for larger files
                return await self._convert_txt_to_csv(input_path, output_path, "lines")
        
        # Write CSV
        if rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'source'])
                writer.writeheader()
                writer.writerows(rows)
        
        return {
            'status': 'success',
            'rows': len(rows),
            'strategy': strategy
        }
    
    async def _convert_json_to_csv(
        self,
        input_path: str,
        output_path: str,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Convert JSON file to CSV"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rows = []
        
        if isinstance(data, list):
            # Array of objects
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # Use existing structure
                    if 'prompt' not in item and 'request' not in item:
                        # Try to find a text field
                        text_fields = ['text', 'content', 'message', 'query', 'question']
                        for field in text_fields:
                            if field in item:
                                item['prompt'] = item[field]
                                break
                        else:
                            # Use first string value
                            for value in item.values():
                                if isinstance(value, str):
                                    item['prompt'] = value
                                    break
                    
                    if 'id' not in item:
                        item['id'] = i + 1
                    
                    rows.append(item)
                
                elif isinstance(item, str):
                    # Array of strings
                    rows.append({
                        'id': i + 1,
                        'prompt': item,
                        'source': Path(input_path).name
                    })
        
        elif isinstance(data, dict):
            # Single object or nested structure
            if 'prompts' in data and isinstance(data['prompts'], list):
                # Has prompts array
                for i, prompt in enumerate(data['prompts']):
                    rows.append({
                        'id': i + 1,
                        'prompt': prompt if isinstance(prompt, str) else json.dumps(prompt),
                        'source': Path(input_path).name
                    })
            
            elif 'data' in data and isinstance(data['data'], list):
                # Has data array
                return await self._convert_json_to_csv(input_path, output_path, strategy)
            
            else:
                # Single object - convert to single row
                if 'prompt' not in data and 'request' not in data:
                    # Find text content
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 10:
                            data['prompt'] = value
                            break
                    else:
                        data['prompt'] = json.dumps(data)
                
                if 'id' not in data:
                    data['id'] = 1
                
                rows.append(data)
        
        # Convert to DataFrame and save
        if rows:
            df = pd.DataFrame(rows)
            
            # Ensure prompt column exists
            if 'prompt' not in df.columns and 'request' in df.columns:
                df['prompt'] = df['request']
            elif 'prompt' not in df.columns:
                df['prompt'] = df.apply(lambda x: json.dumps(x.to_dict()), axis=1)
            
            df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'rows': len(rows),
            'strategy': strategy
        }
    
    async def _convert_jsonl_to_csv(
        self,
        input_path: str,
        output_path: str,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Convert JSONL (JSON Lines) file to CSV"""
        
        rows = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line)
                        
                        if isinstance(item, dict):
                            if 'id' not in item:
                                item['id'] = i + 1
                            
                            if 'prompt' not in item and 'request' not in item:
                                # Find text field
                                for field in ['text', 'content', 'message', 'query']:
                                    if field in item:
                                        item['prompt'] = item[field]
                                        break
                                else:
                                    item['prompt'] = json.dumps(item)
                            
                            rows.append(item)
                        
                        elif isinstance(item, str):
                            rows.append({
                                'id': i + 1,
                                'prompt': item,
                                'source': Path(input_path).name
                            })
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {i+1}: {e}")
        
        # Convert to DataFrame and save
        if rows:
            df = pd.DataFrame(rows)
            
            # Ensure prompt column
            if 'prompt' not in df.columns and 'request' in df.columns:
                df['prompt'] = df['request']
            
            df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'rows': len(rows),
            'strategy': strategy
        }
    
    async def _convert_xml_to_csv(
        self,
        input_path: str,
        output_path: str,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Convert XML file to CSV"""
        
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            rows = []
            
            # Try to find repeating elements
            for i, element in enumerate(root):
                row = {
                    'id': i + 1,
                    'source': Path(input_path).name
                }
                
                # Extract text from element
                if element.text:
                    row['prompt'] = element.text.strip()
                else:
                    # Get all text from sub-elements
                    texts = []
                    for sub in element.iter():
                        if sub.text:
                            texts.append(sub.text.strip())
                    row['prompt'] = ' '.join(texts)
                
                # Add attributes
                for key, value in element.attrib.items():
                    row[key] = value
                
                if row.get('prompt'):
                    rows.append(row)
            
            # If no rows found, treat whole XML as text
            if not rows:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                rows.append({
                    'id': 1,
                    'prompt': content,
                    'source': Path(input_path).name
                })
            
            # Save to CSV
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
            
            return {
                'status': 'success',
                'rows': len(rows),
                'strategy': strategy
            }
            
        except Exception as e:
            logger.error(f"XML conversion failed: {e}")
            # Fallback to text conversion
            return await self._convert_txt_to_csv(input_path, output_path, strategy)
    
    async def _convert_markdown_to_csv(
        self,
        input_path: str,
        output_path: str,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Convert Markdown file to CSV"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        rows = []
        
        if strategy == "sections":
            # Split by headers
            sections = re.split(r'^#+\s+', content, flags=re.MULTILINE)
            
            for i, section in enumerate(sections):
                if section.strip():
                    # First line might be header title
                    lines = section.strip().split('\n', 1)
                    if len(lines) == 2:
                        title, body = lines
                        rows.append({
                            'id': i + 1,
                            'title': title.strip(),
                            'prompt': body.strip(),
                            'source': Path(input_path).name
                        })
                    else:
                        rows.append({
                            'id': i + 1,
                            'prompt': section.strip(),
                            'source': Path(input_path).name
                        })
        
        elif strategy == "code_blocks":
            # Extract code blocks
            code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', content, re.DOTALL)
            
            for i, block in enumerate(code_blocks):
                rows.append({
                    'id': i + 1,
                    'prompt': block.strip(),
                    'type': 'code',
                    'source': Path(input_path).name
                })
        
        else:  # auto or paragraphs
            # Use paragraph splitting
            return await self._convert_txt_to_csv(input_path, output_path, "paragraphs")
        
        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'rows': len(rows),
            'strategy': strategy
        }
    
    async def _convert_log_to_csv(
        self,
        input_path: str,
        output_path: str,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Convert log file to CSV"""
        
        rows = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    # Try to parse log format
                    # Common format: [TIMESTAMP] LEVEL MESSAGE
                    match = re.match(r'^\[(.*?)\]\s*(\w+)\s*(.*)$', line)
                    
                    if match:
                        timestamp, level, message = match.groups()
                        rows.append({
                            'id': i + 1,
                            'timestamp': timestamp,
                            'level': level,
                            'prompt': message.strip(),
                            'source': Path(input_path).name
                        })
                    else:
                        # Plain line
                        rows.append({
                            'id': i + 1,
                            'prompt': line.strip(),
                            'source': Path(input_path).name
                        })
        
        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'rows': len(rows),
            'strategy': strategy
        }
    
    async def _convert_tsv_to_csv(
        self,
        input_path: str,
        output_path: str,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Convert TSV file to CSV"""
        
        try:
            # Read TSV
            df = pd.read_csv(input_path, sep='\t')
            
            # Ensure prompt column
            if 'prompt' not in df.columns:
                # Use first text column
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df['prompt'] = df[col]
                        break
                else:
                    df['prompt'] = df[df.columns[0]]
            
            # Save as CSV
            df.to_csv(output_path, index=False)
            
            return {
                'status': 'success',
                'rows': len(df),
                'strategy': strategy
            }
            
        except Exception as e:
            logger.error(f"TSV conversion failed: {e}")
            raise


# Quick test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        converter = UniversalConverter()
        
        # Test with a simple text file
        test_content = "What is 2 + 2?\nExplain quantum computing\nWrite a poem about AI"
        
        with open('test_input.txt', 'w') as f:
            f.write(test_content)
        
        result = await converter.convert_to_csv('test_input.txt', 'test_output.csv')
        print(f"Conversion result: {result}")
        
        # Check output
        import pandas as pd
        df = pd.read_csv('test_output.csv')
        print(f"Output CSV:\n{df}")
    
    asyncio.run(test())