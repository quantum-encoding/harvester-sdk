#!/usr/bin/env python3
"""
Sovereign Directory Architecture - Universal Output Path Management

Provides unified, professional directory structure across all CLI tools
for the Sacred Wrapper SDK ecosystem.
"""

import re
from pathlib import Path
from datetime import datetime

def generate_cli_output_directory(script_name: str, source_input: str) -> Path:
    """
    Generate sovereign output directory following the Divine Reorganization pattern.
    
    Pattern: ~/harvester-sdk/{script_name}/{clean_source}_{timestamp}/
    
    Args:
        script_name: Name of the CLI tool (batch_code, image_cli, ai_assistant, etc.)
        source_input: Source input (directory path, prompt snippet, etc.)
    
    Returns:
        Path: Professional output directory path
    """
    # Get user home directory
    home_dir = Path.home()
    base_dir = "harvester-sdk"
    
    # Clean source input for safe directory naming
    clean_source = clean_for_directory_name(source_input)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create job name
    job_name = f"{clean_source}_{timestamp}"
    
    # Construct sovereign path
    output_dir = home_dir / base_dir / script_name / job_name
    
    return output_dir

def clean_for_directory_name(input_string: str) -> str:
    """
    Clean input string to create safe directory names.
    
    Args:
        input_string: Raw input string (path, prompt, etc.)
    
    Returns:
        str: Cleaned string safe for directory names
    """
    # Extract meaningful part from path or use first 30 chars
    if '/' in input_string or '\\' in input_string:
        # It's a path - use the last directory name
        path_obj = Path(input_string)
        meaningful = path_obj.name or path_obj.parent.name
    else:
        # It's a prompt or other text - use first 30 chars
        meaningful = input_string[:30]
    
    # Clean the string
    # Replace spaces and special chars with underscores
    cleaned = re.sub(r'[^\w\-_]', '_', meaningful)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # Ensure it's not empty
    if not cleaned:
        cleaned = "output"
    
    # Limit length
    if len(cleaned) > 30:
        cleaned = cleaned[:30].rstrip('_')
    
    return cleaned

def ensure_directory_exists(directory_path: Path) -> None:
    """
    Ensure the directory exists, creating parent directories as needed.
    
    Args:
        directory_path: Path to ensure exists
    """
    directory_path.mkdir(parents=True, exist_ok=True)

def get_sovereign_base_directory() -> Path:
    """
    Get the base sovereign directory path.
    
    Returns:
        Path: ~/harvester-sdk/
    """
    return Path.home() / "harvester-sdk"