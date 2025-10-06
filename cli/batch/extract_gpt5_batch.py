#!/usr/bin/env python3
"""
GPT-5 Batch Results Extractor
Extract text responses from GPT-5 batch results with support for multiple output formats.

© 2025 Quantum Encoding Ltd
"""
import json
import click
from pathlib import Path
from datetime import datetime
import logging
import sys

# Add parent SDK directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def extract_gpt5_response(json_response_str: str) -> str:
    """
    Extract the text response from a GPT-5 JSON response.

    GPT-5 response structure:
    {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "THE ACTUAL RESPONSE TEXT HERE"
                    }
                ]
            }
        ]
    }

    Args:
        json_response_str: JSON string from GPT-5 batch result

    Returns:
        Extracted text or error message
    """
    try:
        # Parse the JSON string
        response_data = json.loads(json_response_str)

        # Navigate to the text field
        # response_data -> output -> [0] -> content -> [0] -> text
        if 'output' in response_data and isinstance(response_data['output'], list):
            for output_item in response_data['output']:
                if output_item.get('type') == 'message' and 'content' in output_item:
                    content_list = output_item['content']
                    if isinstance(content_list, list):
                        for content_item in content_list:
                            if content_item.get('type') == 'output_text' and 'text' in content_item:
                                # Found it! Return the text
                                return content_item['text']

        # If we couldn't find the expected structure, return error message
        return f"ERROR: Could not extract text from GPT-5 response structure"

    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON - {e}"
    except Exception as e:
        return f"ERROR: {e}"


def extract_code_blocks(content: str, language: str = None) -> str:
    """
    Extract code from markdown code blocks.

    Args:
        content: Text content potentially containing code blocks
        language: Optional language filter

    Returns:
        Extracted code or full content if no code blocks found
    """
    if "```" not in content:
        return content

    lines = content.split('\n')
    in_code = False
    code_lines = []

    for line in lines:
        if line.startswith("```") and not in_code:
            in_code = True
            continue
        elif line.startswith("```") and in_code:
            in_code = False
            continue
        elif in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines)
    else:
        return content


@click.command()
@click.argument('json_file', type=click.Path(exists=True))
@click.option('--output', '-o', default=None,
              help='Output directory (default: extracted_TIMESTAMP next to JSON file)')
@click.option('--keep-structure', is_flag=True,
              help='Maintain original directory structure from custom IDs')
@click.option('--extract-code', is_flag=True,
              help='Extract code blocks from markdown responses')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'auto']), default='auto',
              help='Output format: text (plain text), json (preserve JSON), auto (detect)')
def main(json_file, output, keep_structure, extract_code, format):
    """
    Extract GPT-5 batch results to individual files.

    This tool works with GPT-5 batch results from the OpenAI Batch API.
    It extracts the raw text response from each result.

    Examples:
        # Extract all results to text files
        extract-gpt5-batch batch_results.json

        # Maintain directory structure from custom IDs
        extract-gpt5-batch batch_results.json --keep-structure

        # Extract code blocks only (for code generation tasks)
        extract-gpt5-batch batch_results.json --extract-code

        # Custom output directory
        extract-gpt5-batch batch_results.json -o ./extracted_files
    """

    json_path = Path(json_file)

    # Load the batch results JSON
    logger.info(f"📂 Loading: {json_path}")
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Handle different result formats
    if isinstance(results, list):
        # Format: [{"custom_id": "...", "response": {...}}, ...]
        results_dict = {item.get('custom_id', f'result_{i}'): item.get('response', {})
                       for i, item in enumerate(results)}
    elif isinstance(results, dict):
        # Format: {"custom_id_1": {...}, "custom_id_2": {...}}
        results_dict = results
    else:
        logger.error("❌ Unsupported result format")
        return 1

    logger.info(f"📦 Found {len(results_dict)} responses to extract")

    # Determine output directory
    if output:
        output_dir = Path(output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = json_path.parent / f"gpt5_extracted_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")

    # Process each result
    success_count = 0
    error_count = 0

    for custom_id, response_data in results_dict.items():
        try:
            # Convert response to JSON string if needed
            if isinstance(response_data, dict):
                json_response_str = json.dumps(response_data)
            else:
                json_response_str = str(response_data)

            # Extract the text from GPT-5 response
            extracted_text = extract_gpt5_response(json_response_str)

            # Check if extraction failed
            if extracted_text.startswith("ERROR:"):
                logger.error(f"  ❌ {custom_id}: {extracted_text}")
                error_count += 1
                continue

            # Optionally extract code blocks
            if extract_code:
                extracted_text = extract_code_blocks(extracted_text)

            # Convert custom_id to file path
            # e.g., "globals.css" -> "globals.css"
            # e.g., "solar_page.tsx" -> "solar/page.tsx"
            file_path = custom_id.replace('_', '/')

            # Determine output path
            if keep_structure:
                output_path = output_dir / file_path
            else:
                # Flatten - just use the filename
                output_path = output_dir / Path(file_path).name

            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the extracted text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            logger.info(f"  ✅ {custom_id} -> {output_path.relative_to(output_dir)}")
            success_count += 1

        except Exception as e:
            logger.error(f"  ❌ {custom_id}: Unexpected error - {e}")
            error_count += 1

    # Summary
    logger.info(f"\n📊 Extraction Summary:")
    logger.info(f"   ✅ Success: {success_count} files")
    if error_count > 0:
        logger.info(f"   ❌ Errors: {error_count} files")

    if success_count > 0:
        logger.info(f"\n✨ Files extracted to: {output_dir}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
