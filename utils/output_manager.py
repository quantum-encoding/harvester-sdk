#!/usr/bin/env python3
"""
Output Manager for Summon AI Orchestration System

Creates intuitive, timestamped output structure:
./outputs/
  ├── 2025-06-17_19-30-00_maths_vtx-2/
  │   ├── response.md           # Main human-readable output
  │   ├── response.json         # Complete metadata
  │   ├── response.py           # Code content (if applicable)
  │   └── prompt.txt            # Original prompt
  └── manifests/
      └── 2025-06-17/           # JSON manifests organized by date
          └── session_manifest.json

Simple, clean, timestamped. No crazy staging folders.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class OutputManager:
    """Manages intuitive output saving for AI orchestration responses"""
    
    def __init__(self, base_output_dir: Optional[Path] = None):
        # Use the proper SDK output structure: ~/harvest_sdk/
        self.base_dir = base_output_dir or Path.home() / 'harvest_sdk'
        self.manifests_dir = self.base_dir / 'manifests'
        
        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✨ Output manager initialized: {self.base_dir}")
    
    def save_response(
        self,
        result: Dict[str, Any],
        template_name: str,
        model_used: str,
        auto_save: bool = True
    ) -> Dict[str, Path]:
        """
        Save AI response in multiple formats with intelligent detection
        
        Args:
            result: The complete response result from summon
            template_name: Template used (maths, code_assist, etc.)
            model_used: Model that generated the response
            auto_save: Whether to auto-save (default True)
            
        Returns:
            Dictionary of saved file paths by format
        """
        
        if not auto_save:
            return {}
            
        # Create organized output directory structure
        # Format: ~/harvest_sdk/[image|text]/provider_name/model_folder_timestamp/
        content_type = "text"  # Default to text, could be enhanced to detect images
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Extract provider name from model if possible
        provider_name = self._extract_provider_from_model(model_used)
        model_folder = f"{model_used}-{timestamp}"
        
        session_dir = self.base_dir / content_type / provider_name / model_folder
        session_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        response_content = result.get('response', '')
        
        # 1. Always save JSON manifest
        json_path = session_dir / 'response.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        saved_files['json'] = json_path
        
        # 2. Always save original prompt
        prompt_path = session_dir / 'prompt.txt'
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(result.get('original_prompt', ''))
        saved_files['prompt'] = prompt_path
        
        # 3. Save markdown version (human-readable)
        md_path = session_dir / 'response.md'
        md_content = self._format_as_markdown(result)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        saved_files['markdown'] = md_path
        
        # 4. Smart content-based saving
        content_type = self._detect_content_type(response_content, template_name)
        
        if content_type == 'code':
            # Save as appropriate code file
            ext = self._detect_code_language(response_content, template_name)
            code_path = session_dir / f'response.{ext}'
            code_content = self._extract_code_content(response_content)
            
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            saved_files['code'] = code_path
            
        elif content_type == 'mixed':
            # For mixed content, also save as HTML for rich formatting
            html_path = session_dir / 'response.html'
            html_content = self._format_as_html(result)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            saved_files['html'] = html_path
        
        # 5. Update manifest
        self._update_manifest(result, session_dir, saved_files)
        
        logger.info(f"💾 Saved response: {session_dir.name}")
        logger.info(f"📁 Files: {', '.join(saved_files.keys())}")
        
        return saved_files
    
    def _detect_content_type(self, content: str, template_name: str) -> str:
        """Detect if content is code, mixed, or general"""
        
        # Template-based detection
        code_templates = {'code_assist', 'code_review', 'debug', 'architecture'}
        if template_name in code_templates:
            return 'code'
        
        # Content-based detection
        code_indicators = [
            '```', 'def ', 'function ', 'class ', 'import ', 'from ',
            '{', '}', '()', '=>', 'const ', 'let ', 'var ', 'async ',
            'return ', 'print(', 'console.log', '<script', '</script>'
        ]
        
        code_count = sum(1 for indicator in code_indicators if indicator in content)
        
        if code_count >= 3:
            return 'code'
        elif code_count >= 1:
            return 'mixed'
        else:
            return 'general'
    
    def _detect_code_language(self, content: str, template_name: str) -> str:
        """Detect programming language and return appropriate file extension"""
        
        # Language indicators
        patterns = {
            'py': ['def ', 'import ', 'from ', 'print(', '__init__', 'self.'],
            'js': ['function ', 'const ', 'let ', 'var ', '=>', 'console.log'],
            'ts': ['interface ', 'type ', ': string', ': number', ': boolean'],
            'html': ['<html', '<div', '<script', '<!DOCTYPE'],
            'css': ['{', '}', 'color:', 'margin:', 'padding:'],
            'sql': ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE'],
            'sh': ['#!/bin/bash', 'echo ', '$1', '${'],
            'yaml': ['---', 'name:', 'version:', '  - '],
            'json': ['{"', '":', '"}']
        }
        
        scores = {}
        for ext, indicators in patterns.items():
            scores[ext] = sum(1 for indicator in indicators if indicator in content)
        
        # Get highest scoring language
        best_lang = max(scores, key=scores.get)
        
        # Default fallbacks based on template
        if scores[best_lang] == 0:
            if 'architecture' in template_name:
                return 'md'  # Architecture as markdown
            return 'txt'  # General text
            
        return best_lang
    
    def _extract_provider_from_model(self, model_used: str) -> str:
        """Extract provider name from model alias or name"""
        # Map common model prefixes/names to providers
        if model_used.startswith('gpt') or 'openai' in model_used.lower():
            return 'openai'
        elif model_used.startswith('claude') or model_used.startswith('ant-') or 'anthropic' in model_used.lower():
            return 'anthropic'
        elif model_used.startswith('gemini') or model_used.startswith('goo-') or 'google' in model_used.lower():
            return 'google'
        elif model_used.startswith('grok') or 'xai' in model_used.lower():
            return 'xai'
        elif 'deepseek' in model_used.lower():
            return 'deepseek'
        elif 'vertex' in model_used.lower():
            return 'vertex'
        else:
            return 'unknown'
    
    def _extract_code_content(self, content: str) -> str:
        """Extract code from markdown code blocks or return as-is"""
        
        # Extract from markdown code blocks
        code_block_pattern = r'```(?:\w+)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, content, re.DOTALL)
        
        if matches:
            # Return all code blocks concatenated
            return '\n\n'.join(matches)
        
        # If no code blocks, return content as-is (might be inline code)
        return content
    
    def _format_as_markdown(self, result: Dict[str, Any]) -> str:
        """Format result as clean markdown"""
        
        template_name = result.get('template_used', 'Unknown')
        model = result.get('model_used', 'Unknown')
        prompt = result.get('original_prompt', '')
        response = result.get('response', '')
        processing_time = result.get('processing_time', 0)
        timestamp = result.get('timestamp', datetime.now().isoformat())
        
        md_content = f"""# {template_name} Response

**Model:** {model}  
**Generated:** {timestamp}  
**Processing Time:** {processing_time:.1f}s  

## Original Prompt

{prompt}

## Response

{response}

---
*Generated by AI Orchestration System*
"""
        return md_content
    
    def _format_as_html(self, result: Dict[str, Any]) -> str:
        """Format result as HTML with styling"""
        
        template_name = result.get('template_used', 'Unknown')
        model = result.get('model_used', 'Unknown')
        prompt = result.get('original_prompt', '')
        response = result.get('response', '')
        processing_time = result.get('processing_time', 0)
        timestamp = result.get('timestamp', datetime.now().isoformat())
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{template_name} Response</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .prompt {{ background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0; }}
        .response {{ background: #f5f5f5; padding: 20px; border-radius: 6px; }}
        pre {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .meta {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{template_name} Response</h1>
        <div class="meta">
            <strong>Model:</strong> {model}<br>
            <strong>Generated:</strong> {timestamp}<br>
            <strong>Processing Time:</strong> {processing_time:.1f}s
        </div>
    </div>
    
    <div class="prompt">
        <h2>Original Prompt</h2>
        <p>{prompt}</p>
    </div>
    
    <div class="response">
        <h2>Response</h2>
        <pre>{response}</pre>
    </div>
    
    <footer class="meta">
        <p><em>Generated by AI Orchestration System</em></p>
    </footer>
</body>
</html>"""
        return html_content
    
    def _update_manifest(
        self,
        result: Dict[str, Any],
        session_dir: Path,
        saved_files: Dict[str, Path]
    ):
        """Update daily manifest with session info"""
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        daily_dir = self.manifests_dir / date_str
        daily_dir.mkdir(exist_ok=True)
        
        manifest_file = daily_dir / 'session_manifest.json'
        
        # Load existing manifest or create new
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                'date': date_str,
                'sessions': []
            }
        
        # Add current session
        session_info = {
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'session_id': session_dir.name,
            'template': result.get('template_used'),
            'model': result.get('model_used'),
            'council_mode': result.get('council_mode', False),
            'processing_time': result.get('processing_time'),
            'prompt_length': len(result.get('original_prompt', '')),
            'response_length': len(result.get('response', '')),
            'files_saved': [str(path.relative_to(self.base_dir)) for path in saved_files.values()],
            'output_directory': str(session_dir.relative_to(self.base_dir))
        }
        
        manifest['sessions'].append(session_info)
        
        # Save updated manifest
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def get_recent_sessions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent sessions from manifests"""
        
        sessions = []
        
        for i in range(days):
            date = datetime.now() - datetime.timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            manifest_file = self.manifests_dir / date_str / 'session_manifest.json'
            
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                    sessions.extend(manifest.get('sessions', []))
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def cleanup_old_outputs(self, keep_days: int = 30):
        """Clean up old output directories"""
        
        cutoff_date = datetime.now() - datetime.timedelta(days=keep_days)
        
        for session_dir in self.base_dir.iterdir():
            if session_dir.is_dir() and session_dir.name != 'manifests':
                # Parse timestamp from directory name
                try:
                    timestamp_str = session_dir.name.split('_')[0] + '_' + session_dir.name.split('_')[1]
                    session_date = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                    
                    if session_date < cutoff_date:
                        import shutil
                        shutil.rmtree(session_dir)
                        logger.info(f"🗑️ Cleaned up old session: {session_dir.name}")
                        
                except (ValueError, IndexError):
                    # Skip directories that don't match our naming pattern
                    continue


def create_output_manager(base_dir: Optional[Path] = None) -> OutputManager:
    """Factory function to create output manager"""
    return OutputManager(base_dir)