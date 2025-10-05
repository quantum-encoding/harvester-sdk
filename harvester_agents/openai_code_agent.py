"""
OpenAI Code Agent - Agentic coding assistant powered by OpenAI Agents SDK

Provides file operations and code execution capabilities using GPT-4o, o1, etc.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

# Import OpenAI Agents SDK components
try:
    from agents import Agent, Runner, function_tool, ModelSettings
except ImportError:
    raise ImportError(
        "OpenAI Agents SDK not installed. Install with: "
        "pip install 'harvester-sdk[computer]'"
    )


# File operation tools
@function_tool
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file, creating directories if needed.
    
    Args:
        filepath: Path to the file to write
        content: Content to write to the file
    
    Returns:
        Success message with file path
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"✓ Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"✗ Error writing to {filepath}: {str(e)}"


@function_tool
def read_file(filepath: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        filepath: Path to the file to read
    
    Returns:
        File contents or error message
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return f"✗ File not found: {filepath}"
        content = path.read_text()
        return f"File: {filepath}\n{'='*60}\n{content}"
    except Exception as e:
        return f"✗ Error reading {filepath}: {str(e)}"


@function_tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to list files from (default: current directory)
        pattern: Glob pattern to match files (default: all files)
    
    Returns:
        List of matching files
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"✗ Directory not found: {directory}"
        
        files = list(path.glob(pattern))
        if not files:
            return f"No files matching '{pattern}' in {directory}"
        
        file_list = "\n".join(f"  - {f.relative_to(path)}" for f in files if f.is_file())
        dir_list = "\n".join(f"  📁 {f.relative_to(path)}" for f in files if f.is_dir())
        
        result = f"Files in {directory} matching '{pattern}':\n"
        if file_list:
            result += f"\nFiles:\n{file_list}"
        if dir_list:
            result += f"\n\nDirectories:\n{dir_list}"
        
        return result
    except Exception as e:
        return f"✗ Error listing files in {directory}: {str(e)}"


@function_tool
def edit_file(filepath: str, old_text: str, new_text: str) -> str:
    """
    Edit a file by replacing old_text with new_text.
    
    Args:
        filepath: Path to the file to edit
        old_text: Text to find and replace
        new_text: Text to replace with
    
    Returns:
        Success message or error
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return f"✗ File not found: {filepath}"
        
        content = path.read_text()
        if old_text not in content:
            return f"✗ Text not found in {filepath}: '{old_text[:50]}...'"
        
        new_content = content.replace(old_text, new_text)
        path.write_text(new_content)
        
        return f"✓ Successfully edited {filepath} (replaced {len(old_text)} chars with {len(new_text)} chars)"
    except Exception as e:
        return f"✗ Error editing {filepath}: {str(e)}"


@function_tool
def execute_shell(command: str) -> str:
    """
    Execute a shell command and return the output.
    
    Args:
        command: Shell command to execute
    
    Returns:
        Command output or error
    """
    import subprocess
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = f"Command: {command}\n"
        output += f"Exit code: {result.returncode}\n"
        
        if result.stdout:
            output += f"\nOutput:\n{result.stdout}"
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        
        return output
    except subprocess.TimeoutExpired:
        return f"✗ Command timed out after 30 seconds: {command}"
    except Exception as e:
        return f"✗ Error executing command: {str(e)}"


class OpenAICodeAgent:
    """
    Agentic coding assistant powered by OpenAI Agents SDK.
    
    Features:
    - File reading/writing/editing
    - Shell command execution
    - Multi-step reasoning with o1/o3-mini
    - Automatic tool use and planning
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        max_iterations: int = 50
    ):
        """
        Initialize OpenAI Code Agent.
        
        Args:
            model: Model to use (gpt-4o, gpt-4o-mini, o1, o3-mini)
            temperature: Sampling temperature (None for default)
            max_iterations: Maximum agent iterations
        """
        self.model = model
        self.max_iterations = max_iterations
        
        # Build model settings
        model_settings = ModelSettings(temperature=temperature)
        
        # Create agent with file operation tools
        self.agent = Agent(
            name="OpenAI Code Agent",
            instructions="""You are an expert coding assistant. You can read, write, and edit files.

When asked to create or modify code:
1. Think step-by-step about what needs to be done
2. Use the provided tools to read existing files if needed
3. Write or edit files as requested
4. Verify your changes if possible
5. Provide a clear summary of what you did

Always use the tools to actually perform file operations - don't just describe what to do!""",
            model=model,
            model_settings=model_settings,
            tools=[
                write_file,
                read_file,
                list_files,
                edit_file,
                execute_shell,
            ],
        )
    
    async def execute_task(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a coding task.
        
        Args:
            description: Task description
            context: Optional context dict with files, project info, etc.
            show_progress: Whether to show progress during execution
        
        Returns:
            Dict with status, result, and messages
        """
        # Build context-enhanced prompt
        full_prompt = description
        
        if context:
            if 'files' in context:
                full_prompt += "\n\nContext files:\n"
                for filepath, content in context['files'].items():
                    full_prompt += f"\n{filepath}:\n```\n{content}\n```\n"
            
            if 'project_structure' in context:
                full_prompt += f"\n\nProject structure:\n{context['project_structure']}\n"
        
        if show_progress:
            print(f"🚀 Starting OpenAI Code Agent (model: {self.model})...")
            print()
        
        # Run the agent
        result = await Runner.run(self.agent, full_prompt)
        
        if show_progress:
            print()
            print("✓ Agent execution complete")
        
        return {
            'status': 'completed',
            'result': result.final_output,
            'messages': result.messages if hasattr(result, 'messages') else [],
        }
    
    def run_sync(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous version of execute_task.
        
        Args:
            description: Task description
            context: Optional context
            show_progress: Show progress
        
        Returns:
            Result dict
        """
        import asyncio
        return asyncio.run(self.execute_task(description, context, show_progress))
