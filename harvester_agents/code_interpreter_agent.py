"""
Code Interpreter Agent - Python code execution in sandboxed containers

Allows models to write and run Python code to solve complex problems in:
- Data analysis and visualization
- Mathematical computations
- File processing and transformation
- Iterative problem solving
"""

from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

# Import OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI SDK not installed. Install with: "
        "pip install 'harvester-sdk[computer]'"
    )


class CodeInterpreterAgent:
    """
    Agent that can write and run Python code in sandboxed containers.

    Features:
    - Sandboxed Python execution
    - File upload and download
    - Data analysis and visualization
    - Iterative problem solving
    - Automatic container management
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        container_mode: Literal["auto", "explicit"] = "auto",
        container_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None
    ):
        """
        Initialize Code Interpreter Agent.

        Args:
            model: Model to use (gpt-4.1, gpt-5, o3, o4-mini)
            container_mode: Container creation mode (auto or explicit)
            container_id: Explicit container ID (required if mode=explicit)
            file_ids: List of file IDs to include in container
        """
        self.client = OpenAI()
        self.model = model
        self.container_mode = container_mode
        self.container_id = container_id
        self.file_ids = file_ids or []

        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable")

        # Create explicit container if needed
        if container_mode == "explicit" and not container_id:
            container = self.client.containers.create(name="code-interpreter-container")
            self.container_id = container.id
            logger.info(f"Created container: {self.container_id}")

    def execute_task(
        self,
        task: str,
        instructions: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a task using Code Interpreter.

        Args:
            task: Task description
            instructions: Optional system instructions
            show_progress: Whether to show progress during execution

        Returns:
            Dict with status, result, container_id, and generated files
        """
        # Build system instructions
        if instructions is None:
            instructions = """You are a helpful coding assistant with Python execution capabilities.

When asked to solve problems:
1. Write Python code to solve the problem
2. Run the code and verify results
3. If code fails, debug and retry
4. Create visualizations or files as needed
5. Provide clear explanations of your approach

Always use the python tool to execute code and solve problems programmatically."""

        if show_progress:
            print(f"🐍 Starting Code Interpreter Agent (model: {self.model})...")
            print()

        # Configure tool
        if self.container_mode == "auto":
            tool_config = {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "file_ids": self.file_ids
                }
            }
        else:
            tool_config = {
                "type": "code_interpreter",
                "container": self.container_id
            }

        # Create response
        response = self.client.responses.create(
            model=self.model,
            tools=[tool_config],
            instructions=instructions,
            input=task
        )

        # Extract results
        result_text = ""
        generated_files = []
        used_container_id = self.container_id
        code_calls = []

        for item in response.output:
            if item.type == "code_interpreter_call":
                # Track container ID from first code call
                if hasattr(item, 'container_id') and not used_container_id:
                    used_container_id = item.container_id

                if show_progress and hasattr(item, 'code'):
                    print(f"{'='*60}")
                    print("🔧 Executing Python Code:")
                    print(f"{'='*60}")
                    print(item.code)
                    print()

                code_calls.append({
                    'code': item.code if hasattr(item, 'code') else None,
                    'call_id': item.call_id if hasattr(item, 'call_id') else None
                })

            elif item.type == "code_interpreter_call_output":
                if show_progress:
                    print(f"{'='*60}")
                    print("📤 Output:")
                    print(f"{'='*60}")
                    if hasattr(item, 'logs'):
                        print(item.logs)
                    print()

            elif item.type == "message":
                if hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'text'):
                            result_text = content.text
                            if show_progress:
                                print(f"📝 Result: {content.text}")

                        # Extract file citations
                        if hasattr(content, 'annotations'):
                            for annotation in content.annotations:
                                if annotation.type == "container_file_citation":
                                    generated_files.append({
                                        'file_id': annotation.file_id,
                                        'container_id': annotation.container_id,
                                        'filename': annotation.filename
                                    })

        if show_progress:
            print()
            print("✓ Code Interpreter execution complete")
            if generated_files:
                print(f"📁 Generated {len(generated_files)} file(s)")

        return {
            'status': 'completed',
            'result': result_text,
            'container_id': used_container_id,
            'generated_files': generated_files,
            'code_calls': code_calls
        }

    def upload_file(self, filepath: str) -> str:
        """
        Upload a file to the container.

        Args:
            filepath: Path to file to upload

        Returns:
            File ID of uploaded file
        """
        if not self.container_id:
            raise ValueError("Container ID required. Use explicit mode or run a task first.")

        with open(filepath, 'rb') as f:
            file_obj = self.client.containers.files.create(
                container_id=self.container_id,
                file=f
            )

        logger.info(f"Uploaded {filepath} as {file_obj.id}")
        return file_obj.id

    def download_file(self, file_id: str, output_path: str):
        """
        Download a file from the container.

        Args:
            file_id: File ID to download
            output_path: Path to save downloaded file
        """
        if not self.container_id:
            raise ValueError("Container ID required. Use explicit mode or run a task first.")

        content = self.client.containers.files.content(
            container_id=self.container_id,
            file_id=file_id
        )

        Path(output_path).write_bytes(content.read())
        logger.info(f"Downloaded {file_id} to {output_path}")

    def list_files(self) -> List[Dict[str, str]]:
        """
        List all files in the container.

        Returns:
            List of file metadata dicts
        """
        if not self.container_id:
            raise ValueError("Container ID required. Use explicit mode or run a task first.")

        files = self.client.containers.files.list(container_id=self.container_id)
        return [
            {
                'id': f.id,
                'filename': f.filename,
                'bytes': f.bytes
            }
            for f in files.data
        ]

    def cleanup(self):
        """Delete the container and all its files."""
        if self.container_id and self.container_mode == "explicit":
            try:
                self.client.containers.delete(self.container_id)
                logger.info(f"Deleted container: {self.container_id}")
            except Exception as e:
                logger.warning(f"Failed to delete container: {e}")
