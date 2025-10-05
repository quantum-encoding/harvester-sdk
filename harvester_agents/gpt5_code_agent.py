"""
GPT-5 Code Agent - Advanced reasoning agent powered by GPT-5

Provides file operations and code execution with configurable reasoning effort
and verbosity control for optimal performance on coding and agentic tasks.
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


# Custom tools for GPT-5
WRITE_FILE_TOOL = {
    "type": "custom",
    "name": "write_file",
    "description": """Write content to a file, creating directories if needed.

Usage: write_file <filepath> <content>
Example: write_file src/main.py print("Hello World")"""
}

READ_FILE_TOOL = {
    "type": "custom",
    "name": "read_file",
    "description": """Read the contents of a file.

Usage: read_file <filepath>
Example: read_file src/main.py"""
}

LIST_FILES_TOOL = {
    "type": "custom",
    "name": "list_files",
    "description": """List files in a directory matching a pattern.

Usage: list_files <directory> [pattern]
Example: list_files src/ *.py"""
}

EDIT_FILE_TOOL = {
    "type": "custom",
    "name": "edit_file",
    "description": """Edit a file by replacing old_text with new_text.

Usage: edit_file <filepath>
OLD_TEXT:
<text to replace>
NEW_TEXT:
<replacement text>"""
}

EXECUTE_SHELL_TOOL = {
    "type": "custom",
    "name": "execute_shell",
    "description": """Execute a shell command and return the output.

Usage: execute_shell <command>
Example: execute_shell ls -la"""
}


class GPT5CodeAgent:
    """
    Advanced coding assistant powered by GPT-5 with reasoning capabilities.

    Features:
    - File reading/writing/editing
    - Shell command execution
    - Configurable reasoning effort (minimal, low, medium, high)
    - Verbosity control (low, medium, high)
    - Custom tools with freeform text inputs
    - Preambles for tool-calling transparency
    """

    def __init__(
        self,
        model: str = "gpt-5",
        reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium",
        verbosity: Literal["low", "medium", "high"] = "medium",
        max_iterations: int = 50
    ):
        """
        Initialize GPT-5 Code Agent.

        Args:
            model: Model to use (gpt-5, gpt-5-mini, gpt-5-nano)
            reasoning_effort: Reasoning depth (minimal=fastest, high=most thorough)
            verbosity: Output length control (low=concise, high=detailed)
            max_iterations: Maximum agent iterations
        """
        self.client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.max_iterations = max_iterations

        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable")

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool call and return the result."""
        try:
            if tool_name == "write_file":
                return self._write_file(tool_input)
            elif tool_name == "read_file":
                return self._read_file(tool_input)
            elif tool_name == "list_files":
                return self._list_files(tool_input)
            elif tool_name == "edit_file":
                return self._edit_file(tool_input)
            elif tool_name == "execute_shell":
                return self._execute_shell(tool_input)
            else:
                return f"✗ Unknown tool: {tool_name}"
        except Exception as e:
            return f"✗ Error executing {tool_name}: {str(e)}"

    def _write_file(self, input_text: str) -> str:
        """Parse and execute write_file command."""
        lines = input_text.strip().split('\n', 1)
        if len(lines) < 2:
            return "✗ Invalid write_file format. Expected: filepath on first line, content on remaining lines"

        filepath = lines[0].strip()
        content = lines[1] if len(lines) > 1 else ""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"✓ Successfully wrote {len(content)} characters to {filepath}"

    def _read_file(self, input_text: str) -> str:
        """Parse and execute read_file command."""
        filepath = input_text.strip()
        path = Path(filepath)

        if not path.exists():
            return f"✗ File not found: {filepath}"

        content = path.read_text()
        return f"File: {filepath}\n{'='*60}\n{content}"

    def _list_files(self, input_text: str) -> str:
        """Parse and execute list_files command."""
        parts = input_text.strip().split()
        directory = parts[0] if parts else "."
        pattern = parts[1] if len(parts) > 1 else "*"

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

    def _edit_file(self, input_text: str) -> str:
        """Parse and execute edit_file command."""
        # Expected format:
        # filepath
        # OLD_TEXT:
        # old text here
        # NEW_TEXT:
        # new text here

        if "OLD_TEXT:" not in input_text or "NEW_TEXT:" not in input_text:
            return "✗ Invalid edit_file format. Expected: filepath\\nOLD_TEXT:\\n<old>\\nNEW_TEXT:\\n<new>"

        parts = input_text.split("OLD_TEXT:")
        filepath = parts[0].strip()

        remaining = parts[1].split("NEW_TEXT:")
        old_text = remaining[0].strip()
        new_text = remaining[1].strip() if len(remaining) > 1 else ""

        path = Path(filepath)
        if not path.exists():
            return f"✗ File not found: {filepath}"

        content = path.read_text()
        if old_text not in content:
            return f"✗ Text not found in {filepath}"

        new_content = content.replace(old_text, new_text)
        path.write_text(new_content)

        return f"✓ Successfully edited {filepath} (replaced {len(old_text)} chars with {len(new_text)} chars)"

    def _execute_shell(self, input_text: str) -> str:
        """Parse and execute execute_shell command with SECURITY CHECKS."""
        from python_security import secure_shell_execute, SecurityViolation

        command = input_text.strip()

        try:
            # 🛡️ PROJECT PURGE: All commands go through Python Security Interceptor
            return secure_shell_execute(command, timeout=30)
        except SecurityViolation as e:
            return f"🚨 SECURITY VIOLATION: {str(e)}\nCommand blocked by Python Interceptor."
        except Exception as e:
            return f"✗ Error executing command: {str(e)}"

    def execute_task(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a coding task using GPT-5.

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

        # Add preamble instruction for transparency
        system_prompt = """You are an expert coding assistant with file operation and shell execution tools.

Before calling any tool, briefly explain why you are calling it.

When asked to create or modify code:
1. Think step-by-step about what needs to be done
2. Use the provided tools to read existing files if needed
3. Write or edit files as requested
4. Verify your changes if possible
5. Provide a clear summary of what you did

Always use the tools to actually perform file operations - don't just describe what to do!"""

        if show_progress:
            print(f"🚀 Starting GPT-5 Code Agent (model: {self.model}, reasoning: {self.reasoning_effort})...")
            print()

        # Create initial request
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "developer",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            reasoning={
                "effort": self.reasoning_effort
            },
            text={
                "verbosity": self.verbosity
            },
            tools=[
                WRITE_FILE_TOOL,
                READ_FILE_TOOL,
                LIST_FILES_TOOL,
                EDIT_FILE_TOOL,
                EXECUTE_SHELL_TOOL,
            ]
        )

        # Process response iterations
        iteration = 0
        messages = []

        while iteration < self.max_iterations:
            iteration += 1

            if show_progress:
                print(f"{'='*60}")
                print(f"Iteration {iteration}")
                print(f"{'='*60}")

            # Extract output items
            has_tool_calls = False
            tool_outputs = []

            for item in response.output:
                if item.type == "reasoning":
                    if show_progress and hasattr(item, 'summary') and item.summary:
                        summary_text = item.summary[0].text if item.summary else ""
                        print(f"💭 Reasoning: {summary_text}")

                elif item.type == "message":
                    if hasattr(item, 'content'):
                        for content in item.content:
                            if hasattr(content, 'text'):
                                if show_progress:
                                    print(f"📝 {content.text}")
                                messages.append(content.text)

                elif item.type == "custom_tool_call":
                    has_tool_calls = True
                    tool_name = item.name
                    tool_input = item.input
                    call_id = item.call_id

                    if show_progress:
                        print(f"🔧 Tool: {tool_name}")
                        print(f"   Input: {tool_input[:100]}...")

                    # Execute tool
                    output = self._execute_tool(tool_name, tool_input)

                    if show_progress:
                        print(f"   Output: {output[:200]}...")

                    tool_outputs.append({
                        "type": "custom_tool_call_output",
                        "call_id": call_id,
                        "output": output
                    })

            # If no tool calls, we're done
            if not has_tool_calls:
                if show_progress:
                    print()
                    print("✓ Agent execution complete")
                break

            # Continue with tool outputs
            response = self.client.responses.create(
                model=self.model,
                previous_response_id=response.id,
                input=tool_outputs,
                reasoning={
                    "effort": self.reasoning_effort
                },
                text={
                    "verbosity": self.verbosity
                },
                tools=[
                    WRITE_FILE_TOOL,
                    READ_FILE_TOOL,
                    LIST_FILES_TOOL,
                    EDIT_FILE_TOOL,
                    EXECUTE_SHELL_TOOL,
                ]
            )

        if iteration >= self.max_iterations:
            if show_progress:
                print("⚠ Maximum iterations reached")

        # Get final output
        final_output = ""
        for item in response.output:
            if item.type == "message" and hasattr(item, 'content'):
                for content in item.content:
                    if hasattr(content, 'text'):
                        final_output = content.text

        return {
            'status': 'completed',
            'result': final_output,
            'messages': messages,
        }
