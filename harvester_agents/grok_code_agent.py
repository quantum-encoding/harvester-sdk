"""
Grok Code Agent - Agentic Coding Assistant

Optimized for grok-code-fast-1 with:
- Streaming reasoning traces
- Native tool calling
- Context-aware prompting
- Cache optimization
- Iterative refinement

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Enable LD_PRELOAD safety for all subprocess calls
_SAFE_EXEC_LIB = str(Path(__file__).parent / "safe_exec.so")
if os.path.exists(_SAFE_EXEC_LIB):
    os.environ['LD_PRELOAD'] = _SAFE_EXEC_LIB
    logger.info(f"Safe execution library loaded: {_SAFE_EXEC_LIB}")


@dataclass
class ReasoningTrace:
    """Container for reasoning content from grok-code-fast-1"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    step_number: int = 0


@dataclass
class ToolCall:
    """Container for tool call information"""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class AgentTask:
    """Container for agentic task execution"""
    task_id: str
    description: str
    context: Dict[str, Any]
    reasoning_traces: List[ReasoningTrace] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    iterations: int = 0
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None


class GrokCodeAgent:
    """
    Grok Code Agent - Agentic coding assistant using grok-code-fast-1

    Designed for:
    - Multi-step coding tasks with tool use
    - Large codebase navigation
    - Iterative problem solving
    - Fast, cost-effective iterations
    """

    # Available tools for grok-code-fast-1
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file from the filesystem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file"
                        }
                    },
                    "required": ["file_path"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search for code patterns or text in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern or text to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory path to search in (optional)"
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "File pattern to filter (e.g., '*.py')"
                        }
                    },
                    "required": ["pattern"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory matching a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g., '**/*.py')"
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Execute a shell command and return output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory (optional)"
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "json_write",
                "description": "Write a Python dictionary to a JSON file with proper serialization (prevents corruption)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the JSON file to write"
                        },
                        "data": {
                            "type": "object",
                            "description": "Dictionary to serialize as JSON"
                        }
                    },
                    "required": ["file_path", "data"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "json_update",
                "description": "Read a JSON file, modify it with Python code, and write it back with proper serialization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the JSON file to update"
                        },
                        "update_code": {
                            "type": "string",
                            "description": "Python code to modify 'data' dict (e.g., \"data['key'] = 'value'\")"
                        }
                    },
                    "required": ["file_path", "update_code"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_text_file",
                "description": "Write plain text content to a file (use for non-JSON files like .py, .txt, .md, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write"
                        },
                        "mode": {
                            "type": "string",
                            "description": "Write mode: 'w' (overwrite) or 'a' (append)",
                            "enum": ["w", "a"]
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding (default: utf-8)"
                        }
                    },
                    "required": ["file_path", "content"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit a file by replacing specific text (safer than rewriting entire file)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to edit"
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Exact text to find and replace"
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text"
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace all occurrences (default: false, only first)"
                        }
                    },
                    "required": ["file_path", "old_text", "new_text"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "insert_lines",
                "description": "Insert lines at a specific line number in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file"
                        },
                        "line_number": {
                            "type": "integer",
                            "description": "Line number to insert at (1-based)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to insert"
                        }
                    },
                    "required": ["file_path", "line_number", "content"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete a file or directory (use for cleanup, removing temp files, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to file or directory to delete"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Delete directories recursively (default: false)"
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_directory",
                "description": "Create a directory (and parent directories if needed)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to create"
                        },
                        "parents": {
                            "type": "boolean",
                            "description": "Create parent directories if needed (default: true)"
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False
                }
            }
        }
    ]

    def __init__(self,
                 api_key: str = None,
                 model: str = "grok-code-fast-1",
                 max_iterations: int = 100,
                 temperature: float = 0.7):
        """
        Initialize Grok Code Agent

        Args:
            api_key: xAI API key
            model: Model to use (default: grok-code-fast-1)
            max_iterations: Maximum iterations for agentic tasks
            temperature: Model temperature
        """
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature

        # Message history for cache optimization
        self.message_history = []

        # Current task
        self.current_task: Optional[AgentTask] = None

        logger.info(f"🤖 Grok Code Agent initialized with {model}")

    def _get_api_key(self) -> str:
        """Get API key from environment"""
        import os
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")
        return api_key

    def format_context(self,
                      files: Dict[str, str] = None,
                      project_structure: str = None,
                      dependencies: List[str] = None,
                      **kwargs) -> str:
        """
        Format context with XML/Markdown tags for clarity

        Args:
            files: Dict of file_path: content
            project_structure: Project structure description
            dependencies: List of dependencies
            **kwargs: Additional context items

        Returns:
            Formatted context string
        """
        context_parts = []

        if project_structure:
            context_parts.append(f"""<project_structure>
{project_structure}
</project_structure>""")

        if dependencies:
            deps_str = "\n".join(f"- {dep}" for dep in dependencies)
            context_parts.append(f"""<dependencies>
{deps_str}
</dependencies>""")

        if files:
            for file_path, content in files.items():
                context_parts.append(f"""<file path="{file_path}">
```
{content}
```
</file>""")

        # Add any additional context
        for key, value in kwargs.items():
            context_parts.append(f"""<{key}>
{value}
</{key}>""")

        return "\n\n".join(context_parts)

    def create_system_prompt(self, task_type: str = "general") -> str:
        """
        Create detailed system prompt for grok-code-fast-1

        Args:
            task_type: Type of task (general, debugging, refactoring, feature)

        Returns:
            System prompt string
        """
        base_prompt = """You are an expert coding assistant powered by grok-code-fast-1, designed for agentic programming tasks.

**Your Capabilities:**
- Navigate large codebases efficiently
- Use multiple tools to gather information
- Think step-by-step and show your reasoning
- Iterate and refine solutions based on feedback
- Handle complex multi-step coding tasks

**Available Tools (11 total):**
You have access to file operations, code search, command execution, and more. Use them strategically to solve problems.

**Tool Selection Guidelines:**
- **For JSON files**: ALWAYS use json_write or json_update (never write_text_file)
- **For editing existing files**: Prefer edit_file (find/replace) over rewriting entire file
- **For new text files**: Use write_text_file for .py, .txt, .md, etc. (supports append mode)
- **For inserting code**: Use insert_lines when adding to specific line numbers
- **For reading**: Use read_file to examine existing code
- **For code search**: Uses ripgrep (10x faster) when available, fallback to grep
- **For cleanup**: Use delete_file to remove temp files or failed builds
- **For project setup**: Use create_directory to create folder structures
- **For safety**: Dangerous commands (rm -rf /, dd, fork bombs) are automatically blocked

**IMPORTANT - Task Execution Strategy:**
1. **Plan First**: Break the task into clear sequential steps
2. **Execute Decisively**: Don't just explore - write files, create structure
3. **Be Productive**: Each iteration should make concrete progress
4. **Minimize Checks**: Only verify what's necessary, then move forward
5. **Complete Fully**: Don't stop until the entire task is done

**Avoid These Mistakes:**
- ❌ Repeatedly checking versions or environments
- ❌ Listing files without a specific purpose
- ❌ Exploring without creating
- ✅ Make concrete progress with each tool call
- ✅ Write files, create structure, build features

**Output Format:**
- Use Markdown for code blocks with proper syntax highlighting
- Explain your reasoning clearly
- Provide detailed error handling
- When task is complete, provide summary WITHOUT calling more tools
"""

        task_specific = {
            "debugging": "\n**Debugging Focus:**\n- Identify root cause systematically\n- Use search tools to find related code\n- Test fixes thoroughly\n- Document the issue and solution",
            "refactoring": "\n**Refactoring Focus:**\n- Preserve functionality\n- Improve code quality and maintainability\n- Follow project conventions\n- Update related documentation",
            "feature": "\n**Feature Development Focus:**\n- Design before implementing\n- Consider edge cases and error handling\n- Write clean, testable code\n- Document new functionality"
        }

        return base_prompt + task_specific.get(task_type, "")

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool call

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # Tool implementations
        if tool_name == "read_file":
            file_path = arguments["file_path"]
            try:
                with open(file_path, 'r') as f:
                    return {"success": True, "content": f.read()}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "search_code":
            # Implement code search using ripgrep (fast) or grep (fallback)
            pattern = arguments["pattern"]
            path = arguments.get("path", ".")
            file_pattern = arguments.get("file_pattern", "*")

            import subprocess
            import shutil
            try:
                # Prefer ripgrep (10x faster) if available
                rg_path = shutil.which('rg')
                if rg_path:
                    cmd = ["rg", "-n", pattern, path]
                    if file_pattern != "*":
                        cmd.extend(["--glob", file_pattern])
                else:
                    # Fallback to grep
                    cmd = ["grep", "-r", "-n", pattern, path]
                    if file_pattern != "*":
                        cmd.extend(["--include", file_pattern])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                return {"success": True, "matches": result.stdout}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "list_files":
            path = arguments["path"]
            pattern = arguments.get("pattern", "*")

            from glob import glob
            try:
                files = glob(f"{path}/{pattern}", recursive=True)
                return {"success": True, "files": files}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "execute_command":
            command = arguments["command"]
            cwd = arguments.get("cwd", ".")

            # Safety: Block dangerous commands
            dangerous_patterns = [
                'rm -rf /', 'rm -rf ~', 'rm -rf *',  # Destructive rm
                'mkfs', 'dd if=', 'dd of=',  # Filesystem/disk operations
                ':(){:|:&};:',  # Fork bomb
                '> /dev/sd', '> /dev/hd',  # Direct disk writes
                'chmod -R 777 /',  # Dangerous permissions
                'chown -R'  # Mass ownership changes
            ]

            command_lower = command.lower()
            for pattern in dangerous_patterns:
                if pattern.lower() in command_lower:
                    return {
                        "success": False,
                        "error": f"Potentially dangerous command rejected: contains '{pattern}'"
                    }

            import subprocess
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "json_write":
            file_path = arguments["file_path"]
            data = arguments["data"]
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                return {"success": True, "message": f"Wrote JSON to {file_path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "json_update":
            file_path = arguments["file_path"]
            update_code = arguments["update_code"]

            try:
                # Read existing JSON
                if not Path(file_path).exists():
                    return {"success": False, "error": f"File {file_path} does not exist"}

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Execute update code with restricted environment
                # Only allow basic Python operations, no imports or dangerous builtins
                safe_globals = {
                    '__builtins__': {
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'bool': bool,
                        'list': list,
                        'dict': dict,
                        'set': set,
                        'tuple': tuple,
                        'range': range,
                        'enumerate': enumerate,
                        'zip': zip,
                        'map': map,
                        'filter': filter,
                        'sorted': sorted,
                        'sum': sum,
                        'min': min,
                        'max': max,
                        'abs': abs,
                        'round': round,
                        'any': any,
                        'all': all,
                    },
                    'data': data
                }

                exec(update_code, safe_globals)

                # Write back with proper serialization
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                return {"success": True, "message": f"Updated JSON in {file_path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "write_text_file":
            file_path = arguments["file_path"]
            content = arguments["content"]
            mode = arguments.get("mode", "w")  # Default: overwrite
            encoding = arguments.get("encoding", "utf-8")
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, mode, encoding=encoding) as f:
                    f.write(content)
                action = "Appended" if mode == "a" else "Wrote"
                return {"success": True, "message": f"{action} {len(content)} chars to {file_path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "edit_file":
            file_path = arguments["file_path"]
            old_text = arguments["old_text"]
            new_text = arguments["new_text"]
            replace_all = arguments.get("replace_all", False)

            try:
                if not Path(file_path).exists():
                    return {"success": False, "error": f"File {file_path} does not exist"}

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if old_text not in content:
                    return {"success": False, "error": f"Text not found in {file_path}"}

                if replace_all:
                    new_content = content.replace(old_text, new_text)
                    count = content.count(old_text)
                else:
                    new_content = content.replace(old_text, new_text, 1)
                    count = 1

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                return {"success": True, "message": f"Replaced {count} occurrence(s) in {file_path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "insert_lines":
            file_path = arguments["file_path"]
            line_number = arguments["line_number"]
            content = arguments["content"]

            try:
                if not Path(file_path).exists():
                    return {"success": False, "error": f"File {file_path} does not exist"}

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Insert at line_number (1-based index)
                if line_number < 1 or line_number > len(lines) + 1:
                    return {"success": False, "error": f"Line number {line_number} out of range (1-{len(lines)+1})"}

                # Insert content (ensure it ends with newline)
                insert_content = content if content.endswith('\n') else content + '\n'
                lines.insert(line_number - 1, insert_content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                return {"success": True, "message": f"Inserted content at line {line_number} in {file_path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "delete_file":
            path = arguments["path"]
            recursive = arguments.get("recursive", False)

            try:
                import shutil
                path_obj = Path(path)

                if not path_obj.exists():
                    return {"success": False, "error": f"Path does not exist: {path}"}

                if path_obj.is_file():
                    path_obj.unlink()
                    return {"success": True, "message": f"Deleted file: {path}"}
                elif path_obj.is_dir():
                    if recursive:
                        shutil.rmtree(path)
                        return {"success": True, "message": f"Deleted directory recursively: {path}"}
                    else:
                        path_obj.rmdir()  # Only works if empty
                        return {"success": True, "message": f"Deleted empty directory: {path}"}
                else:
                    return {"success": False, "error": f"Unknown path type: {path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "create_directory":
            path = arguments["path"]
            parents = arguments.get("parents", True)

            try:
                path_obj = Path(path)
                path_obj.mkdir(parents=parents, exist_ok=True)
                return {"success": True, "message": f"Created directory: {path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Unknown tool: {tool_name}"}

    async def stream_completion(self,
                               messages: List[Dict[str, str]],
                               tools: List[Dict] = None,
                               on_reasoning: Callable[[str], None] = None,
                               on_chunk: Callable[[str], None] = None) -> Dict[str, Any]:
        """
        Stream completion with reasoning trace extraction

        Args:
            messages: Chat messages
            tools: Available tools
            on_reasoning: Callback for reasoning content
            on_chunk: Callback for text chunks

        Returns:
            Complete response with reasoning and tool calls
        """
        import aiohttp

        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True
        }

        if tools:
            payload["tools"] = tools

        reasoning_parts = []
        content_parts = []
        tool_calls_dict = {}  # Accumulate tool calls by index

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                delta = chunk["choices"][0]["delta"]

                                # Extract reasoning content
                                if "reasoning_content" in delta:
                                    reasoning = delta["reasoning_content"]
                                    reasoning_parts.append(reasoning)
                                    if on_reasoning:
                                        on_reasoning(reasoning)

                                # Extract content
                                if "content" in delta:
                                    content = delta["content"]
                                    content_parts.append(content)
                                    if on_chunk:
                                        on_chunk(content)

                                # Extract tool calls (streamed incrementally)
                                if "tool_calls" in delta:
                                    for tc_delta in delta["tool_calls"]:
                                        idx = tc_delta.get("index", 0)

                                        if idx not in tool_calls_dict:
                                            tool_calls_dict[idx] = {
                                                "id": "",
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": ""
                                                }
                                            }

                                        # Accumulate tool call data
                                        if "id" in tc_delta:
                                            tool_calls_dict[idx]["id"] = tc_delta["id"]
                                        if "function" in tc_delta:
                                            if "name" in tc_delta["function"]:
                                                tool_calls_dict[idx]["function"]["name"] += tc_delta["function"]["name"]
                                            if "arguments" in tc_delta["function"]:
                                                tool_calls_dict[idx]["function"]["arguments"] += tc_delta["function"]["arguments"]

                            except json.JSONDecodeError:
                                continue

        # Convert accumulated tool calls to final format
        tool_calls = []
        for idx in sorted(tool_calls_dict.keys()):
            tc = tool_calls_dict[idx]
            try:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {tc['function']['arguments']}")

        return {
            "reasoning": "".join(reasoning_parts),
            "content": "".join(content_parts),
            "tool_calls": tool_calls
        }

    async def execute_task(self,
                          description: str,
                          context: Dict[str, Any] = None,
                          task_type: str = "general") -> AgentTask:
        """
        Execute an agentic coding task

        Args:
            description: Task description
            context: Initial context (files, structure, etc.)
            task_type: Type of task for system prompt

        Returns:
            Completed AgentTask
        """
        task_id = f"task_{datetime.now().timestamp()}"
        task = AgentTask(
            task_id=task_id,
            description=description,
            context=context or {},
            status="running"
        )

        self.current_task = task

        # Build initial message with context
        system_prompt = self.create_system_prompt(task_type)

        if context:
            formatted_context = self.format_context(**context)
            user_message = f"{description}\n\n{formatted_context}"
        else:
            user_message = description

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Store for cache optimization (don't modify prefix)
        self.message_history = messages.copy()

        # Track recent tool calls to detect loops (file reads only)
        recent_file_reads = []
        loop_threshold = 3  # If same file read 3+ times in row, warn

        # Execute with tool calling loop
        for iteration in range(self.max_iterations):
            task.iterations = iteration + 1

            print(f"\n{'='*60}")
            print(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")

            # Stream completion with reasoning
            response = await self.stream_completion(
                messages=messages,
                tools=self.TOOLS,
                on_reasoning=lambda r: logger.debug(f"💭 Reasoning: {r[:100]}..."),
                on_chunk=lambda c: logger.debug(f"📝 Content: {c[:100]}...")
            )

            # Store reasoning trace
            if response["reasoning"]:
                trace = ReasoningTrace(
                    content=response["reasoning"],
                    step_number=iteration + 1
                )
                task.reasoning_traces.append(trace)

            # Handle tool calls
            if response.get("tool_calls") and len(response["tool_calls"]) > 0:
                print(f"\n🔧 Tool Calls ({len(response['tool_calls'])})")

                # Check for loops on read_file operations only
                for tc in response["tool_calls"]:
                    if tc["name"] == "read_file":
                        file_path = tc.get("arguments", {}).get("file_path", "")
                        if file_path:
                            recent_file_reads.append(file_path)
                            recent_file_reads = recent_file_reads[-10:]  # Keep last 10

                # Detect if stuck reading same file
                if len(recent_file_reads) >= loop_threshold:
                    last_reads = recent_file_reads[-loop_threshold:]
                    if len(set(last_reads)) == 1:  # Same file read 3+ times
                        print(f"⚠️  Warning: Reading '{last_reads[0]}' {loop_threshold} times in a row - possible loop")

                for tool_call in response["tool_calls"]:
                    print(f"  → {tool_call['name']}({tool_call['arguments']})", flush=True)

                    result = await self.execute_tool(
                        tool_call["name"],
                        tool_call["arguments"]
                    )

                    # Show brief result
                    if result.get("success"):
                        if "content" in result:
                            content_preview = result["content"][:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
                            print(f"    ✓ {content_preview}")
                        elif "message" in result:
                            print(f"    ✓ {result['message']}")
                        else:
                            print(f"    ✓ Success")
                    else:
                        print(f"    ✗ Error: {result.get('error', 'Unknown error')}")

                    task.tool_calls.append(ToolCall(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                        result=result
                    ))

                    # Add tool result to messages (preserve cache prefix)
                    # Format tool_call in OpenAI/xAI expected format
                    formatted_tool_call = {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"])
                        }
                    }

                    messages.append({
                        "role": "assistant",
                        "content": response["content"] or "",
                        "tool_calls": [formatted_tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result)
                    })
            else:
                # No more tool calls, task complete
                logger.info(f"✅ Task completed - no tool calls in response")
                logger.info(f"📝 Final content: {response['content'][:200]}..." if response['content'] else "📝 No content")
                task.status = "completed"
                task.result = response["content"] or "No response content"
                break

        if task.status != "completed":
            task.status = "failed"
            task.result = "Max iterations reached without completion"

        return task


# Export
__all__ = ['GrokCodeAgent', 'AgentTask', 'ReasoningTrace', 'ToolCall']
