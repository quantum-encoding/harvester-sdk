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
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


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
                "name": "write_file",
                "description": "Write or update a file on the filesystem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
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

**Available Tools:**
You have access to file operations, code search, command execution, and more. Use them strategically to solve problems.

**Best Practices:**
1. Break complex tasks into smaller steps
2. Use tools to gather necessary context before making changes
3. Show your reasoning process
4. Verify changes work correctly
5. Handle edge cases and errors properly

**Output Format:**
- Use Markdown for code blocks with proper syntax highlighting
- Explain your reasoning clearly
- Provide detailed error handling
- Suggest improvements when relevant
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

        elif tool_name == "write_file":
            file_path = arguments["file_path"]
            content = arguments["content"]
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(content)
                return {"success": True, "message": f"Wrote {len(content)} chars to {file_path}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name == "search_code":
            # Implement code search using grep/ripgrep
            pattern = arguments["pattern"]
            path = arguments.get("path", ".")
            file_pattern = arguments.get("file_pattern", "*")

            import subprocess
            try:
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
