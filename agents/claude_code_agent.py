"""
Claude Code Agent - Professional Agentic Coding Assistant

Built on Anthropic's official Claude Agent SDK with:
- Production-tested agent loop (no custom implementation needed)
- Built-in context compaction (automatic cache optimization)
- Proper verification and error handling
- Subagents for parallel work
- MCP integration for external services
- Custom tools for harvester integration

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Claude Agent SDK imports
from claude_agent_sdk import query, tool, ClaudeAgentOptions

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Container for agentic task execution"""
    task_id: str
    description: str
    context: Dict[str, Any]
    iterations: int = 0
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)


class ClaudeCodeAgent:
    """
    Claude Code Agent - Professional agentic coding assistant

    Built on Anthropic's Claude Agent SDK - the same infrastructure
    powering Claude Code CLI.

    Key Advantages over Custom Implementations:
    - Production-tested agent loop (Anthropic's own)
    - Automatic context management and compaction
    - Built-in verification hooks
    - Subagents for parallel processing
    - MCP protocol support
    - Professional error handling

    Use Cases:
    - Multi-step coding tasks
    - Large codebase navigation
    - Feature development
    - Debugging and refactoring
    - Integration with harvester providers
    """

    def __init__(self,
                 model: str = "claude-sonnet-4-5",
                 max_iterations: int = 100,
                 working_directory: str = None):
        """
        Initialize Claude Code Agent

        Args:
            model: Claude model to use (default: claude-sonnet-4-5)
            max_iterations: Maximum iterations for agentic tasks
            working_directory: Working directory for file operations
        """
        self.model = model
        self.max_iterations = max_iterations
        self.working_directory = working_directory or str(Path.cwd())

        # Custom tools registry
        self.custom_tools = []

        # Current task
        self.current_task: Optional[AgentTask] = None

        # Register our custom harvester tools
        self._register_harvester_tools()

        logger.info(f"🤖 Claude Code Agent initialized with {model}")

    def _register_harvester_tools(self):
        """Register custom tools for harvester integration"""

        # Tool: Query other providers via harvester
        @tool("query_provider",
              "Query another AI provider through harvester (OpenAI, DeepSeek, xAI, etc.)",
              {"provider": str, "model": str, "prompt": str})
        async def query_provider(args):
            """Query another provider via harvester"""
            try:
                from core.harvester import Harvester

                harvester = Harvester()
                response = await harvester.query(
                    provider=args["provider"],
                    model=args["model"],
                    prompt=args["prompt"]
                )

                return {
                    "content": [{
                        "type": "text",
                        "text": f"Provider {args['provider']} ({args['model']}) response:\n{response}"
                    }]
                }
            except Exception as e:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error querying provider: {str(e)}"
                    }],
                    "isError": True
                }

        # Tool: JSON write (corruption-proof)
        @tool("json_write",
              "Write a Python dictionary to JSON file with proper serialization",
              {"file_path": str, "data": dict})
        async def json_write_tool(args):
            """Write JSON with proper serialization"""
            try:
                file_path = Path(self.working_directory) / args["file_path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(args["data"], f, indent=4, ensure_ascii=False)

                return {
                    "content": [{
                        "type": "text",
                        "text": f"✅ Successfully wrote JSON to {file_path}"
                    }]
                }
            except Exception as e:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"❌ Error writing JSON: {str(e)}"
                    }],
                    "isError": True
                }

        # Tool: JSON update (sandboxed)
        @tool("json_update",
              "Update a JSON file by executing Python code on the data dict",
              {"file_path": str, "update_code": str})
        async def json_update_tool(args):
            """Update JSON with sandboxed Python code"""
            try:
                file_path = Path(self.working_directory) / args["file_path"]

                if not file_path.exists():
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"❌ File does not exist: {file_path}"
                        }],
                        "isError": True
                    }

                # Read JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Execute update code with restricted environment
                safe_globals = {
                    '__builtins__': {
                        'len': len, 'str': str, 'int': int, 'float': float,
                        'bool': bool, 'list': list, 'dict': dict, 'set': set,
                        'tuple': tuple, 'range': range, 'enumerate': enumerate,
                        'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                        'sum': sum, 'min': min, 'max': max, 'abs': abs,
                        'round': round, 'any': any, 'all': all,
                    },
                    'data': data
                }

                exec(args["update_code"], safe_globals)

                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                return {
                    "content": [{
                        "type": "text",
                        "text": f"✅ Successfully updated JSON in {file_path}"
                    }]
                }
            except Exception as e:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"❌ Error updating JSON: {str(e)}"
                    }],
                    "isError": True
                }

        self.custom_tools.extend([query_provider, json_write_tool, json_update_tool])
        logger.debug(f"Registered {len(self.custom_tools)} custom tools")

    def create_system_prompt(self, task_type: str = "general") -> str:
        """
        Create system prompt for Claude Code Agent

        Args:
            task_type: Type of task (general, debugging, refactoring, feature)

        Returns:
            System prompt string
        """
        base_prompt = """You are an expert coding assistant powered by Claude Sonnet 4.5, with access to the Claude Agent SDK's professional tooling.

**Your Capabilities:**
- Navigate and understand large codebases
- Execute multi-step coding tasks autonomously
- Use tools strategically to gather information and make changes
- Leverage context compaction for efficient iteration
- Access to harvester platform (query other AI providers)

**Available Tools:**
You have access to file operations, shell commands, JSON tools, and more through the Claude Agent SDK.

**Special Harvester Tools:**
- **query_provider**: Query other AI providers (OpenAI, DeepSeek, Grok, etc.) for specialized tasks
- **json_write**: Write JSON files with corruption-proof serialization
- **json_update**: Update JSON files with sandboxed Python code

**Best Practices:**
1. **Plan before acting**: Break complex tasks into clear steps
2. **Use appropriate tools**: Choose the right tool for each operation
3. **Verify your work**: Check file contents after modifications
4. **Leverage other providers**: Use query_provider for specialized tasks
5. **Be concise**: Focus on the task at hand

**For JSON files**: ALWAYS use json_write or json_update (never plain file writes)
**For code edits**: Use SDK's built-in edit tools
**For complex queries**: Consider query_provider to leverage specialized models
"""

        task_specific = {
            "debugging": """
**Debugging Focus:**
- Identify root cause systematically
- Use search tools to find related code
- Test fixes thoroughly with execute commands
- Consider using query_provider for alternative perspectives
- Document the issue and solution clearly
""",
            "refactoring": """
**Refactoring Focus:**
- Preserve functionality (verify with tests)
- Improve code quality and maintainability
- Follow project conventions
- Update related documentation
- Use git to track changes
""",
            "feature": """
**Feature Development Focus:**
- Design the feature architecture first
- Consider edge cases and error handling
- Write clean, testable, documented code
- Test the feature thoroughly
- Update documentation and examples
"""
        }

        return base_prompt + task_specific.get(task_type, "")

    def format_context(self,
                      files: Dict[str, str] = None,
                      project_structure: str = None,
                      dependencies: List[str] = None,
                      **kwargs) -> str:
        """
        Format context for the agent

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

    async def execute_task(self,
                          description: str,
                          context: Dict[str, Any] = None,
                          task_type: str = "general",
                          show_progress: bool = True) -> AgentTask:
        """
        Execute an agentic coding task using Claude Agent SDK

        Args:
            description: Task description
            context: Context dict with 'files', 'project_structure', etc.
            task_type: Type of task (general, debugging, refactoring, feature)
            show_progress: Show progress during execution

        Returns:
            AgentTask with results
        """
        import uuid

        # Create task
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            description=description,
            context=context or {}
        )
        task.status = "running"
        self.current_task = task

        if show_progress:
            print(f"🤖 Claude Code Agent")
            print(f"📋 Task: {description}")
            print(f"🎯 Type: {task_type}")
            print()

        # Format context
        context_str = ""
        if context:
            context_str = self.format_context(
                files=context.get('files'),
                project_structure=context.get('project_structure'),
                dependencies=context.get('dependencies')
            )

        # Build prompt
        system_prompt = self.create_system_prompt(task_type)
        user_prompt = f"""{context_str}

**Task**: {description}

Please complete this task using the available tools. When done, provide a summary of what was accomplished."""

        # Configure Claude Agent SDK options
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            working_directory=self.working_directory,
            tools=self.custom_tools  # Add our custom tools
        )

        # Execute with SDK's query() function
        # The SDK handles: agent loop, tool execution, context management, verification
        messages = []
        try:
            async for message in query(
                prompt=user_prompt,
                options=options
            ):
                task.iterations += 1
                messages.append(message)

                if show_progress:
                    # Show messages as they come
                    if isinstance(message, dict):
                        if message.get('type') == 'text':
                            print(message.get('content', ''))
                        elif message.get('type') == 'tool_use':
                            print(f"🔧 Using tool: {message.get('name', 'unknown')}")
                    else:
                        print(message)

                # Safety: prevent infinite loops
                if task.iterations >= self.max_iterations:
                    if show_progress:
                        print(f"\n⚠️  Reached max iterations ({self.max_iterations})")
                    break

            task.status = "completed"
            task.messages = messages

            # Extract final result
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    task.result = last_message.get('content', str(last_message))
                else:
                    task.result = str(last_message)
            else:
                task.result = "No response from agent"

        except Exception as e:
            task.status = "failed"
            task.result = f"Error: {str(e)}"
            logger.error(f"Task failed: {e}", exc_info=True)

        if show_progress:
            print(f"\n📊 Status: {task.status}")
            print(f"🔄 Iterations: {task.iterations}")
            if task.result:
                print(f"\n✅ Result:\n{task.result}")

        return task


# Export
__all__ = ['ClaudeCodeAgent', 'AgentTask']
