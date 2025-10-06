"""
Function Calling & Tool Use Engine - Premium Tier Feature

This module provides agentic AI capabilities with function calling, tool use,
and workflow orchestration. Enables AI models to interact with external systems,
execute code, manipulate files, and perform complex multi-step operations.

Key Features:
- Built-in tool library (file ops, web, code execution)
- Custom tool registration
- Tool chaining and workflows
- Provider-agnostic function calling
- Automatic function schema generation
- Tool execution sandboxing

Copyright (c) 2025 Quantum Encoding Ltd.
Licensed under the Harvester Commercial License.
"""

import json
import logging
import asyncio
import subprocess
import tempfile
import os
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, get_type_hints
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import importlib.util

try:
    import requests
    WEB_TOOLS_AVAILABLE = True
except ImportError:
    WEB_TOOLS_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, create_model
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

logger = logging.getLogger(__name__)


class FunctionCallingError(Exception):
    """Base exception for function calling errors"""
    pass


class ToolNotFoundError(FunctionCallingError):
    """Raised when requested tool is not available"""
    pass


class ToolExecutionError(FunctionCallingError):
    """Raised when tool execution fails"""
    pass


class SecurityError(FunctionCallingError):
    """Raised when tool execution violates security constraints"""
    pass


@dataclass
class ToolDefinition:
    """Definition of a callable tool"""
    name: str
    description: str
    function: Callable
    parameters_schema: Dict[str, Any]
    category: str = "custom"
    security_level: str = "safe"  # safe, elevated, dangerous
    requires_confirmation: bool = False
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema
        }


@dataclass
class FunctionCall:
    """Represents a function call request from AI"""
    name: str
    arguments: Dict[str, Any]
    call_id: str
    model: str
    provider: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FunctionResult:
    """Result of a function call execution"""
    call: FunctionCall
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    security_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'call_id': self.call.call_id,
            'function_name': self.call.name,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'security_warnings': self.security_warnings,
            'timestamp': self.call.timestamp.isoformat()
        }


class ToolCategory(Enum):
    """Categories of available tools"""
    FILE_OPERATIONS = "file_operations"
    WEB_TOOLS = "web_tools"
    CODE_EXECUTION = "code_execution"
    SYSTEM_TOOLS = "system_tools"
    DATA_PROCESSING = "data_processing"
    CUSTOM = "custom"


class FunctionRegistry:
    """
    Central registry for all available functions and tools
    
    Manages tool discovery, registration, and execution with security controls
    """
    
    def __init__(self, security_mode: str = "safe"):
        """
        Initialize function registry
        
        Args:
            security_mode: Security level - 'safe', 'elevated', 'unrestricted'
        """
        self.tools: Dict[str, ToolDefinition] = {}
        self.security_mode = security_mode
        self.execution_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'security_violations': 0
        }
        
        # Load built-in tools
        self._load_builtin_tools()
        
        logger.info(f"🔧 Function Registry initialized with {len(self.tools)} tools")
    
    def _load_builtin_tools(self):
        """Load the built-in tool library"""
        
        # File Operations (Professional+)
        self.register_tool(
            name="read_file",
            description="Read contents of a text file",
            function=self._read_file,
            category=ToolCategory.FILE_OPERATIONS.value,
            security_level="safe"
        )
        
        self.register_tool(
            name="write_file", 
            description="Write content to a file",
            function=self._write_file,
            category=ToolCategory.FILE_OPERATIONS.value,
            security_level="elevated",
            requires_confirmation=True
        )
        
        self.register_tool(
            name="list_directory",
            description="List files and directories in a path",
            function=self._list_directory,
            category=ToolCategory.FILE_OPERATIONS.value,
            security_level="safe"
        )
        
        # Web Tools (Premium+)
        if WEB_TOOLS_AVAILABLE:
            self.register_tool(
                name="web_search",
                description="Search the web and return results",
                function=self._web_search,
                category=ToolCategory.WEB_TOOLS.value,
                security_level="safe"
            )
            
            self.register_tool(
                name="fetch_url",
                description="Fetch content from a URL",
                function=self._fetch_url,
                category=ToolCategory.WEB_TOOLS.value,
                security_level="safe"
            )
        
        # Code Execution (Premium+)
        self.register_tool(
            name="execute_python",
            description="Execute Python code safely in a sandbox",
            function=self._execute_python,
            category=ToolCategory.CODE_EXECUTION.value,
            security_level="elevated",
            requires_confirmation=True
        )
        
        self.register_tool(
            name="execute_shell",
            description="Execute shell command (restricted)",
            function=self._execute_shell,
            category=ToolCategory.SYSTEM_TOOLS.value,
            security_level="dangerous",
            requires_confirmation=True
        )
        
        # Data Processing Tools (Premium+)
        self.register_tool(
            name="analyze_json",
            description="Parse and analyze JSON data",
            function=self._analyze_json,
            category=ToolCategory.DATA_PROCESSING.value,
            security_level="safe"
        )
    
    def register_tool(self, 
                     name: str,
                     description: str, 
                     function: Callable,
                     category: str = "custom",
                     security_level: str = "safe",
                     requires_confirmation: bool = False):
        """
        Register a new tool in the registry
        
        Args:
            name: Unique tool name
            description: Human-readable description
            function: The callable function
            category: Tool category
            security_level: Security level (safe, elevated, dangerous)
            requires_confirmation: Whether to require user confirmation
        """
        # Generate parameter schema from function signature
        schema = self._generate_parameter_schema(function)
        
        tool_def = ToolDefinition(
            name=name,
            description=description,
            function=function,
            parameters_schema=schema,
            category=category,
            security_level=security_level,
            requires_confirmation=requires_confirmation
        )
        
        self.tools[name] = tool_def
        logger.debug(f"🔧 Registered tool: {name} ({category})")
    
    def _generate_parameter_schema(self, function: Callable) -> Dict[str, Any]:
        """Generate JSON schema from function signature"""
        sig = inspect.signature(function)
        type_hints = get_type_hints(function)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name, str)
            
            # Convert Python types to JSON schema types
            if param_type == str:
                prop_schema = {"type": "string"}
            elif param_type == int:
                prop_schema = {"type": "integer"}
            elif param_type == float:
                prop_schema = {"type": "number"}
            elif param_type == bool:
                prop_schema = {"type": "boolean"}
            elif param_type == list:
                prop_schema = {"type": "array", "items": {"type": "string"}}
            elif param_type == dict:
                prop_schema = {"type": "object"}
            else:
                prop_schema = {"type": "string"}  # Default fallback
            
            # Add description from docstring if available
            if function.__doc__:
                prop_schema["description"] = f"Parameter for {function.__name__}"
            
            properties[param_name] = prop_schema
            
            # Check if parameter is required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a specific category"""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def get_tools_for_tier(self, tier: str) -> List[ToolDefinition]:
        """Get available tools for a specific license tier"""
        if tier in ['freemium']:
            return []  # No function calling
        elif tier in ['professional']:
            return self.get_tools_by_category(ToolCategory.FILE_OPERATIONS.value)
        elif tier in ['premium']:
            categories = [
                ToolCategory.FILE_OPERATIONS.value,
                ToolCategory.WEB_TOOLS.value,
                ToolCategory.CODE_EXECUTION.value,
                ToolCategory.DATA_PROCESSING.value
            ]
            tools = []
            for category in categories:
                tools.extend(self.get_tools_by_category(category))
            return tools
        elif tier in ['enterprise']:
            return list(self.tools.values())  # All tools
        else:
            return []
    
    async def execute_function(self, call: FunctionCall) -> FunctionResult:
        """
        Execute a function call with security checks
        
        Args:
            call: The function call to execute
            
        Returns:
            Function execution result
        """
        start_time = datetime.now()
        self.execution_stats['total_calls'] += 1
        
        # Check if tool exists
        if call.name not in self.tools:
            self.execution_stats['failed_calls'] += 1
            return FunctionResult(
                call=call,
                success=False,
                error=f"Tool '{call.name}' not found"
            )
        
        tool = self.tools[call.name]
        
        # Security checks
        security_warnings = []
        if not self._check_security_permissions(tool):
            self.execution_stats['security_violations'] += 1
            return FunctionResult(
                call=call,
                success=False,
                error=f"Security violation: Tool '{call.name}' requires {tool.security_level} permissions",
                security_warnings=["Security level insufficient"]
            )
        
        try:
            # Validate arguments against schema
            self._validate_arguments(call.arguments, tool.parameters_schema)
            
            # Execute the function
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**call.arguments)
            else:
                result = tool.function(**call.arguments)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.execution_stats['successful_calls'] += 1
            
            return FunctionResult(
                call=call,
                success=True,
                result=result,
                execution_time=execution_time,
                security_warnings=security_warnings
            )
            
        except Exception as e:
            self.execution_stats['failed_calls'] += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return FunctionResult(
                call=call,
                success=False,
                error=str(e),
                execution_time=execution_time,
                security_warnings=security_warnings
            )
    
    def _check_security_permissions(self, tool: ToolDefinition) -> bool:
        """Check if current security mode allows tool execution"""
        if self.security_mode == "unrestricted":
            return True
        elif self.security_mode == "elevated":
            return tool.security_level in ["safe", "elevated"]
        elif self.security_mode == "safe":
            return tool.security_level == "safe"
        else:
            return False
    
    def _validate_arguments(self, arguments: Dict[str, Any], schema: Dict[str, Any]):
        """Validate function arguments against schema"""
        # Basic validation - could be enhanced with jsonschema library
        required = schema.get('required', [])
        properties = schema.get('properties', {})
        
        # Check required parameters
        for req_param in required:
            if req_param not in arguments:
                raise ValueError(f"Missing required parameter: {req_param}")
        
        # Check parameter types (basic validation)
        for param_name, value in arguments.items():
            if param_name in properties:
                expected_type = properties[param_name].get('type')
                if expected_type == 'string' and not isinstance(value, str):
                    raise ValueError(f"Parameter {param_name} must be a string")
                elif expected_type == 'integer' and not isinstance(value, int):
                    raise ValueError(f"Parameter {param_name} must be an integer")
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter {param_name} must be a number")
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    raise ValueError(f"Parameter {param_name} must be a boolean")
    
    # Built-in tool implementations
    def _read_file(self, file_path: str) -> str:
        """Read contents of a text file"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Security check - no reading system files
            if str(path).startswith(('/etc/', '/proc/', '/sys/')):
                raise SecurityError("Access to system files denied")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content[:10000]  # Limit to 10KB
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to read file: {e}")
    
    def _write_file(self, file_path: str, content: str) -> str:
        """Write content to a file"""
        try:
            path = Path(file_path)
            
            # Security checks
            if str(path).startswith(('/etc/', '/proc/', '/sys/', '/usr/')):
                raise SecurityError("Access to system directories denied")
            
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {file_path}"
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to write file: {e}")
    
    def _list_directory(self, directory_path: str) -> List[str]:
        """List files and directories in a path"""
        try:
            path = Path(directory_path)
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            # Security check
            if str(path).startswith(('/etc/', '/proc/', '/sys/')):
                raise SecurityError("Access to system directories denied")
            
            items = []
            for item in path.iterdir():
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
            
            return items[:100]  # Limit to 100 items
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to list directory: {e}")
    
    def _web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web and return results"""
        # Placeholder implementation - would integrate with search API
        return [
            {
                "title": f"Search result for: {query}",
                "url": "https://example.com/search",
                "snippet": f"This would be a real search result for '{query}'"
            }
        ]
    
    def _fetch_url(self, url: str) -> str:
        """Fetch content from a URL"""
        try:
            if not WEB_TOOLS_AVAILABLE:
                raise ToolExecutionError("Web tools not available - install requests")
            
            # Security checks
            if not url.startswith(('http://', 'https://')):
                raise SecurityError("Only HTTP/HTTPS URLs allowed")
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Harvester-SDK/1.0'
            })
            response.raise_for_status()
            
            return response.text[:50000]  # Limit to 50KB
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to fetch URL: {e}")
    
    def _execute_python(self, code: str) -> str:
        """Execute Python code safely in a sandbox"""
        try:
            # Security checks
            dangerous_imports = ['os', 'sys', 'subprocess', 'socket', 'urllib']
            for dangerous in dangerous_imports:
                if f'import {dangerous}' in code or f'from {dangerous}' in code:
                    raise SecurityError(f"Import of '{dangerous}' is not allowed")
            
            # Create a restricted environment
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                }
            }
            
            # Capture output
            from io import StringIO
            import contextlib
            
            output = StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, restricted_globals)
            
            return output.getvalue() or "Code executed successfully (no output)"
            
        except Exception as e:
            raise ToolExecutionError(f"Python execution failed: {e}")
    
    def _execute_shell(self, command: str) -> str:
        """Execute shell command (restricted)"""
        # Very restricted shell execution
        allowed_commands = ['ls', 'pwd', 'date', 'whoami', 'echo']
        
        cmd_parts = command.split()
        if not cmd_parts or cmd_parts[0] not in allowed_commands:
            raise SecurityError(f"Command '{cmd_parts[0] if cmd_parts else 'empty'}' not allowed")
        
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return f"Command failed: {result.stderr}"
            
            return result.stdout
            
        except Exception as e:
            raise ToolExecutionError(f"Shell execution failed: {e}")
    
    def _analyze_json(self, json_data: str) -> Dict[str, Any]:
        """Parse and analyze JSON data"""
        try:
            data = json.loads(json_data)
            
            analysis = {
                "type": type(data).__name__,
                "valid_json": True,
                "structure": self._analyze_json_structure(data)
            }
            
            if isinstance(data, dict):
                analysis["keys"] = list(data.keys())
                analysis["key_count"] = len(data)
            elif isinstance(data, list):
                analysis["length"] = len(data)
                analysis["item_types"] = list(set(type(item).__name__ for item in data))
            
            return analysis
            
        except json.JSONDecodeError as e:
            return {
                "valid_json": False,
                "error": str(e),
                "type": "invalid"
            }
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 3) -> Dict[str, Any]:
        """Recursively analyze JSON structure"""
        if max_depth <= 0:
            return {"type": type(data).__name__, "truncated": True}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": {k: self._analyze_json_structure(v, max_depth - 1) for k, v in data.items()}
            }
        elif isinstance(data, list):
            if data:
                return {
                    "type": "array",
                    "length": len(data),
                    "sample_item": self._analyze_json_structure(data[0], max_depth - 1)
                }
            else:
                return {"type": "array", "length": 0}
        else:
            return {"type": type(data).__name__}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            **self.execution_stats,
            'total_tools': len(self.tools),
            'tools_by_category': {
                category.value: len(self.get_tools_by_category(category.value))
                for category in ToolCategory
            },
            'success_rate': (
                self.execution_stats['successful_calls'] / self.execution_stats['total_calls']
                if self.execution_stats['total_calls'] > 0 else 0
            )
        }


# Global function registry instance
_function_registry = None

def get_function_registry() -> FunctionRegistry:
    """Get or create the global function registry instance"""
    global _function_registry
    if _function_registry is None:
        _function_registry = FunctionRegistry()
    return _function_registry


# Export commonly used items
__all__ = [
    'FunctionRegistry',
    'ToolDefinition',
    'FunctionCall',
    'FunctionResult',
    'ToolCategory',
    'FunctionCallingError',
    'ToolNotFoundError', 
    'ToolExecutionError',
    'SecurityError',
    'get_function_registry'
]