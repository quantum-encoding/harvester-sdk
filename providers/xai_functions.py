"""
xAI Function Calling Module
Enable models to use external tools and systems
"""
import json
import logging
import asyncio
from typing import Any, Dict, List, Union, Optional, Callable, Literal
from dataclasses import dataclass, field
from datetime import datetime
import os
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import inspect

logger = logging.getLogger(__name__)

@dataclass
class ToolDefinition:
    """Tool definition for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

@dataclass
class ToolCall:
    """Represents a tool call request from the model"""
    id: str
    name: str
    arguments: Dict[str, Any]
    
@dataclass
class ToolResult:
    """Result from executing a tool"""
    tool_call_id: str
    content: Any
    error: Optional[str] = None
    
    def to_message(self) -> Dict[str, Any]:
        """Convert to message format for API"""
        content = json.dumps(self.content) if not isinstance(self.content, str) else self.content
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": self.tool_call_id
        }

class XaiFunctionCaller:
    """
    Function calling handler for xAI models.
    Manages tool definitions, executions, and conversation flow.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 300,
        auto_execute: bool = True
    ):
        """
        Initialize function caller.
        
        Args:
            api_key: xAI API key
            timeout: Request timeout
            auto_execute: Automatically execute tool calls
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key required")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
            timeout=timeout
        )
        
        self.auto_execute = auto_execute
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_functions: Dict[str, Callable] = {}
        
        logger.info("Function caller initialized")
    
    def register_tool(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Union[Dict, BaseModel]] = None
    ) -> ToolDefinition:
        """
        Register a tool for function calling.
        
        Args:
            name: Tool name
            description: Tool description
            function: Python function to execute
            parameters: Parameters schema (dict or Pydantic model)
            
        Returns:
            ToolDefinition
        """
        # Auto-generate parameters from function signature if not provided
        if parameters is None:
            parameters = self._generate_parameters_from_function(function)
        elif isinstance(parameters, type) and issubclass(parameters, BaseModel):
            # Convert Pydantic model to JSON schema
            parameters = parameters.model_json_schema()
        
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        
        self.tools[name] = tool
        self.tool_functions[name] = function
        
        logger.info(f"Registered tool: {name}")
        return tool
    
    def _generate_parameters_from_function(self, func: Callable) -> Dict[str, Any]:
        """Generate parameter schema from function signature"""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Determine type
            param_type = "string"  # Default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            # Check if required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def register_pydantic_tool(
        self,
        name: str,
        description: str,
        function: Callable,
        model: BaseModel
    ):
        """
        Register a tool using Pydantic model for parameters.
        
        Args:
            name: Tool name
            description: Tool description
            function: Function that accepts Pydantic model instance
            model: Pydantic model class
        """
        parameters = model.model_json_schema()
        
        # Wrap function to accept dict and convert to Pydantic
        def wrapped_function(**kwargs):
            instance = model(**kwargs)
            return function(instance)
        
        self.register_tool(name, description, wrapped_function, parameters)
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.
        
        Args:
            tool_call: Tool call from model
            
        Returns:
            ToolResult with execution result or error
        """
        try:
            if tool_call.name not in self.tool_functions:
                raise ValueError(f"Unknown tool: {tool_call.name}")
            
            func = self.tool_functions[tool_call.name]
            
            # Execute function (handle both sync and async)
            if asyncio.iscoroutinefunction(func):
                result = await func(**tool_call.arguments)
            else:
                result = func(**tool_call.arguments)
            
            return ToolResult(
                tool_call_id=tool_call.id,
                content=result
            )
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content={"error": str(e)},
                error=str(e)
            )
    
    async def process_conversation(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-4",
        tool_choice: Union[str, Dict] = "auto",
        parallel_function_calling: bool = True,
        max_iterations: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a conversation with function calling.
        
        Args:
            messages: Conversation history
            model: Model to use
            tool_choice: "auto", "required", "none", or specific function
            parallel_function_calling: Enable parallel function calls
            max_iterations: Maximum tool call iterations
            **kwargs: Additional parameters
            
        Returns:
            Final response with conversation history
        """
        conversation_messages = messages.copy()
        iterations = 0
        
        # Get tool definitions
        tools = [tool.to_dict() for tool in self.tools.values()] if self.tools else None
        
        while iterations < max_iterations:
            # Make API call
            request_params = {
                "model": model,
                "messages": conversation_messages,
                **kwargs
            }
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = tool_choice
                request_params["parallel_tool_calls"] = parallel_function_calling
            
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract message
            message = response.choices[0].message
            
            # Add assistant message to history
            assistant_msg = {"role": "assistant", "content": message.content}
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
                
                conversation_messages.append(assistant_msg)
                
                if self.auto_execute:
                    # Execute tool calls
                    for tool_call in message.tool_calls:
                        tc = ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=json.loads(tool_call.function.arguments)
                        )
                        
                        result = await self.execute_tool(tc)
                        conversation_messages.append(result.to_message())
                    
                    iterations += 1
                    continue  # Continue conversation
                else:
                    # Return tool calls for manual execution
                    return {
                        "response": message.content,
                        "tool_calls": message.tool_calls,
                        "messages": conversation_messages,
                        "requires_tool_execution": True
                    }
            else:
                # No tool calls, conversation complete
                conversation_messages.append(assistant_msg)
                # Convert usage to dict safely
                usage_dict = None
                if response.usage:
                    try:
                        if hasattr(response.usage, 'model_dump'):
                            usage_dict = response.usage.model_dump()
                        elif hasattr(response.usage, '_asdict'):
                            usage_dict = response.usage._asdict()
                        else:
                            usage_dict = {
                                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                                'total_tokens': getattr(response.usage, 'total_tokens', 0)
                            }
                    except Exception:
                        usage_dict = None
                
                return {
                    "response": message.content,
                    "messages": conversation_messages,
                    "usage": usage_dict
                }
        
        raise Exception(f"Maximum iterations ({max_iterations}) reached")
    
    async def simple_query(
        self,
        prompt: str,
        model: str = "grok-4",
        **kwargs
    ) -> str:
        """
        Simple query with automatic function calling.
        
        Args:
            prompt: User prompt
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Final response text
        """
        messages = [{"role": "user", "content": prompt}]
        result = await self.process_conversation(messages, model, **kwargs)
        return result["response"]

# Utility functions for common tools

def create_web_search_tool():
    """Create a web search tool definition"""
    def web_search(query: str, max_results: int = 5):
        # This would normally call a search API
        return {
            "query": query,
            "results": [f"Result {i+1} for '{query}'" for i in range(max_results)]
        }
    
    return ToolDefinition(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum results", "default": 5}
            },
            "required": ["query"]
        },
        function=web_search
    )

def create_calculator_tool():
    """Create a calculator tool definition"""
    def calculate(expression: str):
        try:
            # Safe evaluation of mathematical expressions
            result = eval(expression, {"__builtins__": {}}, {})
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}
    
    return ToolDefinition(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression"}
            },
            "required": ["expression"]
        },
        function=calculate
    )

# Example Pydantic models for tools

class WeatherRequest(BaseModel):
    """Weather request parameters"""
    location: str = Field(description="City and state, e.g. San Francisco, CA")
    unit: Literal["celsius", "fahrenheit"] = Field("fahrenheit", description="Temperature unit")

class DatabaseQuery(BaseModel):
    """Database query parameters"""
    table: str = Field(description="Table name")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    limit: int = Field(10, description="Result limit")

# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize function caller
        caller = XaiFunctionCaller()
        
        # Register tools
        def get_weather(request: WeatherRequest):
            """Get weather information"""
            temp = 72 if request.unit == "fahrenheit" else 22
            return {
                "location": request.location,
                "temperature": temp,
                "unit": request.unit,
                "conditions": "sunny"
            }
        
        def get_time(timezone: str = "UTC"):
            """Get current time"""
            from datetime import datetime
            import pytz
            tz = pytz.timezone(timezone)
            return {
                "timezone": timezone,
                "time": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Register with Pydantic model
        caller.register_pydantic_tool(
            name="get_weather",
            description="Get current weather",
            function=get_weather,
            model=WeatherRequest
        )
        
        # Register with auto-generated parameters
        caller.register_tool(
            name="get_time",
            description="Get current time in timezone",
            function=get_time
        )
        
        # Example 1: Simple query
        print("=== Simple Query ===")
        response = await caller.simple_query(
            "What's the weather like in San Francisco and what time is it there?"
        )
        print(response)
        
        # Example 2: Manual tool execution
        print("\n=== Manual Execution ===")
        caller.auto_execute = False
        
        messages = [
            {"role": "user", "content": "Calculate 15 * 73 + 42"}
        ]
        
        # Register calculator
        calc_tool = create_calculator_tool()
        caller.tools["calculate"] = calc_tool
        caller.tool_functions["calculate"] = calc_tool.function
        
        result = await caller.process_conversation(messages)
        
        if result.get("requires_tool_execution"):
            print(f"Tool calls required: {result['tool_calls']}")
            
            # Manually execute and continue
            for tool_call in result['tool_calls']:
                tc = ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                )
                tool_result = await caller.execute_tool(tc)
                messages.append(tool_result.to_message())
            
            # Continue conversation
            final_result = await caller.process_conversation(messages)
            print(f"Final response: {final_result['response']}")
        
        # Example 3: Parallel function calling
        print("\n=== Parallel Functions ===")
        caller.auto_execute = True
        
        response = await caller.simple_query(
            "Get the weather in NYC and SF, and tell me the time in both cities",
            parallel_function_calling=True
        )
        print(response)
    
    asyncio.run(example())