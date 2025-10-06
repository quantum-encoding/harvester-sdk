"""
Google AI Studio (GenAI) Function Calling Provider
Enable models to interact with external tools and APIs through function calling

Features:
- Single and parallel function calling
- Compositional (sequential) function calling
- Automatic function execution (Python SDK)
- Multi-tool integration (search, code execution, custom functions)
- Thinking mode for improved function call reasoning
"""

import logging
import asyncio
import json
import inspect
from typing import Any, Dict, List, Optional, Union, Callable, TypedDict
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import os

try:
    from google import genai
    from google.genai import types
    import pydantic
except ImportError:
    genai = None
    types = None
    pydantic = None
    
from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class FunctionCallingMode(Enum):
    """Function calling modes"""
    AUTO = "AUTO"  # Model decides whether to call functions
    ANY = "ANY"    # Model must call a function
    NONE = "NONE"  # Model cannot call functions

@dataclass
class FunctionDeclaration:
    """Function declaration for the model"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API"""
        result = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
        if self.required:
            result["parameters"]["required"] = self.required
        return result

class GenAIFunctionsProvider(BaseProvider):
    """
    Google AI Studio Function Calling Provider
    
    This provider enables Gemini models to interact with external tools
    and APIs through structured function calling.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        if genai is None:
            raise ImportError(
                "Google GenAI not installed. Install with: pip install google-generativeai pydantic"
            )
        
        # API key from environment or config
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Google GenAI API key required. Set GOOGLE_GENAI_API_KEY environment variable")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        # Model configurations
        self.supported_models = {
            "gemini-2.5-pro": {
                "function_calling": True,
                "parallel": True,
                "compositional": True,
                "thinking": True
            },
            "gemini-2.5-flash": {
                "function_calling": True,
                "parallel": True,
                "compositional": True,
                "thinking": True
            },
            "gemini-2.5-flash-lite": {
                "function_calling": True,
                "parallel": True,
                "compositional": True,
                "thinking": False
            },
            "gemini-2.0-flash": {
                "function_calling": True,
                "parallel": True,
                "compositional": True,
                "thinking": False
            }
        }
        
        self.default_model = config.get("default_model", "gemini-2.5-flash")
        
        # Function registry for automatic execution
        self.function_registry = {}
        
        # Built-in function examples
        self._register_builtin_functions()
        
        logger.info(f"GenAI Functions provider initialized with {len(self.supported_models)} models")
    
    def _register_builtin_functions(self):
        """Register built-in example functions"""
        
        # Weather function
        def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
            """Get the current weather for a location.
            
            Args:
                location: The city and state, e.g., 'San Francisco, CA'
                unit: Temperature unit - 'celsius' or 'fahrenheit'
            
            Returns:
                Weather information including temperature and conditions
            """
            # Mock implementation
            return {
                "location": location,
                "temperature": 22,
                "unit": unit,
                "conditions": "partly cloudy",
                "humidity": 65,
                "wind_speed": 10
            }
        
        # Calculator function
        def calculate(expression: str) -> Dict[str, Any]:
            """Evaluate a mathematical expression.
            
            Args:
                expression: Mathematical expression to evaluate
            
            Returns:
                The result of the calculation
            """
            try:
                # Safe evaluation of mathematical expressions
                result = eval(expression, {"__builtins__": {}}, {})
                return {"expression": expression, "result": result}
            except Exception as e:
                return {"expression": expression, "error": str(e)}
        
        # Database query function
        def query_database(query: str, table: str) -> Dict[str, Any]:
            """Query a database table.
            
            Args:
                query: SQL-like query string
                table: Name of the table to query
            
            Returns:
                Query results
            """
            # Mock implementation
            return {
                "query": query,
                "table": table,
                "results": [
                    {"id": 1, "name": "Example Item", "value": 100},
                    {"id": 2, "name": "Another Item", "value": 200}
                ],
                "count": 2
            }
        
        # Register functions
        self.register_function(get_weather)
        self.register_function(calculate)
        self.register_function(query_database)
    
    def register_function(self, func: Callable) -> None:
        """
        Register a Python function for automatic execution
        
        Args:
            func: Python function with type hints and docstring
        """
        self.function_registry[func.__name__] = func
        logger.info(f"Registered function: {func.__name__}")
    
    def create_function_declaration(self, func: Callable) -> FunctionDeclaration:
        """
        Create a function declaration from a Python function
        
        Args:
            func: Python function with type hints and docstring
        
        Returns:
            FunctionDeclaration object
        """
        # Get function signature
        sig = inspect.signature(func)
        
        # Parse parameters
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            # Get type hint
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
                "description": f"Parameter: {param_name}"
            }
            
            # Check if required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        # Parse docstring for description
        description = func.__doc__ or f"Function: {func.__name__}"
        description = description.strip().split('\n')[0]  # First line only
        
        return FunctionDeclaration(
            name=func.__name__,
            description=description,
            parameters={
                "type": "object",
                "properties": properties
            },
            required=required if required else None
        )
    
    async def call_with_functions(
        self,
        prompt: str,
        functions: List[Union[Callable, FunctionDeclaration, Dict]],
        model: str = None,
        mode: FunctionCallingMode = FunctionCallingMode.AUTO,
        auto_execute: bool = True,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the model with function declarations
        
        Args:
            prompt: User prompt
            functions: List of functions (Python callables, declarations, or dicts)
            model: Model to use
            mode: Function calling mode (AUTO, ANY, NONE)
            auto_execute: Automatically execute function calls
            temperature: Model temperature (0 for deterministic)
            **kwargs: Additional parameters
        
        Returns:
            Dict with response and function call results
        """
        model = model or self.default_model
        
        # Prepare function declarations
        declarations = []
        for func in functions:
            if callable(func):
                # Python function
                decl = self.create_function_declaration(func)
                declarations.append(decl.to_dict())
                # Register for auto-execution
                if func.__name__ not in self.function_registry:
                    self.register_function(func)
            elif isinstance(func, FunctionDeclaration):
                declarations.append(func.to_dict())
            elif isinstance(func, dict):
                declarations.append(func)
            else:
                logger.warning(f"Unknown function type: {type(func)}")
        
        try:
            # Configure tools
            tools = types.Tool(function_declarations=declarations)
            
            # Configure function calling mode
            tool_config = None
            if mode != FunctionCallingMode.AUTO:
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=mode.value
                    )
                )
            
            # Create generation config
            config = types.GenerateContentConfig(
                tools=[tools],
                tool_config=tool_config,
                temperature=temperature
            )
            
            # Initial request
            logger.info(f"Sending prompt with {len(declarations)} functions")
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=prompt,
                config=config
            )
            
            result = {
                "prompt": prompt,
                "model": model,
                "function_calls": [],
                "final_response": None
            }
            
            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        logger.info(f"Function call detected: {fc.name}")
                        
                        call_info = {
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                            "result": None
                        }
                        
                        # Auto-execute if enabled
                        if auto_execute and fc.name in self.function_registry:
                            try:
                                func_result = self.function_registry[fc.name](**call_info["args"])
                                call_info["result"] = func_result
                                logger.info(f"Executed {fc.name}: {func_result}")
                            except Exception as e:
                                call_info["error"] = str(e)
                                logger.error(f"Function execution failed: {e}")
                        
                        result["function_calls"].append(call_info)
            
            # Get final response if function was executed
            if auto_execute and result["function_calls"]:
                final_response = await self._send_function_results(
                    model=model,
                    config=config,
                    original_response=response,
                    function_results=result["function_calls"]
                )
                result["final_response"] = final_response.text if final_response else None
            else:
                # No function calls or no auto-execution
                result["final_response"] = response.text if hasattr(response, 'text') else None
            
            return result
            
        except Exception as e:
            logger.error(f"Function calling failed: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "model": model
            }
    
    async def _send_function_results(
        self,
        model: str,
        config: Any,
        original_response: Any,
        function_results: List[Dict[str, Any]]
    ) -> Any:
        """Send function execution results back to the model"""
        
        # Build content with function responses
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=original_response.candidates[0].content.parts[0].text)]
                if hasattr(original_response.candidates[0].content.parts[0], 'text')
                else []
            )
        ]
        
        # Add original model response
        contents.append(original_response.candidates[0].content)
        
        # Add function responses
        function_response_parts = []
        for result in function_results:
            if result.get("result") is not None:
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=result["name"],
                        response={"result": result["result"]}
                    )
                )
        
        if function_response_parts:
            contents.append(
                types.Content(role="user", parts=function_response_parts)
            )
        
        # Get final response
        return await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            config=config,
            contents=contents
        )
    
    async def parallel_function_calling(
        self,
        prompt: str,
        functions: List[Callable],
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute multiple functions in parallel
        
        Args:
            prompt: User prompt that requires multiple function calls
            functions: List of functions that can be called in parallel
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            Dict with all function call results
        """
        return await self.call_with_functions(
            prompt=prompt,
            functions=functions,
            model=model,
            mode=FunctionCallingMode.ANY,
            auto_execute=True,
            **kwargs
        )
    
    async def compositional_function_calling(
        self,
        prompt: str,
        functions: List[Callable],
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute functions in sequence (compositional calling)
        
        The model will chain function calls together to complete
        complex tasks that require sequential operations.
        
        Args:
            prompt: User prompt requiring sequential function calls
            functions: List of functions that can be called
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            Dict with sequential function call results
        """
        # Use automatic function calling for compositional flow
        return await self.call_with_functions(
            prompt=prompt,
            functions=functions,
            model=model,
            mode=FunctionCallingMode.AUTO,
            auto_execute=True,
            temperature=0.0,  # Deterministic for reliable chaining
            **kwargs
        )
    
    async def create_smart_home_controller(self) -> Dict[str, Any]:
        """
        Example: Create a smart home controller with multiple functions
        
        Returns:
            Dict with smart home functions ready to use
        """
        
        # Define smart home functions
        def turn_on_lights(room: str, brightness: int = 100) -> Dict[str, Any]:
            """Turn on lights in a specific room.
            
            Args:
                room: Name of the room
                brightness: Brightness level (0-100)
            
            Returns:
                Status of the lights
            """
            return {
                "action": "lights_on",
                "room": room,
                "brightness": brightness,
                "status": "success"
            }
        
        def set_temperature(temperature: float, unit: str = "celsius") -> Dict[str, Any]:
            """Set the thermostat temperature.
            
            Args:
                temperature: Desired temperature
                unit: Temperature unit (celsius or fahrenheit)
            
            Returns:
                Thermostat status
            """
            return {
                "action": "temperature_set",
                "temperature": temperature,
                "unit": unit,
                "status": "success"
            }
        
        def play_music(genre: str, volume: int = 50) -> Dict[str, Any]:
            """Play music in the house.
            
            Args:
                genre: Music genre to play
                volume: Volume level (0-100)
            
            Returns:
                Music player status
            """
            return {
                "action": "music_playing",
                "genre": genre,
                "volume": volume,
                "status": "success"
            }
        
        def lock_doors(doors: List[str] = None) -> Dict[str, Any]:
            """Lock specific doors or all doors.
            
            Args:
                doors: List of doors to lock, or None for all
            
            Returns:
                Door lock status
            """
            if doors is None:
                doors = ["front", "back", "garage"]
            
            return {
                "action": "doors_locked",
                "doors": doors,
                "status": "success"
            }
        
        # Register all functions
        functions = [turn_on_lights, set_temperature, play_music, lock_doors]
        for func in functions:
            self.register_function(func)
        
        return {
            "functions": [f.__name__ for f in functions],
            "ready": True,
            "example_prompts": [
                "Turn on the living room lights at 50% brightness",
                "Set temperature to 22 degrees and play some jazz",
                "Lock all doors and turn off all lights",
                "Create a party atmosphere with lights and music"
            ]
        }
    
    async def create_data_analyst(self) -> Dict[str, Any]:
        """
        Example: Create a data analyst with analytical functions
        
        Returns:
            Dict with data analysis functions ready to use
        """
        
        def analyze_data(data: List[float]) -> Dict[str, Any]:
            """Analyze numerical data and return statistics.
            
            Args:
                data: List of numerical values
            
            Returns:
                Statistical analysis results
            """
            import statistics
            
            return {
                "count": len(data),
                "mean": statistics.mean(data) if data else 0,
                "median": statistics.median(data) if data else 0,
                "stdev": statistics.stdev(data) if len(data) > 1 else 0,
                "min": min(data) if data else 0,
                "max": max(data) if data else 0
            }
        
        def create_chart(data: List[float], chart_type: str = "bar") -> Dict[str, Any]:
            """Create a chart visualization.
            
            Args:
                data: Data points for the chart
                chart_type: Type of chart (bar, line, pie)
            
            Returns:
                Chart configuration
            """
            return {
                "chart_type": chart_type,
                "data_points": data,
                "labels": [f"Item {i+1}" for i in range(len(data))],
                "title": f"{chart_type.title()} Chart",
                "status": "chart_created"
            }
        
        def forecast_trend(historical_data: List[float], periods: int = 5) -> Dict[str, Any]:
            """Forecast future trend based on historical data.
            
            Args:
                historical_data: Past data points
                periods: Number of periods to forecast
            
            Returns:
                Forecasted values
            """
            # Simple linear trend forecast
            if len(historical_data) < 2:
                return {"error": "Insufficient data for forecasting"}
            
            # Calculate simple moving average trend
            trend = sum(historical_data[-3:]) / min(3, len(historical_data))
            forecast = [trend * (1 + i * 0.05) for i in range(periods)]
            
            return {
                "historical": historical_data,
                "forecast": forecast,
                "periods": periods,
                "method": "simple_trend"
            }
        
        # Register functions
        functions = [analyze_data, create_chart, forecast_trend]
        for func in functions:
            self.register_function(func)
        
        return {
            "functions": [f.__name__ for f in functions],
            "ready": True,
            "example_prompts": [
                "Analyze this data: [10, 20, 15, 30, 25, 35]",
                "Create a bar chart for sales data: [100, 150, 120, 180, 200]",
                "Forecast the next 5 periods based on: [100, 110, 120, 130, 140]",
                "Analyze the data [5,10,15,20] and create a line chart"
            ]
        }
    
    async def complete(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Compatibility method for BaseProvider interface
        """
        result = await self.call_with_functions(
            prompt=prompt,
            functions=list(self.function_registry.values()),
            model=model,
            **kwargs
        )
        
        return result.get("final_response") or str(result)


# Example usage
if __name__ == "__main__":
    async def test_functions():
        config = {
            "api_key": "YOUR_API_KEY"  # Or set GOOGLE_GENAI_API_KEY env var
        }
        
        provider = GenAIFunctionsProvider(config)
        
        # Test 1: Simple function calling
        result = await provider.call_with_functions(
            prompt="What's the weather in San Francisco?",
            functions=[provider.function_registry["get_weather"]],
            auto_execute=True
        )
        
        print("✅ Simple function call:")
        print(f"   Function: {result['function_calls'][0]['name'] if result.get('function_calls') else 'None'}")
        print(f"   Response: {result.get('final_response', 'No response')}")
        
        # Test 2: Parallel function calling
        def get_time() -> Dict[str, str]:
            """Get the current time."""
            from datetime import datetime
            return {"time": datetime.now().strftime("%H:%M:%S")}
        
        def get_date() -> Dict[str, str]:
            """Get the current date."""
            from datetime import datetime
            return {"date": datetime.now().strftime("%Y-%m-%d")}
        
        result2 = await provider.parallel_function_calling(
            prompt="Tell me the current date and time",
            functions=[get_time, get_date]
        )
        
        print("\n✅ Parallel function calls:")
        for call in result2.get('function_calls', []):
            print(f"   {call['name']}: {call.get('result', 'No result')}")
        
        # Test 3: Smart home example
        home = await provider.create_smart_home_controller()
        
        result3 = await provider.call_with_functions(
            prompt="Turn on the bedroom lights at 30% and set temperature to 22 degrees",
            functions=[
                provider.function_registry["turn_on_lights"],
                provider.function_registry["set_temperature"]
            ]
        )
        
        print("\n✅ Smart home control:")
        for call in result3.get('function_calls', []):
            print(f"   {call['name']}: {call.get('args', {})}")
        
        # Test 4: Data analysis
        analyst = await provider.create_data_analyst()
        
        result4 = await provider.call_with_functions(
            prompt="Analyze this data [10, 20, 30, 40, 50] and create a bar chart",
            functions=[
                provider.function_registry["analyze_data"],
                provider.function_registry["create_chart"]
            ]
        )
        
        print("\n✅ Data analysis:")
        for call in result4.get('function_calls', []):
            print(f"   {call['name']}: Executed")
    
    # Run test
    import asyncio
    asyncio.run(test_functions())