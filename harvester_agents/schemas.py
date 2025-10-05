"""Agent output schemas and function schemas for tool definitions."""

import json
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from dataclasses import dataclass
from inspect import Signature
from enum import Enum

try:
    from pydantic import BaseModel, create_model
    from pydantic.json_schema import JsonSchemaValue
    HAS_PYDANTIC = True
except ImportError:
    BaseModel = object  # type: ignore
    HAS_PYDANTIC = False


class DocstringStyle(str, Enum):
    """Supported docstring styles."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    AUTO = "auto"


class ModelBehaviorError(Exception):
    """Raised when the model produces invalid output."""
    pass


# ============================================================================
# Agent Output Schema
# ============================================================================


class AgentOutputSchemaBase(ABC):
    """An object that captures the JSON schema of the output.

    Also handles validating/parsing JSON produced by the LLM into the output type.
    """

    @abstractmethod
    def is_plain_text(self) -> bool:
        """Whether the output type is plain text (versus a JSON object)."""
        pass

    @abstractmethod
    def name(self) -> str:
        """The name of the output type."""
        pass

    @abstractmethod
    def json_schema(self) -> dict[str, Any]:
        """Returns the JSON schema of the output.

        Will only be called if the output type is not plain text.
        """
        pass

    @abstractmethod
    def is_strict_json_schema(self) -> bool:
        """Whether the JSON schema is in strict mode.

        Strict mode constrains the JSON schema features, but guarantees valid JSON.
        See: https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
        """
        pass

    @abstractmethod
    def validate_json(self, json_str: str) -> Any:
        """Validate a JSON string against the output type.

        You must return the validated object, or raise a ModelBehaviorError if the JSON is invalid.

        Args:
            json_str: The JSON string to validate.

        Returns:
            The validated object.

        Raises:
            ModelBehaviorError: If the JSON is invalid.
        """
        pass


@dataclass
class AgentOutputSchema(AgentOutputSchemaBase):
    """An object that captures the JSON schema of the output.

    Also handles validating/parsing JSON produced by the LLM into the output type.
    """

    output_type: type[Any]
    """The type of the output."""

    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode."""

    def __init__(self, output_type: type[Any], strict_json_schema: bool = True):
        """Initialize the AgentOutputSchema.

        Args:
            output_type: The type of the output.
            strict_json_schema: Whether the JSON schema is in strict mode. We strongly
                recommend setting this to True, as it increases the likelihood of correct
                JSON input.
        """
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema
        self._schema_cache: Optional[dict] = None

    def is_plain_text(self) -> bool:
        """Whether the output type is plain text (versus a JSON object)."""
        return self.output_type is str

    def is_strict_json_schema(self) -> bool:
        """Whether the JSON schema is in strict mode."""
        return self.strict_json_schema

    def json_schema(self) -> dict[str, Any]:
        """The JSON schema of the output type."""
        if self._schema_cache is not None:
            return self._schema_cache

        if not HAS_PYDANTIC:
            raise ImportError("Pydantic is required for JSON schema generation")

        if hasattr(self.output_type, "model_json_schema"):
            # Pydantic v2
            schema = self.output_type.model_json_schema()
        elif hasattr(self.output_type, "schema"):
            # Pydantic v1
            schema = self.output_type.schema()
        else:
            # Fallback for basic types
            schema = {"type": "object", "properties": {}}

        self._schema_cache = schema
        return schema

    def validate_json(self, json_str: str) -> Any:
        """Validate a JSON string against the output type.

        Returns the validated object, or raises a ModelBehaviorError if the JSON is invalid.

        Args:
            json_str: The JSON string to validate.

        Returns:
            The validated object.

        Raises:
            ModelBehaviorError: If the JSON is invalid.
        """
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ModelBehaviorError(f"Invalid JSON: {e}")

        if not HAS_PYDANTIC:
            return json_data

        try:
            if hasattr(self.output_type, "model_validate"):
                # Pydantic v2
                return self.output_type.model_validate(json_data)
            elif hasattr(self.output_type, "parse_obj"):
                # Pydantic v1
                return self.output_type.parse_obj(json_data)
            else:
                return json_data
        except Exception as e:
            raise ModelBehaviorError(f"Validation failed: {e}")

    def name(self) -> str:
        """The name of the output type."""
        if hasattr(self.output_type, "__name__"):
            return self.output_type.__name__
        return str(self.output_type)


# ============================================================================
# Function Schema
# ============================================================================


@dataclass
class FuncDocumentation:
    """Contains metadata about a Python function, extracted from its docstring."""

    name: str
    """The name of the function, via __name__."""

    description: Optional[str]
    """The description of the function, derived from the docstring."""

    param_descriptions: Optional[dict[str, str]]
    """The parameter descriptions of the function, derived from the docstring."""


@dataclass
class FuncSchema:
    """Captures the schema for a python function.

    This is used to prepare it for sending to an LLM as a tool.
    """

    name: str
    """The name of the function."""

    description: Optional[str]
    """The description of the function."""

    params_pydantic_model: type[BaseModel]
    """A Pydantic model that represents the function's parameters."""

    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters, derived from the Pydantic model."""

    signature: Signature
    """The signature of the function."""

    takes_context: bool = False
    """Whether the function takes a RunContextWrapper argument (must be the first argument)."""

    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode. We strongly recommend setting this to True,
    as it increases the likelihood of correct JSON input."""

    def to_call_args(
        self,
        data: BaseModel,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Converts validated data from the Pydantic model into (args, kwargs).

        This is suitable for calling the original function.

        Args:
            data: The validated Pydantic model instance.

        Returns:
            A tuple of (args, kwargs) for calling the function.
        """
        if hasattr(data, "model_dump"):
            # Pydantic v2
            kwargs = data.model_dump()
        elif hasattr(data, "dict"):
            # Pydantic v1
            kwargs = data.dict()
        else:
            kwargs = {}

        return [], kwargs


def generate_func_documentation(
    func: Callable[..., Any],
    style: Optional[DocstringStyle] = None,
) -> FuncDocumentation:
    """Extracts metadata from a function docstring.

    This is in preparation for sending it to an LLM as a tool.

    Args:
        func: The function to extract documentation from.
        style: The style of the docstring to use for parsing. If not provided,
            we will attempt to auto-detect the style.

    Returns:
        A FuncDocumentation object containing the function's name, description, and
        parameter descriptions.
    """
    docstring = inspect.getdoc(func)
    name = func.__name__

    if not docstring:
        return FuncDocumentation(
            name=name,
            description=None,
            param_descriptions=None,
        )

    # Simple extraction - just use the first line as description
    lines = docstring.strip().split('\n')
    description = lines[0] if lines else None

    # Extract parameter descriptions (simple Google-style parsing)
    param_descriptions = {}
    in_args_section = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ('args:', 'arguments:', 'parameters:'):
            in_args_section = True
            continue

        if in_args_section:
            if stripped and not stripped.startswith(' '):
                in_args_section = False
            elif ':' in stripped:
                parts = stripped.split(':', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    param_descriptions[param_name] = param_desc

    return FuncDocumentation(
        name=name,
        description=description,
        param_descriptions=param_descriptions if param_descriptions else None,
    )


def function_schema(
    func: Callable[..., Any],
    docstring_style: Optional[DocstringStyle] = None,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """Given a Python function, extracts a FuncSchema from it.

    This captures the name, description, parameter descriptions, and other metadata.

    Args:
        func: The function to extract the schema from.
        docstring_style: The style of the docstring to use for parsing. If not provided,
            we will attempt to auto-detect the style.
        name_override: If provided, use this name instead of the function's __name__.
        description_override: If provided, use this description instead of the one
            derived from the docstring.
        use_docstring_info: If True, uses the docstring to generate the description
            and parameter descriptions.
        strict_json_schema: Whether the JSON schema is in strict mode. If True, we'll
            ensure that the schema adheres to the "strict" standard the OpenAI API expects.
            We strongly recommend setting this to True, as it increases the likelihood of
            the LLM producing correct JSON input.

    Returns:
        A FuncSchema object containing the function's name, description, parameter
        descriptions, and other metadata.
    """
    if not HAS_PYDANTIC:
        raise ImportError("Pydantic is required for function schema generation")

    sig = inspect.signature(func)
    name = name_override or func.__name__

    # Extract documentation
    doc = None
    if use_docstring_info:
        doc = generate_func_documentation(func, docstring_style)

    description = description_override or (doc.description if doc else None)

    # Build Pydantic model for parameters
    fields = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # Get parameter type
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any

        # Get default value
        default = ... if param.default == inspect.Parameter.empty else param.default

        # Get description from docstring
        field_desc = None
        if doc and doc.param_descriptions:
            field_desc = doc.param_descriptions.get(param_name)

        fields[param_name] = (param_type, default)

    # Create Pydantic model
    if HAS_PYDANTIC:
        params_model = create_model(f"{name}Params", **fields)

        if hasattr(params_model, "model_json_schema"):
            # Pydantic v2
            params_schema = params_model.model_json_schema()
        elif hasattr(params_model, "schema"):
            # Pydantic v1
            params_schema = params_model.schema()
        else:
            params_schema = {"type": "object", "properties": {}}
    else:
        params_model = None  # type: ignore
        params_schema = {"type": "object", "properties": {}}

    return FuncSchema(
        name=name,
        description=description,
        params_pydantic_model=params_model,
        params_json_schema=params_schema,
        signature=sig,
        takes_context=False,
        strict_json_schema=strict_json_schema,
    )
