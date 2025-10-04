"""OpenAI model implementations for Responses API and Chat Completions API."""

import json
import logging
from typing import Any, AsyncIterator, Literal, cast, overload
from openai import AsyncOpenAI, NOT_GIVEN, AsyncStream, APIStatusError
from openai.types.responses import (
    Response,
    ResponseStreamEvent,
    ResponseCompletedEvent,
)
from openai.types.chat.chat_model import ChatModel
from openai.types.responses.response_includable import ResponseIncludable
from openai.types.responses.response_prompt_param import ResponsePromptParam


logger = logging.getLogger(__name__)


class ModelSettings:
    """Settings to use when calling an LLM.

    This class holds optional model configuration parameters (e.g. temperature, top_p,
    penalties, truncation, etc.).

    Not all models/providers support all of these parameters, so please check the API
    documentation for the specific model and provider you are using.
    """

    def __init__(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: str | dict | None = None,
        response_include: list[str] | None = None,
        truncation: str | None = None,
        top_logprobs: int | None = None,
        verbosity: str | None = None,
        store: bool | None = None,
        include_usage: bool | None = None,
        reasoning: dict | None = None,
        metadata: dict[str, str] | None = None,
        extra_headers: dict | None = None,
        extra_query: dict | None = None,
        extra_body: dict | None = None,
        extra_args: dict[str, Any] | None = None,
    ):
        """Initialize ModelSettings.

        Args:
            temperature: The temperature to use when calling the model.
            top_p: The top_p to use when calling the model.
            frequency_penalty: The frequency penalty to use when calling the model.
            presence_penalty: The presence penalty to use when calling the model.
            max_tokens: The maximum number of output tokens to generate.
            parallel_tool_calls: Controls whether the model can make multiple parallel tool
                calls in a single turn. If not provided (i.e., set to None), this behavior
                defers to the underlying model provider's default.
            tool_choice: The tool choice to use when calling the model.
            response_include: Additional output data to include in the model response.
            truncation: The truncation strategy to use when calling the model ('auto' or 'disabled').
            top_logprobs: Number of top tokens to return logprobs for. Setting this will
                automatically include "message.output_text.logprobs" in the response.
            verbosity: Constrains the verbosity of the model's response ('low', 'medium', 'high').
            store: Whether to store the generated model response for later retrieval.
            include_usage: Whether to include usage chunk. Only available for Chat Completions API.
            reasoning: Configuration options for reasoning models.
            metadata: Metadata to include with the model response call.
            extra_headers: Additional headers to provide with the request.
            extra_query: Additional query fields to provide with the request.
            extra_body: Additional body fields to provide with the request.
            extra_args: Arbitrary keyword arguments to pass to the model API call.
        """
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_choice = tool_choice
        self.response_include = response_include
        self.truncation = truncation
        self.top_logprobs = top_logprobs
        self.verbosity = verbosity
        self.store = store
        self.include_usage = include_usage
        self.reasoning = reasoning
        self.metadata = metadata
        self.extra_headers = extra_headers
        self.extra_query = extra_query
        self.extra_body = extra_body
        self.extra_args = extra_args

    def resolve(self, override: "ModelSettings | None") -> "ModelSettings":
        """Produce a new ModelSettings by overlaying any non-None values from the override.

        Args:
            override: ModelSettings to overlay on top of this instance.

        Returns:
            A new ModelSettings instance with merged values.
        """
        if override is None:
            return self

        return ModelSettings(
            temperature=override.temperature if override.temperature is not None else self.temperature,
            top_p=override.top_p if override.top_p is not None else self.top_p,
            frequency_penalty=override.frequency_penalty if override.frequency_penalty is not None else self.frequency_penalty,
            presence_penalty=override.presence_penalty if override.presence_penalty is not None else self.presence_penalty,
            max_tokens=override.max_tokens if override.max_tokens is not None else self.max_tokens,
            parallel_tool_calls=override.parallel_tool_calls if override.parallel_tool_calls is not None else self.parallel_tool_calls,
            tool_choice=override.tool_choice if override.tool_choice is not None else self.tool_choice,
            response_include=override.response_include if override.response_include is not None else self.response_include,
            truncation=override.truncation if override.truncation is not None else self.truncation,
            top_logprobs=override.top_logprobs if override.top_logprobs is not None else self.top_logprobs,
            verbosity=override.verbosity if override.verbosity is not None else self.verbosity,
            store=override.store if override.store is not None else self.store,
            include_usage=override.include_usage if override.include_usage is not None else self.include_usage,
            reasoning=override.reasoning if override.reasoning is not None else self.reasoning,
            metadata=override.metadata if override.metadata is not None else self.metadata,
            extra_headers=override.extra_headers if override.extra_headers is not None else self.extra_headers,
            extra_query=override.extra_query if override.extra_query is not None else self.extra_query,
            extra_body=override.extra_body if override.extra_body is not None else self.extra_body,
            extra_args=override.extra_args if override.extra_args is not None else self.extra_args,
        )


class Usage:
    """Token usage information."""

    def __init__(
        self,
        requests: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        input_tokens_details: dict | None = None,
        output_tokens_details: dict | None = None,
    ):
        self.requests = requests
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.input_tokens_details = input_tokens_details
        self.output_tokens_details = output_tokens_details


class ModelResponse:
    """Response from a model."""

    def __init__(self, output: list, usage: Usage, response_id: str | None = None):
        self.output = output
        self.usage = usage
        self.response_id = response_id


class ModelTracing:
    """Tracing configuration for models."""

    def __init__(self, disabled: bool = False, include_data: bool = True):
        self._disabled = disabled
        self._include_data = include_data

    def is_disabled(self) -> bool:
        return self._disabled

    def include_data(self) -> bool:
        return self._include_data


class Tool:
    """Represents a tool that can be used by the model."""

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters


class Handoff:
    """Represents a handoff to another agent."""

    def __init__(self, target: str, description: str | None = None):
        self.target = target
        self.description = description


class AgentOutputSchemaBase:
    """Base class for output schemas."""

    pass


class Converter:
    """Converter utilities for tool and response format conversion."""

    @staticmethod
    def convert_tool_choice(tool_choice: str | dict | None) -> str | dict:
        """Convert tool choice to API format."""
        if tool_choice is None:
            return NOT_GIVEN
        return tool_choice

    @staticmethod
    def convert_tools(tools: list[Tool], handoffs: list[Handoff]) -> dict:
        """Convert tools and handoffs to API format."""
        converted = []
        includes = []

        for tool in tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })

        for handoff in handoffs:
            converted.append({
                "type": "handoff",
                "handoff": {
                    "target": handoff.target,
                    "description": handoff.description or "",
                }
            })

        return {"tools": converted, "includes": includes}

    @staticmethod
    def get_response_format(output_schema: AgentOutputSchemaBase | None) -> dict | str:
        """Get response format from output schema."""
        if output_schema is None:
            return NOT_GIVEN
        # Implement schema conversion logic
        return NOT_GIVEN


class ItemHelpers:
    """Helper utilities for input item conversion."""

    @staticmethod
    def input_to_new_input_list(input: str | list) -> list:
        """Convert input to list format."""
        if isinstance(input, str):
            return [{"type": "input_text", "text": input}]
        return input


def _to_dump_compatible(data: Any) -> Any:
    """Convert data to JSON-serializable format."""
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if isinstance(data, dict):
        return {k: _to_dump_compatible(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_dump_compatible(item) for item in data]
    return data


class _Debug:
    """Debug configuration."""

    DONT_LOG_MODEL_DATA = False


_debug = _Debug()
_HEADERS = {}
_HEADERS_OVERRIDE = type('ContextVar', (), {'get': lambda self: None})()


class SpanError:
    """Error information for tracing spans."""

    def __init__(self, message: str, data: dict | None = None):
        self.message = message
        self.data = data or {}


class ResponseSpan:
    """Context manager for response tracing."""

    def __init__(self, disabled: bool = False):
        self.disabled = disabled
        self.span_data = type('SpanData', (), {'response': None, 'input': None})()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_error(self, error: SpanError):
        """Set error information."""
        logger.error(f"{error.message}: {error.data}")


def response_span(disabled: bool = False) -> ResponseSpan:
    """Create a response tracing span."""
    return ResponseSpan(disabled=disabled)


# ============================================================================
# OpenAI Responses Model (for GPT-5 and newer models)
# ============================================================================


class OpenAIResponsesModel:
    """Implementation of Model that uses the OpenAI Responses API.

    This is the newer API used by GPT-5 and latest OpenAI models.
    """

    def __init__(
        self,
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = openai_client

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value if value is not None else NOT_GIVEN

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        """Get a complete response from the model."""
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                response = await self._fetch_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    stream=False,
                    prompt=prompt,
                )

                if _debug.DONT_LOG_MODEL_DATA:
                    logger.debug("LLM responded")
                else:
                    logger.debug(
                        "LLM resp:\n"
                        f"""{
                            json.dumps(
                                [x.model_dump() for x in response.output],
                                indent=2,
                                ensure_ascii=False,
                            )
                        }\n"""
                    )

                usage = (
                    Usage(
                        requests=1,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.total_tokens,
                        input_tokens_details=response.usage.input_tokens_details,
                        output_tokens_details=response.usage.output_tokens_details,
                    )
                    if response.usage
                    else Usage()
                )

                if tracing.include_data():
                    span_response.span_data.response = response
                    span_response.span_data.input = input
            except Exception as e:
                span_response.set_error(
                    SpanError(
                        message="Error getting response",
                        data={
                            "error": str(e) if tracing.include_data() else e.__class__.__name__,
                        },
                    )
                )
                request_id = e.request_id if isinstance(e, APIStatusError) else None
                logger.error(f"Error getting response: {e}. (request_id: {request_id})")
                raise

        return ModelResponse(
            output=response.output,
            usage=usage,
            response_id=response.id,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Yields a partial message as it is generated, as well as the usage information."""
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                stream = await self._fetch_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    stream=True,
                    prompt=prompt,
                )

                final_response: Response | None = None

                async for chunk in stream:
                    if isinstance(chunk, ResponseCompletedEvent):
                        final_response = chunk.response
                    yield chunk

                if final_response and tracing.include_data():
                    span_response.span_data.response = final_response
                    span_response.span_data.input = input

            except Exception as e:
                span_response.set_error(
                    SpanError(
                        message="Error streaming response",
                        data={
                            "error": str(e) if tracing.include_data() else e.__class__.__name__,
                        },
                    )
                )
                logger.error(f"Error streaming response: {e}")
                raise

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None,
        conversation_id: str | None,
        stream: Literal[True],
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncStream[ResponseStreamEvent]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None,
        conversation_id: str | None,
        stream: Literal[False],
        prompt: ResponsePromptParam | None = None,
    ) -> Response: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        stream: Literal[True] | Literal[False] = False,
        prompt: ResponsePromptParam | None = None,
    ) -> Response | AsyncStream[ResponseStreamEvent]:
        """Internal method to fetch response from OpenAI Responses API."""
        list_input = ItemHelpers.input_to_new_input_list(input)
        list_input = _to_dump_compatible(list_input)

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else NOT_GIVEN
        )

        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        converted_tools = Converter.convert_tools(tools, handoffs)
        converted_tools_payload = _to_dump_compatible(converted_tools["tools"])
        response_format = Converter.get_response_format(output_schema)

        include_set: set[str] = set(converted_tools["includes"])
        if model_settings.response_include is not None:
            include_set.update(model_settings.response_include)
        if model_settings.top_logprobs is not None:
            include_set.add("message.output_text.logprobs")
        include = cast(list[ResponseIncludable], list(include_set))

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            input_json = json.dumps(list_input, indent=2, ensure_ascii=False)
            tools_json = json.dumps(converted_tools_payload, indent=2, ensure_ascii=False)
            logger.debug(
                f"Calling LLM {self.model} with input:\n"
                f"{input_json}\n"
                f"Tools:\n{tools_json}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
                f"Previous response id: {previous_response_id}\n"
                f"Conversation id: {conversation_id}\n"
            )

        extra_args = dict(model_settings.extra_args or {})
        if model_settings.top_logprobs is not None:
            extra_args["top_logprobs"] = model_settings.top_logprobs
        if model_settings.verbosity is not None:
            if response_format != NOT_GIVEN:
                response_format["verbosity"] = model_settings.verbosity  # type: ignore
            else:
                response_format = {"verbosity": model_settings.verbosity}

        return await self._client.responses.create(
            previous_response_id=self._non_null_or_not_given(previous_response_id),
            conversation=self._non_null_or_not_given(conversation_id),
            instructions=self._non_null_or_not_given(system_instructions),
            model=self.model,
            input=list_input,
            include=include,
            tools=converted_tools_payload,
            prompt=self._non_null_or_not_given(prompt),
            temperature=self._non_null_or_not_given(model_settings.temperature),
            top_p=self._non_null_or_not_given(model_settings.top_p),
            truncation=self._non_null_or_not_given(model_settings.truncation),
            max_output_tokens=self._non_null_or_not_given(model_settings.max_tokens),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            extra_headers=self._merge_headers(model_settings),
            extra_query=model_settings.extra_query,
            extra_body=model_settings.extra_body,
            text=response_format,
            store=self._non_null_or_not_given(model_settings.store),
            reasoning=self._non_null_or_not_given(model_settings.reasoning),
            metadata=self._non_null_or_not_given(model_settings.metadata),
            **extra_args,
        )

    def _merge_headers(self, model_settings: ModelSettings):
        return {
            **_HEADERS,
            **(model_settings.extra_headers or {}),
            **(_HEADERS_OVERRIDE.get() or {}),
        }


# ============================================================================
# OpenAI Chat Completions Model (for other providers and older OpenAI models)
# ============================================================================


class OpenAIChatCompletionsModel:
    """Implementation of Model that uses the OpenAI Chat Completions API.

    This is primarily for other providers (via compatibility) and older OpenAI models.
    GPT-5 uses the newer Responses API (OpenAIResponsesModel).
    """

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = openai_client

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> AsyncIterator[dict]:
        """Yields a partial message as it is generated, as well as the usage information."""
        messages = []

        if system_instructions:
            messages.append({"role": "system", "content": system_instructions})

        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        else:
            # Convert input list to messages
            for item in input:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    messages.append({"role": role, "content": content})

        # Convert tools to OpenAI format
        tools_payload = []
        for tool in tools:
            tools_payload.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if tools_payload:
            kwargs["tools"] = tools_payload

        if model_settings.temperature is not None:
            kwargs["temperature"] = model_settings.temperature
        if model_settings.top_p is not None:
            kwargs["top_p"] = model_settings.top_p
        if model_settings.max_tokens is not None:
            kwargs["max_tokens"] = model_settings.max_tokens

        stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            yield chunk.model_dump()
