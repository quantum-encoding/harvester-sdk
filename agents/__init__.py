"""
Harvester SDK Agents

Specialized agentic workflows for different AI models and tasks.

Available Agents:
- GrokCodeAgent: Agentic coding assistant powered by grok-code-fast-1
- ClaudeCodeAgent: Professional agentic assistant powered by Claude Agent SDK
- OpenAIAgent: General-purpose agent powered by OpenAI Agents SDK (GPT-5)
"""

from .grok_code_agent import GrokCodeAgent, AgentTask as GrokAgentTask, ReasoningTrace, ToolCall
from .claude_code_agent import ClaudeCodeAgent, AgentTask as ClaudeAgentTask
from .openai_agent import OpenAIAgent, AgentSession, function_tool
from .streaming import StreamHandler, stream_to_console, collect_stream_text
from .repl import run_demo_loop, run_interactive_chat
from .tools import HostedTools, agent_as_tool, create_function_tool, ToolConfig
from .mcp import MCPServer, create_tool_filter, MCPApprovalHandler
from .handoffs import (
    create_handoff,
    create_handoff_with_data,
    HandoffFilters,
    HandoffPrompts,
    HandoffConfig
)
from .guardrails import (
    create_input_guardrail,
    create_output_guardrail,
    CommonGuardrails,
    GuardrailBuilder,
    GuardrailConfig
)
from .orchestration import (
    AgentChain,
    ParallelAgents,
    EvaluationLoop,
    ClassificationRouter,
    AgentChainResult
)
from .usage import (
    UsageStats,
    UsageTracker,
    UsageLimiter,
    get_usage_from_result,
    get_model_pricing,
    MODEL_PRICING
)
from .litellm_support import (
    create_litellm_agent,
    LiteLLMConfig,
    get_litellm_model_id,
    disable_tracing_for_non_openai,
    LITELLM_MODELS
)
from .sqlalchemy_session import SQLAlchemySession, SessionABC
from .sqlite_session import SQLiteSession
from .runner import (
    Runner,
    RunConfig,
    RunResult,
    RunResultStreaming,
    RunHooks,
    MaxTurnsExceeded,
    GuardrailTripwireTriggered,
)
from .schemas import (
    AgentOutputSchemaBase,
    AgentOutputSchema,
    FuncSchema,
    FuncDocumentation,
    DocstringStyle,
    ModelBehaviorError,
    function_schema,
    generate_func_documentation,
)
from .prompt_sanitizer import PromptSanitizer, sanitize_prompt
from .openai_models import (
    OpenAIResponsesModel,
    OpenAIChatCompletionsModel,
    ModelSettings,
    ModelResponse,
    ModelTracing,
    Tool as ModelTool,
    Handoff as ModelHandoff,
)

__all__ = [
    'GrokCodeAgent', 'GrokAgentTask', 'ReasoningTrace', 'ToolCall',
    'ClaudeCodeAgent', 'ClaudeAgentTask',
    'OpenAIAgent', 'AgentSession', 'function_tool',
    'StreamHandler', 'stream_to_console', 'collect_stream_text',
    'run_demo_loop', 'run_interactive_chat',
    'HostedTools', 'agent_as_tool', 'create_function_tool', 'ToolConfig',
    'MCPServer', 'create_tool_filter', 'MCPApprovalHandler',
    'create_handoff', 'create_handoff_with_data', 'HandoffFilters', 'HandoffPrompts', 'HandoffConfig',
    'create_input_guardrail', 'create_output_guardrail', 'CommonGuardrails', 'GuardrailBuilder', 'GuardrailConfig',
    'AgentChain', 'ParallelAgents', 'EvaluationLoop', 'ClassificationRouter', 'AgentChainResult',
    'UsageStats', 'UsageTracker', 'UsageLimiter', 'get_usage_from_result', 'get_model_pricing', 'MODEL_PRICING',
    'create_litellm_agent', 'LiteLLMConfig', 'get_litellm_model_id', 'disable_tracing_for_non_openai', 'LITELLM_MODELS',
    'SQLAlchemySession', 'SQLiteSession', 'SessionABC',
    'Runner', 'RunConfig', 'RunResult', 'RunResultStreaming', 'RunHooks',
    'MaxTurnsExceeded', 'GuardrailTripwireTriggered',
    'AgentOutputSchemaBase', 'AgentOutputSchema', 'FuncSchema', 'FuncDocumentation',
    'DocstringStyle', 'ModelBehaviorError', 'function_schema', 'generate_func_documentation',
    'PromptSanitizer', 'sanitize_prompt',
    'OpenAIResponsesModel', 'OpenAIChatCompletionsModel', 'ModelSettings', 'ModelResponse', 'ModelTracing',
    'ModelTool', 'ModelHandoff'
]
