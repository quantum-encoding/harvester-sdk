"""
Model Context Protocol (MCP) Integration

Provides standardized access to MCP servers for tools and context.
Supports multiple transports: Hosted, Streamable HTTP, SSE, and stdio.
"""

from typing import Optional, Callable, List, Dict, Any, Union
from pathlib import Path
import os


class MCPServer:
    """
    Base wrapper for MCP servers.

    MCP standardizes how applications expose tools and context to LLMs.
    Think of it as USB-C for AI - a universal connector for data sources.
    """

    @staticmethod
    def hosted(
        server_label: str,
        server_url: Optional[str] = None,
        connector_id: Optional[str] = None,
        authorization: Optional[str] = None,
        require_approval: Union[str, Dict[str, str]] = "never",
        on_approval_request: Optional[Callable] = None,
    ):
        """
        Create a hosted MCP tool (runs on OpenAI's infrastructure).

        Best for publicly reachable MCP servers. The Responses API
        lists and calls tools directly without callbacks to your code.

        Args:
            server_label: Label for the MCP server
            server_url: URL for HTTP-based servers
            connector_id: Connector ID for OpenAI connectors
            authorization: Auth token for connectors
            require_approval: Approval policy - "always", "never", or dict
            on_approval_request: Callback for programmatic approval

        Returns:
            HostedMCPTool instance

        Example:
            # Public MCP server
            tool = MCPServer.hosted(
                server_label="gitmcp",
                server_url="https://gitmcp.io/openai/codex",
                require_approval="never"
            )

            # OpenAI connector
            tool = MCPServer.hosted(
                server_label="google_calendar",
                connector_id="connector_googlecalendar",
                authorization=os.environ["GOOGLE_CALENDAR_TOKEN"]
            )
        """
        try:
            from agents import HostedMCPTool
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        tool_config = {
            "type": "mcp",
            "server_label": server_label,
            "require_approval": require_approval,
        }

        if server_url:
            tool_config["server_url"] = server_url
        if connector_id:
            tool_config["connector_id"] = connector_id
        if authorization:
            tool_config["authorization"] = authorization

        kwargs = {"tool_config": tool_config}
        if on_approval_request:
            kwargs["on_approval_request"] = on_approval_request

        return HostedMCPTool(**kwargs)

    @staticmethod
    async def streamable_http(
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        cache_tools: bool = True,
        max_retries: int = 3,
        tool_filter: Optional[Callable] = None,
    ):
        """
        Create a Streamable HTTP MCP server.

        Best for servers you control or run in your infrastructure.
        Manages the network connection yourself for low latency.

        Args:
            name: Server name
            url: Server URL
            headers: HTTP headers (e.g., auth tokens)
            timeout: Request timeout in seconds
            cache_tools: Cache tool list for performance
            max_retries: Retry attempts for failed requests
            tool_filter: Optional filter for exposed tools

        Returns:
            MCPServerStreamableHttp instance (context manager)

        Example:
            async with MCPServer.streamable_http(
                name="My Server",
                url="http://localhost:8000/mcp",
                headers={"Authorization": f"Bearer {token}"}
            ) as server:
                agent = OpenAIAgent(
                    name="Assistant",
                    mcp_servers=[server]
                )
        """
        try:
            from agents.mcp import MCPServerStreamableHttp
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        params = {"url": url, "timeout": timeout}
        if headers:
            params["headers"] = headers

        kwargs = {
            "name": name,
            "params": params,
            "cache_tools_list": cache_tools,
            "max_retry_attempts": max_retries,
        }

        if tool_filter:
            kwargs["tool_filter"] = tool_filter

        return MCPServerStreamableHttp(**kwargs)

    @staticmethod
    async def sse(
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        cache_tools: bool = True,
        tool_filter: Optional[Callable] = None,
    ):
        """
        Create an HTTP with SSE MCP server.

        For servers implementing Server-Sent Events transport.

        Args:
            name: Server name
            url: Server URL
            headers: HTTP headers
            cache_tools: Cache tool list for performance
            tool_filter: Optional filter for exposed tools

        Returns:
            MCPServerSse instance (context manager)

        Example:
            async with MCPServer.sse(
                name="Weather Server",
                url="http://localhost:8000/sse",
                headers={"X-Workspace": workspace_id}
            ) as server:
                agent = OpenAIAgent(mcp_servers=[server])
        """
        try:
            from agents.mcp import MCPServerSse
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        params = {"url": url}
        if headers:
            params["headers"] = headers

        kwargs = {
            "name": name,
            "params": params,
            "cache_tools_list": cache_tools,
        }

        if tool_filter:
            kwargs["tool_filter"] = tool_filter

        return MCPServerSse(**kwargs)

    @staticmethod
    async def stdio(
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        cache_tools: bool = True,
        tool_filter: Optional[Callable] = None,
    ):
        """
        Create a stdio MCP server (local subprocess).

        Spawns a process and communicates over stdin/stdout.
        Good for local tools and quick prototypes.

        Args:
            name: Server name
            command: Command to execute
            args: Command arguments
            cwd: Working directory
            env: Environment variables
            cache_tools: Cache tool list for performance
            tool_filter: Optional filter for exposed tools

        Returns:
            MCPServerStdio instance (context manager)

        Example:
            # NPX-based filesystem server
            async with MCPServer.stdio(
                name="Filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "./files"]
            ) as server:
                agent = OpenAIAgent(mcp_servers=[server])
        """
        try:
            from agents.mcp import MCPServerStdio
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: "
                "pip install harvester-sdk[computer]"
            )

        params = {"command": command}
        if args:
            params["args"] = args
        if cwd:
            params["cwd"] = str(cwd)
        if env:
            params["env"] = env

        kwargs = {
            "name": name,
            "params": params,
            "cache_tools_list": cache_tools,
        }

        if tool_filter:
            kwargs["tool_filter"] = tool_filter

        return MCPServerStdio(**kwargs)


def create_tool_filter(
    allowed_tools: Optional[List[str]] = None,
    blocked_tools: Optional[List[str]] = None,
):
    """
    Create a static tool filter for MCP servers.

    Filters which tools are exposed to the agent.
    Allow-list is applied first, then block-list.

    Args:
        allowed_tools: List of allowed tool names (None = allow all)
        blocked_tools: List of blocked tool names

    Returns:
        Static tool filter function

    Example:
        filter_fn = create_tool_filter(
            allowed_tools=["read_file", "write_file"],
            blocked_tools=["delete_file"]
        )

        server = await MCPServer.stdio(
            name="Filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "./files"],
            tool_filter=filter_fn
        )
    """
    try:
        from agents.mcp import create_static_tool_filter
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: "
            "pip install harvester-sdk[computer]"
        )

    kwargs = {}
    if allowed_tools:
        kwargs["allowed_tool_names"] = allowed_tools
    if blocked_tools:
        kwargs["blocked_tool_names"] = blocked_tools

    return create_static_tool_filter(**kwargs)


class MCPApprovalHandler:
    """
    Helper for handling MCP tool approval requests.

    Allows you to programmatically approve or deny tool executions
    from hosted MCP servers.
    """

    @staticmethod
    def create_handler(
        safe_tools: Optional[List[str]] = None,
        dangerous_tools: Optional[List[str]] = None,
        default_approve: bool = False,
    ) -> Callable:
        """
        Create an approval handler function.

        Args:
            safe_tools: List of always-approved tool names
            dangerous_tools: List of always-denied tool names
            default_approve: Default decision for unlisted tools

        Returns:
            Approval handler function

        Example:
            handler = MCPApprovalHandler.create_handler(
                safe_tools=["read_file", "list_files"],
                dangerous_tools=["delete_file", "format_disk"],
                default_approve=False
            )

            tool = MCPServer.hosted(
                server_label="filesystem",
                server_url="https://fs.example.com",
                require_approval="always",
                on_approval_request=handler
            )
        """
        safe_set = set(safe_tools or [])
        dangerous_set = set(dangerous_tools or [])

        def handler(request):
            tool_name = request.data.name

            # Always approve safe tools
            if tool_name in safe_set:
                return {"approve": True}

            # Always deny dangerous tools
            if tool_name in dangerous_set:
                return {
                    "approve": False,
                    "reason": f"Tool '{tool_name}' is not allowed for security reasons"
                }

            # Default decision for unlisted tools
            if default_approve:
                return {"approve": True}
            else:
                return {
                    "approve": False,
                    "reason": f"Tool '{tool_name}' requires explicit approval"
                }

        return handler


# Example usage
if __name__ == "__main__":
    import asyncio
    from ..openai_agent import OpenAIAgent

    async def main():
        # Example 1: Hosted MCP server
        print("Example 1: Hosted MCP server")
        hosted_tool = MCPServer.hosted(
            server_label="gitmcp",
            server_url="https://gitmcp.io/openai/codex",
            require_approval="never"
        )

        agent1 = OpenAIAgent(
            name="Git Assistant",
            tools=[hosted_tool]
        )

        # Example 2: Local stdio server
        print("\nExample 2: Local stdio MCP server")
        async with MCPServer.stdio(
            name="Filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "./"],
            tool_filter=create_tool_filter(
                allowed_tools=["read_file", "list_directory"]
            )
        ) as server:
            agent2 = OpenAIAgent(
                name="File Assistant",
                mcp_servers=[server]
            )
            print(f"Agent created with MCP server: {server.name}")

        # Example 3: Approval handler
        print("\nExample 3: Approval handler")
        handler = MCPApprovalHandler.create_handler(
            safe_tools=["read_file"],
            dangerous_tools=["delete_file"],
            default_approve=False
        )

        secure_tool = MCPServer.hosted(
            server_label="secure_fs",
            server_url="https://fs.example.com",
            require_approval="always",
            on_approval_request=handler
        )

        print("MCP examples created successfully!")

    asyncio.run(main())
