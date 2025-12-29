"""MCP server configuration builder for Zhipu with auto-detected endpoints."""

import subprocess
from typing import Any, Dict, Optional

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


def _check_nodejs_available() -> bool:
    """Check if npx is available for Vision MCP (requires Node.js 18+).

    Returns:
        True if npx is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            timeout=5,
            text=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def build_zhipu_mcp_servers(
    config: Any,
    include_vision: bool = True,
    include_search: bool = True,
    include_reader: bool = True,
) -> Dict[str, Any]:
    """Build MCP server configurations with auto-detected regional endpoints.

    Args:
        config: Application configuration (with mcp.zhipu settings)
        include_vision: Whether to include Vision MCP server
        include_search: Whether to include Search MCP server
        include_reader: Whether to include Reader MCP server

    Returns:
        Dictionary of MCP server configurations for ClaudeAgentOptions.mcp_servers
    """
    from sleepless_agent.utils.zhipu_env import get_zhipu_config, is_zhipu_enabled

    if not is_zhipu_enabled():
        return {}

    zhipu = get_zhipu_config()

    if not zhipu.api_key:
        logger.warning("mcp_config.no_api_key")
        return {}

    # Check if MCP is enabled in config
    if not hasattr(config, 'mcp') or not getattr(config.mcp, 'enabled', False):
        return {}

    api_key = zhipu.api_key
    endpoints = zhipu.endpoints

    # Get MCP base URL and mode from auto-detected endpoints
    mcp_base = endpoints.mcp_base if endpoints else "https://open.bigmodel.cn/api/mcp"
    mcp_mode = endpoints.mcp_mode if endpoints else "ZHIPU"

    servers: Dict[str, Any] = {}

    # Get Zhipu MCP config from application config
    zhipu_mcp = getattr(getattr(config, 'mcp', None), 'zhipu', None)
    if zhipu_mcp is None:
        # Use defaults if no config (correct npm package: @z_ai/mcp-server)
        zhipu_mcp = type('obj', (object,), {
            'servers': type('obj', (object,), {
                'vision': type('obj', (object,), {'enabled': True, 'command': 'npx', 'args': ['-y', '@z_ai/mcp-server']})(),
                'search': type('obj', (object,), {'enabled': True})(),
                'reader': type('obj', (object,), {'enabled': True})(),
            })()
        })()

    servers_config = zhipu_mcp.servers

    # Vision MCP - Image understanding (stdio mode, requires Node.js)
    if include_vision and getattr(getattr(servers_config, 'vision', None), 'enabled', True):
        # Check Node.js availability before adding Vision MCP
        if not _check_nodejs_available():
            logger.warning("mcp_config.vision.nodejs_unavailable",
                          hint="Vision MCP requires Node.js 18+. Install from https://nodejs.org/")
        else:
            vision_config = servers_config.vision
            servers["zai-mcp-server"] = {
                "type": "stdio",
                "command": getattr(vision_config, 'command', 'npx'),
                "args": getattr(vision_config, 'args', ['-y', '@z_ai/mcp-server']),
                "env": {
                    "Z_AI_API_KEY": api_key,
                    "Z_AI_MODE": mcp_mode  # Auto-detected: ZHIPU or ZAI
                }
            }

    # Search MCP - Web search (HTTP mode for Claude Code)
    if include_search and getattr(getattr(servers_config, 'search', None), 'enabled', True):
        servers["web-search-prime"] = {
            "type": "http",
            "url": f"{mcp_base}/web_search_prime/mcp",
            "headers": {
                "Authorization": f"Bearer {api_key}"
            }
        }

    # Reader MCP - Web page reading (HTTP mode for Claude Code)
    if include_reader and getattr(getattr(servers_config, 'reader', None), 'enabled', True):
        servers["web-reader"] = {
            "type": "http",
            "url": f"{mcp_base}/web_reader/mcp",
            "headers": {
                "Authorization": f"Bearer {api_key}"
            }
        }

    if servers:
        logger.debug(
            "mcp_config.built",
            servers=list(servers.keys()),
            region=endpoints.name if endpoints else "unknown",
        )

    return servers


def get_mcp_servers_for_phase(
    config: Any,
    phase: str = "worker",
) -> Dict[str, Any]:
    """Get MCP servers appropriate for a specific execution phase.

    Different phases may need different MCP capabilities:
    - planner: Vision + Search + Reader (needs Vision for design mockups/screenshots)
    - worker: Vision + Search + Reader (full capabilities for implementation)
    - evaluator: Reader only (for document verification, no search to save quota)
    - chat: Vision + Search + Reader (full capabilities for interactive use)

    Args:
        config: Application configuration
        phase: Execution phase ("planner", "worker", "evaluator", "chat")

    Returns:
        Dictionary of MCP server configurations
    """
    if phase == "evaluator":
        # Evaluator needs Reader for document verification (e.g., compare with API docs)
        # No Vision/Search to save quota - evaluator works with already-fetched context
        return build_zhipu_mcp_servers(
            config,
            include_vision=False,
            include_search=False,
            include_reader=True,
        )

    if phase == "planner":
        # Planner needs all capabilities including Vision for:
        # - Understanding design mockups/screenshots
        # - Researching implementation approaches
        # - Reading reference documentation
        return build_zhipu_mcp_servers(
            config,
            include_vision=True,
            include_search=True,
            include_reader=True,
        )

    # Worker and chat get all MCP capabilities
    return build_zhipu_mcp_servers(config)
