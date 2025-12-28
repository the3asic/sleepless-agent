"""Zhipu environment setup and detection utilities."""

import os
from typing import NamedTuple, Optional
from urllib.parse import urlparse

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


# Endpoint mapping: CN vs Global
ZHIPU_ENDPOINTS = {
    "bigmodel.cn": {
        "name": "zhipu_cn",
        "api_base": "https://open.bigmodel.cn/api/anthropic",
        "usage_api": "https://open.bigmodel.cn/api/monitor/usage/quota/limit",
        "mcp_base": "https://open.bigmodel.cn/api/mcp",
        "mcp_mode": "ZHIPU",
    },
    "z.ai": {
        "name": "zai_global",
        "api_base": "https://api.z.ai/api/anthropic",
        "usage_api": "https://api.z.ai/api/monitor/usage/quota/limit",
        "mcp_base": "https://api.z.ai/api/mcp",
        "mcp_mode": "ZAI",
    }
}


class ZhipuEndpoints(NamedTuple):
    """Zhipu service endpoints."""
    name: str           # "zhipu_cn" or "zai_global"
    api_base: str       # Claude API compatible endpoint
    usage_api: str      # Usage query API
    mcp_base: str       # MCP server base URL
    mcp_mode: str       # MCP mode identifier


class ZhipuConfig(NamedTuple):
    """Zhipu configuration container."""
    enabled: bool
    api_key: str
    base_url: str
    timeout_ms: int
    endpoints: Optional[ZhipuEndpoints]


def detect_zhipu_region(base_url: str) -> ZhipuEndpoints:
    """Detect Zhipu region from base URL and return all endpoints.

    Args:
        base_url: ZHIPU_BASE_URL value

    Returns:
        ZhipuEndpoints with all derived URLs
    """
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()

    # Detect domain
    for domain, endpoints in ZHIPU_ENDPOINTS.items():
        if domain in host:
            return ZhipuEndpoints(**endpoints)

    # Default to CN version
    logger.warning("zhipu_env.unknown_domain", host=host, fallback="zhipu_cn")
    return ZhipuEndpoints(**ZHIPU_ENDPOINTS["bigmodel.cn"])


def is_zhipu_enabled() -> bool:
    """Check if Zhipu mode is enabled via USE_ZHIPU env var."""
    return os.environ.get("USE_ZHIPU", "").lower() in ("true", "1", "yes")


def get_zhipu_config() -> ZhipuConfig:
    """Get Zhipu configuration from environment with auto-detected endpoints."""
    enabled = is_zhipu_enabled()

    if not enabled:
        return ZhipuConfig(
            enabled=False,
            api_key="",
            base_url="",
            timeout_ms=300000,
            endpoints=None,
        )

    base_url = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    endpoints = detect_zhipu_region(base_url)

    return ZhipuConfig(
        enabled=True,
        api_key=os.environ.get("ZHIPU_API_KEY", ""),
        base_url=base_url,
        timeout_ms=int(os.environ.get("ZHIPU_TIMEOUT_MS", "300000")),
        endpoints=endpoints,
    )


def setup_zhipu_environment() -> bool:
    """Setup environment variables for Zhipu if enabled.

    This function should be called early in the application startup.
    It reads ZHIPU_* variables and sets ANTHROPIC_* variables that
    Claude Code CLI expects.

    Returns:
        True if Zhipu mode was enabled and configured, False otherwise.
    """
    config = get_zhipu_config()

    if not config.enabled:
        logger.debug("zhipu_env.disabled")
        return False

    if not config.api_key:
        logger.error("zhipu_env.missing_api_key",
                     hint="Set ZHIPU_API_KEY in .env when USE_ZHIPU=true")
        return False

    # Set environment variables for Claude Code CLI
    os.environ["ANTHROPIC_AUTH_TOKEN"] = config.api_key
    os.environ["ANTHROPIC_BASE_URL"] = config.base_url
    os.environ["API_TIMEOUT_MS"] = str(config.timeout_ms)

    # Optional: disable non-essential traffic for better performance
    os.environ.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")

    logger.info(
        "zhipu_env.configured",
        region=config.endpoints.name if config.endpoints else "unknown",
        base_url=config.base_url,
        timeout_ms=config.timeout_ms,
    )
    return True


def get_usage_checker():
    """Factory function to get the appropriate usage checker.

    Returns ZhipuUsageChecker if Zhipu is enabled, otherwise ProPlanUsageChecker.
    """
    if is_zhipu_enabled():
        from sleepless_agent.monitoring.zhipu_usage import ZhipuUsageChecker
        return ZhipuUsageChecker()
    else:
        from sleepless_agent.monitoring.pro_plan_usage import ProPlanUsageChecker
        from sleepless_agent.utils.config import get_config
        config = get_config()
        return ProPlanUsageChecker(command=config.claude_code.usage_command)
