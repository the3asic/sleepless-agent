#!/usr/bin/env python3
"""Validate Zhipu MCP configuration before running sleepless-agent.

Usage:
    python scripts/validate_mcp_config.py

This script checks:
1. Zhipu environment configuration
2. Node.js availability (for Vision MCP)
3. MCP endpoint connectivity (optional, requires network)
"""

import os
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_env_vars() -> dict:
    """Check Zhipu environment variables."""
    results = {
        "USE_ZHIPU": os.environ.get("USE_ZHIPU", ""),
        "ZHIPU_API_KEY": "***" if os.environ.get("ZHIPU_API_KEY") else "",
        "ZHIPU_BASE_URL": os.environ.get("ZHIPU_BASE_URL", ""),
    }
    return results


def check_nodejs() -> tuple[bool, str]:
    """Check if Node.js/npx is available."""
    try:
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            timeout=5,
            text=True,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, "npx returned non-zero"
    except FileNotFoundError:
        return False, "npx not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "npx timed out"
    except Exception as e:
        return False, str(e)


def check_mcp_config() -> dict:
    """Check MCP configuration from sleepless-agent."""
    try:
        from sleepless_agent.utils.zhipu_env import get_zhipu_config, is_zhipu_enabled
        from sleepless_agent.utils.config import get_config
        from sleepless_agent.utils.mcp_config import (
            get_mcp_servers_for_phase,
            _check_nodejs_available,
        )

        zhipu_enabled = is_zhipu_enabled()
        zhipu_config = get_zhipu_config()
        app_config = get_config()

        result = {
            "zhipu_enabled": zhipu_enabled,
            "region": zhipu_config.endpoints.name if zhipu_config.endpoints else "N/A",
            "nodejs_available": _check_nodejs_available(),
        }

        if zhipu_enabled and zhipu_config.api_key:
            # Check what MCP servers would be configured for each phase
            phases = ["planner", "worker", "evaluator", "chat"]
            for phase in phases:
                servers = get_mcp_servers_for_phase(app_config, phase)
                result[f"phase_{phase}"] = list(servers.keys()) if servers else []

        return result

    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("Sleepless-Agent MCP Configuration Validator")
    print("=" * 60)

    # 1. Environment variables
    print("\n[1/3] Environment Variables:")
    env_vars = check_env_vars()
    for key, value in env_vars.items():
        status = "OK" if value else "NOT SET"
        print(f"  {key}: {value or '(empty)'} [{status}]")

    zhipu_enabled = env_vars["USE_ZHIPU"].lower() in ("true", "1", "yes")

    # 2. Node.js check
    print("\n[2/3] Node.js Availability:")
    nodejs_ok, nodejs_info = check_nodejs()
    status = "OK" if nodejs_ok else "MISSING"
    print(f"  npx: {nodejs_info} [{status}]")
    if not nodejs_ok:
        print("  Warning: Vision MCP requires Node.js 18+")
        print("  Install from: https://nodejs.org/")

    # 3. MCP configuration
    print("\n[3/3] MCP Configuration:")
    if not zhipu_enabled:
        print("  Zhipu mode is DISABLED (USE_ZHIPU != true)")
        print("  MCP servers will not be injected")
    else:
        mcp_result = check_mcp_config()
        if "error" in mcp_result:
            print(f"  Error: {mcp_result['error']}")
        else:
            print(f"  Region: {mcp_result.get('region', 'N/A')}")
            print(f"  Node.js: {'Available' if mcp_result.get('nodejs_available') else 'Missing'}")
            print()
            print("  MCP Servers by Phase:")
            for phase in ["planner", "worker", "evaluator", "chat"]:
                servers = mcp_result.get(f"phase_{phase}", [])
                if servers:
                    print(f"    {phase}: {', '.join(servers)}")
                else:
                    print(f"    {phase}: (none)")

    # Summary
    print("\n" + "=" * 60)
    if zhipu_enabled and env_vars["ZHIPU_API_KEY"] and nodejs_ok:
        print("Status: READY for testing")
    elif not zhipu_enabled:
        print("Status: Using Claude Pro (Zhipu disabled)")
    else:
        issues = []
        if not env_vars["ZHIPU_API_KEY"]:
            issues.append("ZHIPU_API_KEY not set")
        if not nodejs_ok:
            issues.append("Node.js not available")
        print(f"Status: ISSUES - {', '.join(issues)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
