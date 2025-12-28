"""Unified CLI entry point for sleepless agent (daemon or CLI commands)."""

from __future__ import annotations

import sys
from typing import Optional

from dotenv import load_dotenv

# Load .env early
load_dotenv()

# Setup Zhipu environment if enabled (must be before any Claude CLI calls)
from sleepless_agent.utils.zhipu_env import setup_zhipu_environment
setup_zhipu_environment()

from sleepless_agent.interfaces.cli import main as cli_main


def main(argv: Optional[list[str]] = None) -> int:
    """Route to daemon or CLI based on the command."""

    args = argv if argv is not None else sys.argv[1:]

    # If no args or first arg is not "daemon", treat as CLI
    if not args or args[0] != "daemon":
        return cli_main(args)

    # If first arg is "daemon", import and run the daemon (lazy import)
    from sleepless_agent.core.daemon import main as daemon_main
    return daemon_main()


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
