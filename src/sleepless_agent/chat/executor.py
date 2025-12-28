"""Chat-optimized Claude executor for interactive conversations."""

import asyncio
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from sleepless_agent.chat.session import ChatSession, ChatSessionStatus
from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


class ChatExecutor:
    """Execute chat prompts using Claude Code SDK in conversational mode.

    This executor is optimized for interactive conversations where:
    - User sends a message
    - Claude processes and responds
    - Conversation history is maintained
    - Full tool access for project work
    """

    def __init__(
        self,
        workspace_root: str = "./workspace",
        default_model: str = "claude-sonnet-4-5-20250929",
        max_turns: int = 15,
    ):
        """Initialize chat executor.

        Args:
            workspace_root: Root directory for workspaces
            default_model: Claude model to use
            max_turns: Maximum tool-use turns per message
        """
        self.workspace_root = Path(workspace_root)
        self.projects_dir = self.workspace_root / "projects"
        self.default_model = default_model
        self.max_turns = max_turns

        # Ensure directories exist
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _get_workspace_path(self, session: ChatSession) -> Path:
        """Get or create workspace for chat session.

        Uses the project workspace since project is required.
        """
        workspace = self.projects_dir / session.project_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _build_prompt(self, session: ChatSession, user_message: str) -> str:
        """Build the full prompt with conversation context.

        Includes:
        - System context about chat mode
        - Project information
        - Conversation history
        - Current user message
        """
        context = session.get_context_for_claude(max_messages=10)

        prompt_parts = [
            "You are Claude, an AI assistant engaged in an interactive chat session via Slack.",
            "",
            f"## Project: {session.project_name}",
            f"You are working in the project workspace. Project ID: {session.project_id}",
            "",
        ]

        if context:
            prompt_parts.append(context)
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "## Current User Message",
                user_message,
                "",
                "## Instructions",
                "- Respond naturally and conversationally",
                "- If you need clarification, ask the user",
                "- If you're performing file operations, explain what you're doing",
                "- Keep responses concise but informative",
                "- You have full access to read, write, edit files and run commands",
                "- When making changes, briefly summarize what you did",
            ]
        )

        return "\n".join(prompt_parts)

    async def execute_chat_turn(
        self,
        session: ChatSession,
        user_message: str,
    ) -> Tuple[str, Dict]:
        """Execute a single chat turn and return the complete response.

        Args:
            session: The active chat session
            user_message: User's message to process

        Returns:
            Tuple of (response_text, metrics_dict)
        """
        workspace = self._get_workspace_path(session)
        session.workspace_path = str(workspace)

        prompt = self._build_prompt(session, user_message)

        # Record user message in history
        session.add_message("user", user_message)

        # Get MCP servers for chat mode (all capabilities)
        from sleepless_agent.utils.config import get_config
        from sleepless_agent.utils.mcp_config import get_mcp_servers_for_phase
        config = get_config()
        mcp_servers = get_mcp_servers_for_phase(config, phase="chat")

        options = ClaudeAgentOptions(
            cwd=str(workspace),
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",
            max_turns=self.max_turns,
            model=self.default_model,
            mcp_servers=mcp_servers if mcp_servers else None,
        )

        response_parts = []
        tool_uses = []
        metrics = {
            "cost_usd": 0.0,
            "duration_ms": 0,
            "num_turns": 0,
            "is_error": False,
        }

        start_time = time.time()

        try:
            logger.info(
                "chat.executor.start",
                session_id=session.session_id,
                project_id=session.project_id,
                message_preview=user_message[:100],
            )

            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text.strip()
                            if text:
                                response_parts.append(text)
                        elif isinstance(block, ToolUseBlock):
                            # Track tool usage for logging
                            tool_uses.append(block.name)
                            logger.debug(
                                "chat.executor.tool_use",
                                tool=block.name,
                                session_id=session.session_id,
                            )

                elif isinstance(message, ResultMessage):
                    metrics["cost_usd"] = message.total_cost_usd
                    metrics["duration_ms"] = message.duration_ms
                    metrics["num_turns"] = message.num_turns
                    metrics["is_error"] = message.is_error

                    if message.is_error:
                        logger.warning(
                            "chat.executor.error_result",
                            session_id=session.session_id,
                            result=str(message.result)[:200],
                        )

            # Combine response
            full_response = "\n\n".join(response_parts) if response_parts else ""

            # Handle empty response
            if not full_response:
                full_response = "I processed your request but didn't generate a text response. Let me know if you need clarification."

            # Record assistant response in history
            session.add_message(
                "assistant",
                full_response,
                metadata={"tools_used": tool_uses, "cost_usd": metrics["cost_usd"]},
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "chat.executor.complete",
                session_id=session.session_id,
                response_length=len(full_response),
                tools_used=len(tool_uses),
                cost_usd=metrics["cost_usd"],
                duration_ms=elapsed_ms,
            )

            return full_response, metrics

        except Exception as e:
            logger.error(
                "chat.executor.failed",
                session_id=session.session_id,
                error=str(e),
            )
            error_response = f"I encountered an error while processing your request: {str(e)}"
            session.add_message("assistant", error_response)
            metrics["is_error"] = True
            return error_response, metrics

    async def execute_chat_turn_with_timeout(
        self,
        session: ChatSession,
        user_message: str,
        timeout_seconds: int = 300,
    ) -> Tuple[str, Dict]:
        """Execute chat turn with timeout protection.

        Args:
            session: The active chat session
            user_message: User's message to process
            timeout_seconds: Maximum execution time (default 5 minutes)

        Returns:
            Tuple of (response_text, metrics_dict)
        """
        try:
            return await asyncio.wait_for(
                self.execute_chat_turn(session, user_message),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "chat.executor.timeout",
                session_id=session.session_id,
                timeout_seconds=timeout_seconds,
            )
            error_response = f"The request timed out after {timeout_seconds} seconds. Please try a simpler request or break it into smaller steps."
            session.add_message("assistant", error_response)
            return error_response, {"is_error": True, "timeout": True}
