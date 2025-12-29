"""Chat mode command handler for Slack integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional

from sleepless_agent.chat.session import (
    ChatSession,
    ChatSessionManager,
    ChatSessionStatus,
)
from sleepless_agent.chat.executor import ChatExecutor
from sleepless_agent.tasks.utils import slugify_project
from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.interfaces.message_client import MessageClient

logger = get_logger(__name__)


class ChatHandler:
    """Handles chat mode commands and message routing.

    Responsibilities:
    - Parse /chat command and subcommands
    - Validate project requirements
    - Route thread messages to chat executor
    - Format responses for Slack
    """

    def __init__(
        self,
        session_manager: ChatSessionManager,
        chat_executor: ChatExecutor,
        task_queue,  # TaskQueue for project lookup
        message_client: MessageClient,  # Platform-agnostic message client
    ):
        """Initialize chat handler.

        Args:
            session_manager: Manages active chat sessions
            chat_executor: Executes Claude queries
            task_queue: Task queue for project validation
            message_client: Platform-agnostic message client (Slack, Telegram, etc.)
        """
        self.session_manager = session_manager
        self.chat_executor = chat_executor
        self.task_queue = task_queue
        self.message_client = message_client
        self._processing_users: set = set()  # Track users currently being processed

    def handle_chat_command(
        self,
        args: str,
        user_id: str,
        channel_id: str,
        response_url: str,
    ) -> Dict[str, Any]:
        """Handle /chat command.

        Usage:
            /chat <project_name> - Start chat mode for a project
            /chat end - End current chat session
            /chat status - Check current chat session status
            /chat help - Show help

        Returns:
            Dict with response info or action to take
        """
        args = args.strip()
        args_lower = args.lower()

        # Handle subcommands
        if args_lower in ("end", "exit", "quit"):
            return self._end_chat_session(user_id)

        if args_lower == "status":
            return self._get_session_status(user_id)

        if args_lower == "help":
            return self._get_help()

        # No args = error (project required)
        if not args:
            return self._get_project_required_error()

        # Check if user already has active session
        existing_session = self.session_manager.get_session(user_id)
        if existing_session and existing_session.status != ChatSessionStatus.ENDED:
            return {
                "text": "You already have an active chat session.",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"You already have an active chat session for project *{existing_session.project_name}*.\n\n"
                                f"Use `/chat end` to end it first, or continue in the existing thread."
                            ),
                        },
                    },
                ],
            }

        # Parse project name - use the raw args as project name
        project_name = args
        project_id = slugify_project(project_name)

        return self._start_chat_session(
            user_id=user_id,
            channel_id=channel_id,
            project_id=project_id,
            project_name=project_name,
        )

    def _get_project_required_error(self) -> Dict[str, Any]:
        """Return error for missing project name."""
        return {
            "text": "Project name required",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            "*Project name is required.*\n\n"
                            "Usage: `/chat <project_name>`\n\n"
                            "Example: `/chat my-backend-api`"
                        ),
                    },
                },
            ],
        }

    def _start_chat_session(
        self,
        user_id: str,
        channel_id: str,
        project_id: str,
        project_name: str,
    ) -> Dict[str, Any]:
        """Prepare to start a new chat session.

        Returns action dict for bot to create the thread and session.
        """
        return {
            "action": "start_session",
            "user_id": user_id,
            "channel_id": channel_id,
            "project_id": project_id,
            "project_name": project_name,
            "text": f"Starting chat session for project: {project_name}",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Starting Chat Mode"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Initializing chat session for project *{project_name}*...",
                    },
                },
            ],
        }

    def _end_chat_session(self, user_id: str) -> Dict[str, Any]:
        """End user's chat session."""
        session = self.session_manager.end_session(user_id)
        if session:
            duration = self._format_session_duration(session)
            msg_count = len(session.conversation_history)
            return {
                "text": "Chat session ended.",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Chat Session Ended"},
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"*Project:* {session.project_name}\n"
                                f"*Duration:* {duration}\n"
                                f"*Messages:* {msg_count}"
                            ),
                        },
                    },
                ],
            }
        return {
            "text": "No active chat session to end.",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "You don't have an active chat session.\n\nUse `/chat <project_name>` to start one.",
                    },
                },
            ],
        }

    def _get_session_status(self, user_id: str) -> Dict[str, Any]:
        """Get status of user's chat session."""
        session = self.session_manager.get_session(user_id)
        if not session:
            return {
                "text": "No active chat session.",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "You don't have an active chat session.\n\nUse `/chat <project_name>` to start one.",
                        },
                    },
                ],
            }

        status_emoji = {
            ChatSessionStatus.ACTIVE: "ready",
            ChatSessionStatus.WAITING_FOR_INPUT: "waiting for your input",
            ChatSessionStatus.PROCESSING: "processing your message",
            ChatSessionStatus.ERROR: "error state",
        }.get(session.status, "unknown")

        duration = self._format_session_duration(session)
        msg_count = len(session.conversation_history)

        return {
            "text": f"Chat session status: {status_emoji}",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Chat Session Status"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Status:* {status_emoji}\n"
                            f"*Project:* {session.project_name}\n"
                            f"*Messages:* {msg_count}\n"
                            f"*Duration:* {duration}"
                        ),
                    },
                },
            ],
        }

    def _get_help(self) -> Dict[str, Any]:
        """Return help information."""
        return {
            "text": "Chat Mode Help",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Chat Mode Commands"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            "*Commands:*\n"
                            "`/chat <project_name>` - Start chat mode for a project\n"
                            "`/chat end` - End current chat session\n"
                            "`/chat status` - Check current session status\n"
                            "`/chat help` - Show this help\n\n"
                            "*In Chat Mode:*\n"
                            "- Send messages in the chat thread to interact with Claude\n"
                            "- Claude has full access to read, write, and edit files\n"
                            "- Conversation history is maintained across messages\n"
                            "- Type `exit` in the thread to end the session\n"
                            "- Sessions auto-end after 30 minutes of inactivity"
                        ),
                    },
                },
            ],
        }

    async def handle_chat_message(
        self,
        session: ChatSession,
        message_text: str,
        channel_id: str,
        thread_ts: str,
    ) -> None:
        """Handle a message from a user in chat mode.

        Args:
            session: The active chat session
            message_text: The user's message
            channel_id: Slack channel ID
            thread_ts: Thread timestamp
        """
        user_id = session.user_id

        # Check for exit commands in thread
        if message_text.lower().strip() in ("exit", "end", "quit", "/chat end"):
            self.session_manager.end_session(user_id)
            self._send_thread_message(
                channel_id,
                thread_ts,
                self._format_goodbye_message(session),
            )
            # Update welcome message and remove active indicator
            self._update_session_ended(channel_id, thread_ts, session)
            return

        # Prevent concurrent processing for same user
        if user_id in self._processing_users:
            self._send_thread_message(
                channel_id,
                thread_ts,
                "⏳ Please wait, I'm still processing your previous message...",
            )
            return

        self._processing_users.add(user_id)
        self.session_manager.update_session_status(user_id, ChatSessionStatus.PROCESSING)

        # Send processing indicator
        processing_msg_ts = self._send_processing_indicator(channel_id, thread_ts)

        try:
            # Execute chat turn
            response, metrics = await self.chat_executor.execute_chat_turn_with_timeout(
                session, message_text, timeout_seconds=300
            )

            # Delete processing indicator
            self._delete_message(channel_id, processing_msg_ts)

            # Send response (chunked if needed)
            self._send_response_chunked(channel_id, thread_ts, response)

            # Update session status
            self.session_manager.update_session_status(
                user_id, ChatSessionStatus.WAITING_FOR_INPUT
            )

        except Exception as e:
            logger.error(f"Error handling chat message: {e}", exc_info=True)
            # Delete processing indicator
            self._delete_message(channel_id, processing_msg_ts)

            self.session_manager.update_session_status(
                user_id, ChatSessionStatus.ERROR, error_message=str(e)
            )
            self._send_thread_message(
                channel_id,
                thread_ts,
                f"❌ Sorry, I encountered an error: {str(e)}",
            )

        finally:
            self._processing_users.discard(user_id)

    def _send_processing_indicator(self, channel: str, thread_ts: str) -> Optional[str]:
        """Send a processing indicator message and return its ID."""
        result = self.message_client.send_message(
            channel=channel,
            message="🔄 Processing your message...",
            thread_id=thread_ts,
        )
        if result.success:
            return result.message_id
        logger.debug("chat_handler.processing_indicator.failed", error=result.error)
        return None

    def _delete_message(self, channel: str, message_id: Optional[str]) -> None:
        """Delete a message by ID."""
        if not message_id:
            return
        self.message_client.delete_message(channel=channel, message_id=message_id)

    def _update_session_ended(
        self, channel: str, thread_ts: str, session: ChatSession
    ) -> None:
        """Update the welcome message and remove active indicator when session ends."""
        # Remove the active reaction and add ended reaction
        # Note: Not all platforms support reactions (e.g., Telegram doesn't)
        self.message_client.remove_reaction(
            channel=channel,
            message_id=thread_ts,
            reaction="speech_balloon",
        )
        self.message_client.add_reaction(
            channel=channel,
            message_id=thread_ts,
            reaction="white_check_mark",  # ✅ emoji
        )

    def _send_thread_message(
        self, channel: str, thread_ts: str, text: str, blocks: Optional[List] = None
    ) -> None:
        """Send a message to a thread.

        Note: blocks parameter is kept for backward compatibility but FormattedMessage
        is preferred for platform-agnostic formatting.
        """
        result = self.message_client.send_message(
            channel=channel,
            message=text,
            thread_id=thread_ts,
        )
        if not result.success:
            logger.error("chat_handler.send_thread_message.failed", error=result.error)

    def _format_response_blocks(self, text: str) -> List[Dict]:
        """Format Claude's response into Slack Block Kit blocks for better visuals."""
        blocks = []

        # Split by double newlines to get paragraphs/sections
        sections = text.split("\n\n")

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Check if it's a numbered list (starts with "1." or similar)
            if self._is_numbered_list(section):
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": section}
                })
            # Check if it's a header-like line (short, possibly with ** or ##)
            elif self._is_header(section):
                # Convert markdown headers to bold
                header_text = section.replace("##", "").replace("**", "").strip()
                blocks.append({
                    "type": "header",
                    "text": {"type": "plain_text", "text": header_text[:150]}  # Slack limit
                })
            # Check if it's a code block
            elif section.startswith("```"):
                code_content = section.strip("`").strip()
                # Remove language identifier if present
                if "\n" in code_content:
                    first_line, rest = code_content.split("\n", 1)
                    if not " " in first_line and len(first_line) < 20:
                        code_content = rest
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{code_content}```"}
                })
            else:
                # Regular text section
                # Convert markdown bold **text** to Slack bold *text*
                formatted = self._convert_markdown_to_slack(section)
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": formatted}
                })

        # Add a divider at the end for visual separation
        if blocks:
            blocks.append({"type": "divider"})

        return blocks

    def _is_numbered_list(self, text: str) -> bool:
        """Check if text is a numbered list."""
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return False
        # Check if most lines start with number + dot
        numbered = sum(1 for line in lines if line.strip() and line.strip()[0].isdigit())
        return numbered >= len(lines) * 0.5

    def _is_header(self, text: str) -> bool:
        """Check if text looks like a header."""
        text = text.strip()
        # Short line with markdown header syntax or all bold
        if len(text) < 100 and (text.startswith("#") or text.startswith("**")):
            return True
        # Single short line that's likely a title
        if "\n" not in text and len(text) < 60 and text.endswith(":"):
            return True
        return False

    def _convert_markdown_to_slack(self, text: str) -> str:
        """Convert common markdown to Slack mrkdwn format."""
        # Convert **bold** to *bold* (Slack uses single asterisks)
        text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)

        # Convert headers ## to bold
        text = re.sub(r'^##\s*(.+)$', r'*\1*', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s*(.+)$', r'*\1*', text, flags=re.MULTILINE)

        # Keep inline code as-is (Slack supports `code`)
        # Keep links as-is if in Slack format

        return text

    def _send_response_chunked(
        self, channel: str, thread_ts: str, text: str
    ) -> None:
        """Send response with nice formatting, chunking if necessary.

        Slack has a ~4000 character limit per message.
        """
        max_length = 3000  # Lower limit to account for block overhead

        if len(text) <= max_length:
            # Use formatted blocks for shorter responses
            blocks = self._format_response_blocks(text)
            if blocks:
                self._send_thread_message(channel, thread_ts, text, blocks=blocks)
            else:
                self._send_thread_message(channel, thread_ts, text)
            return

        # For longer responses, split into chunks
        chunks = []
        current_chunk = ""

        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                # If single paragraph is too long, split by lines
                if len(para) > max_length:
                    lines = para.split("\n")
                    current_chunk = ""
                    for line in lines:
                        if len(current_chunk) + len(line) + 1 > max_length:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = line
                        else:
                            current_chunk = (
                                current_chunk + "\n" + line if current_chunk else line
                            )
                else:
                    current_chunk = para
            else:
                current_chunk = (
                    current_chunk + "\n\n" + para if current_chunk else para
                )

        if current_chunk:
            chunks.append(current_chunk)

        # Send chunks with part indicators and formatting
        for i, chunk in enumerate(chunks):
            prefix = f"📄 *Part {i + 1}/{len(chunks)}*\n\n" if len(chunks) > 1 else ""
            formatted_chunk = prefix + chunk
            blocks = self._format_response_blocks(formatted_chunk)
            if blocks:
                self._send_thread_message(channel, thread_ts, formatted_chunk, blocks=blocks)
            else:
                self._send_thread_message(channel, thread_ts, formatted_chunk)

    def _format_session_duration(self, session: ChatSession) -> str:
        """Format session duration as human-readable string."""
        try:
            start = datetime.fromisoformat(
                session.started_at.replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                session.last_activity.replace("Z", "+00:00")
            )
            delta = end - start
            total_seconds = int(delta.total_seconds())

            if total_seconds < 60:
                return "< 1 minute"
            elif total_seconds < 3600:
                minutes = total_seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''}"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours}h {minutes}m"
        except Exception:
            return "unknown"

    def _format_goodbye_message(self, session: ChatSession) -> str:
        """Format goodbye message with session stats."""
        duration = self._format_session_duration(session)
        msg_count = len(session.conversation_history)
        return (
            f"Chat session ended.\n\n"
            f"*Project:* {session.project_name}\n"
            f"*Duration:* {duration}\n"
            f"*Messages:* {msg_count}\n\n"
            f"Thanks for chatting! Use `/chat {session.project_name}` to start a new session."
        )
