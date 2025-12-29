"""Abstract message client protocol for platform-agnostic messaging.

This module defines the MessageClient protocol that abstracts away
platform-specific messaging implementations (Slack, Telegram, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class MessagePlatform(str, Enum):
    """Supported messaging platforms."""

    SLACK = "slack"
    TELEGRAM = "telegram"


@dataclass
class FormattedMessage:
    """Platform-agnostic message format.

    This class provides a unified way to construct rich messages that
    can be converted to platform-specific formats (Block Kit, Markdown, etc.).

    Attributes:
        text: Plain text fallback (required for accessibility)
        header: Optional header/title text
        sections: List of section texts (will be formatted as paragraphs)
        code_blocks: List of (language, code) tuples for code blocks
        fields: Key-value pairs for structured data display
        actions: List of action buttons/links (platform-dependent support)
        footer: Optional footer text
    """

    text: str
    header: Optional[str] = None
    sections: List[str] = field(default_factory=list)
    code_blocks: List[tuple[str, str]] = field(default_factory=list)  # (language, code)
    fields: Dict[str, str] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    footer: Optional[str] = None

    @classmethod
    def simple(cls, text: str) -> "FormattedMessage":
        """Create a simple text-only message."""
        return cls(text=text)

    @classmethod
    def with_header(cls, header: str, body: str) -> "FormattedMessage":
        """Create a message with header and body."""
        return cls(text=body, header=header, sections=[body])

    @classmethod
    def status(
        cls,
        title: str,
        fields: Dict[str, str],
        footer: Optional[str] = None,
    ) -> "FormattedMessage":
        """Create a status-style message with key-value fields."""
        # Build plain text version
        text_parts = [title]
        for k, v in fields.items():
            text_parts.append(f"{k}: {v}")
        if footer:
            text_parts.append(footer)

        return cls(
            text="\n".join(text_parts),
            header=title,
            fields=fields,
            footer=footer,
        )

    @classmethod
    def help_message(cls, title: str, commands: Dict[str, str]) -> "FormattedMessage":
        """Create a help message with command descriptions."""
        text_parts = [title, ""]
        sections = []

        command_section = []
        for cmd, desc in commands.items():
            command_section.append(f"`{cmd}` - {desc}")
            text_parts.append(f"{cmd} - {desc}")

        sections.append("\n".join(command_section))

        return cls(
            text="\n".join(text_parts),
            header=title,
            sections=sections,
        )


@dataclass
class MessageResult:
    """Result of sending a message.

    Attributes:
        success: Whether the message was sent successfully
        message_id: Platform-specific message identifier (ts for Slack, message_id for Telegram)
        error: Error message if failed
    """

    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


@runtime_checkable
class MessageClient(Protocol):
    """Protocol for platform-agnostic message operations.

    Implementations should convert FormattedMessage to platform-specific
    formats and handle API calls appropriately.
    """

    @property
    def platform(self) -> MessagePlatform:
        """Return the messaging platform type."""
        ...

    def send_message(
        self,
        channel: str,
        message: FormattedMessage | str,
        thread_id: Optional[str] = None,
    ) -> MessageResult:
        """Send a message to a channel/chat.

        Args:
            channel: Channel/chat ID
            message: Message to send (FormattedMessage or plain text)
            thread_id: Optional thread/reply ID for threaded conversations

        Returns:
            MessageResult with success status and message ID
        """
        ...

    def delete_message(
        self,
        channel: str,
        message_id: str,
    ) -> bool:
        """Delete a message by ID.

        Args:
            channel: Channel/chat ID
            message_id: Message identifier to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        ...

    def add_reaction(
        self,
        channel: str,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Add a reaction/emoji to a message.

        Note: Not all platforms support reactions. Implementations should
        gracefully handle unsupported operations.

        Args:
            channel: Channel/chat ID
            message_id: Message identifier
            reaction: Reaction name (e.g., "thumbsup", "white_check_mark")

        Returns:
            True if added successfully, False otherwise
        """
        ...

    def remove_reaction(
        self,
        channel: str,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Remove a reaction from a message.

        Args:
            channel: Channel/chat ID
            message_id: Message identifier
            reaction: Reaction name to remove

        Returns:
            True if removed successfully, False otherwise
        """
        ...

    def update_message(
        self,
        channel: str,
        message_id: str,
        message: FormattedMessage | str,
    ) -> MessageResult:
        """Update an existing message.

        Args:
            channel: Channel/chat ID
            message_id: Message identifier to update
            message: New message content

        Returns:
            MessageResult with success status
        """
        ...
