"""Telegram message client adapter implementing MessageClient protocol."""

from __future__ import annotations

import re
from typing import Optional

from sleepless_agent.interfaces.message_client import (
    FormattedMessage,
    MessagePlatform,
    MessageResult,
)
from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)

# Import telegram at runtime to avoid dependency issues when not using Telegram
try:
    from telegram import Bot
    from telegram.error import TelegramError
    from telegram.constants import ParseMode

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None  # type: ignore
    TelegramError = Exception  # type: ignore
    ParseMode = None  # type: ignore


class TelegramMessageClient:
    """Telegram implementation of MessageClient protocol.

    Wraps the python-telegram-bot library and converts FormattedMessage
    to Telegram Markdown V2 format.
    """

    def __init__(self, bot: "Bot"):
        """Initialize with a Telegram Bot instance.

        Args:
            bot: Configured telegram.Bot instance
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )
        self._bot = bot

    @property
    def platform(self) -> MessagePlatform:
        """Return Telegram platform identifier."""
        return MessagePlatform.TELEGRAM

    @property
    def raw_bot(self) -> "Bot":
        """Access the underlying Telegram Bot for advanced operations."""
        return self._bot

    def send_message(
        self,
        channel: str,
        message: FormattedMessage | str,
        thread_id: Optional[str] = None,
    ) -> MessageResult:
        """Send a message to a Telegram chat.

        Args:
            channel: Telegram chat_id (can be negative for groups)
            message: FormattedMessage or plain text
            thread_id: Optional message_id to reply to

        Returns:
            MessageResult with message_id
        """
        try:
            if isinstance(message, str):
                text = self._escape_markdown(message)
            else:
                text = self._format_to_markdown(message)

            kwargs = {
                "chat_id": int(channel),
                "text": text,
                "parse_mode": ParseMode.MARKDOWN_V2,
            }
            if thread_id:
                kwargs["reply_to_message_id"] = int(thread_id)

            # Use synchronous send (for async contexts, use send_message_async)
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to handle differently
                future = asyncio.ensure_future(self._bot.send_message(**kwargs))
                # This won't work in sync context, fall back
                raise RuntimeError("Use async method in async context")
            except RuntimeError:
                # No running loop, use synchronous approach
                result = asyncio.run(self._bot.send_message(**kwargs))

            return MessageResult(
                success=True,
                message_id=str(result.message_id),
            )

        except TelegramError as e:
            logger.error("telegram_client.send_message.failed", error=str(e))
            return MessageResult(success=False, error=str(e))
        except Exception as e:
            logger.error("telegram_client.send_message.exception", error=str(e))
            return MessageResult(success=False, error=str(e))

    async def send_message_async(
        self,
        channel: str,
        message: FormattedMessage | str,
        thread_id: Optional[str] = None,
    ) -> MessageResult:
        """Send a message asynchronously.

        Args:
            channel: Telegram chat_id
            message: FormattedMessage or plain text
            thread_id: Optional message_id to reply to

        Returns:
            MessageResult with message_id
        """
        try:
            if isinstance(message, str):
                text = self._escape_markdown(message)
            else:
                text = self._format_to_markdown(message)

            kwargs = {
                "chat_id": int(channel),
                "text": text,
                "parse_mode": ParseMode.MARKDOWN_V2,
            }
            if thread_id:
                kwargs["reply_to_message_id"] = int(thread_id)

            result = await self._bot.send_message(**kwargs)

            return MessageResult(
                success=True,
                message_id=str(result.message_id),
            )

        except TelegramError as e:
            logger.error("telegram_client.send_message_async.failed", error=str(e))
            return MessageResult(success=False, error=str(e))
        except Exception as e:
            logger.error("telegram_client.send_message_async.exception", error=str(e))
            return MessageResult(success=False, error=str(e))

    def delete_message(
        self,
        channel: str,
        message_id: str,
    ) -> bool:
        """Delete a Telegram message.

        Args:
            channel: Telegram chat_id
            message_id: Message ID to delete

        Returns:
            True if deleted successfully
        """
        if not message_id:
            return False

        try:
            import asyncio

            asyncio.run(
                self._bot.delete_message(
                    chat_id=int(channel),
                    message_id=int(message_id),
                )
            )
            return True
        except TelegramError as e:
            logger.debug("telegram_client.delete_message.failed", error=str(e))
            return False
        except Exception as e:
            logger.debug("telegram_client.delete_message.exception", error=str(e))
            return False

    async def delete_message_async(
        self,
        channel: str,
        message_id: str,
    ) -> bool:
        """Delete a message asynchronously."""
        if not message_id:
            return False

        try:
            await self._bot.delete_message(
                chat_id=int(channel),
                message_id=int(message_id),
            )
            return True
        except TelegramError as e:
            logger.debug("telegram_client.delete_message_async.failed", error=str(e))
            return False
        except Exception as e:
            logger.debug("telegram_client.delete_message_async.exception", error=str(e))
            return False

    def add_reaction(
        self,
        channel: str,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Add a reaction to a message.

        Note: Telegram doesn't support reactions in the same way as Slack.
        This is a no-op that returns True for compatibility.
        """
        # Telegram doesn't have emoji reactions like Slack
        # We could use setMessageReaction but it requires specific emoji types
        logger.debug(
            "telegram_client.add_reaction.unsupported",
            hint="Telegram doesn't support reactions in the same way as Slack",
        )
        return True  # Return True for compatibility

    def remove_reaction(
        self,
        channel: str,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Remove a reaction from a message.

        Note: Telegram doesn't support reactions in the same way as Slack.
        This is a no-op that returns True for compatibility.
        """
        logger.debug(
            "telegram_client.remove_reaction.unsupported",
            hint="Telegram doesn't support reactions in the same way as Slack",
        )
        return True  # Return True for compatibility

    def update_message(
        self,
        channel: str,
        message_id: str,
        message: FormattedMessage | str,
    ) -> MessageResult:
        """Update an existing Telegram message.

        Args:
            channel: Telegram chat_id
            message_id: Message ID to update
            message: New message content

        Returns:
            MessageResult with success status
        """
        try:
            if isinstance(message, str):
                text = self._escape_markdown(message)
            else:
                text = self._format_to_markdown(message)

            import asyncio

            asyncio.run(
                self._bot.edit_message_text(
                    chat_id=int(channel),
                    message_id=int(message_id),
                    text=text,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            )
            return MessageResult(success=True, message_id=message_id)

        except TelegramError as e:
            logger.error("telegram_client.update_message.failed", error=str(e))
            return MessageResult(success=False, error=str(e))
        except Exception as e:
            logger.error("telegram_client.update_message.exception", error=str(e))
            return MessageResult(success=False, error=str(e))

    async def update_message_async(
        self,
        channel: str,
        message_id: str,
        message: FormattedMessage | str,
    ) -> MessageResult:
        """Update a message asynchronously."""
        try:
            if isinstance(message, str):
                text = self._escape_markdown(message)
            else:
                text = self._format_to_markdown(message)

            await self._bot.edit_message_text(
                chat_id=int(channel),
                message_id=int(message_id),
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            return MessageResult(success=True, message_id=message_id)

        except TelegramError as e:
            logger.error("telegram_client.update_message_async.failed", error=str(e))
            return MessageResult(success=False, error=str(e))
        except Exception as e:
            logger.error("telegram_client.update_message_async.exception", error=str(e))
            return MessageResult(success=False, error=str(e))

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram Markdown V2.

        Args:
            text: Plain text to escape

        Returns:
            Escaped text safe for Markdown V2
        """
        # Characters that need escaping in Markdown V2
        special_chars = r'_*[]()~`>#+-=|{}.!'
        return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

    def _format_to_markdown(self, message: FormattedMessage) -> str:
        """Convert FormattedMessage to Telegram Markdown V2 format.

        Args:
            message: Platform-agnostic message format

        Returns:
            Telegram Markdown V2 formatted string
        """
        parts = []

        # Header (bold)
        if message.header:
            escaped_header = self._escape_markdown(message.header)
            parts.append(f"*{escaped_header}*")
            parts.append("")  # Empty line after header

        # Fields (key: value format)
        if message.fields:
            for key, value in message.fields.items():
                escaped_key = self._escape_markdown(key)
                escaped_value = self._escape_markdown(value)
                parts.append(f"*{escaped_key}:* {escaped_value}")
            parts.append("")  # Empty line after fields

        # Sections
        for section in message.sections:
            # Convert markdown-style formatting to Telegram format
            formatted = self._convert_markdown_to_telegram(section)
            parts.append(formatted)
            parts.append("")  # Empty line between sections

        # Code blocks
        for language, code in message.code_blocks:
            # Telegram uses triple backticks for code blocks
            # Language hint is not directly supported in Markdown V2
            parts.append(f"```\n{code}\n```")
            parts.append("")

        # Footer (italic)
        if message.footer:
            escaped_footer = self._escape_markdown(message.footer)
            parts.append(f"_{escaped_footer}_")

        return "\n".join(parts).strip()

    def _convert_markdown_to_telegram(self, text: str) -> str:
        """Convert common Markdown to Telegram Markdown V2 format.

        Args:
            text: Markdown text

        Returns:
            Telegram Markdown V2 formatted text
        """
        # First, identify and temporarily replace code blocks and inline code
        # to prevent escaping their contents
        code_blocks = []
        inline_codes = []

        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        def save_inline_code(match):
            inline_codes.append(match.group(0))
            return f"__INLINE_CODE_{len(inline_codes) - 1}__"

        # Save code blocks first (triple backticks)
        text = re.sub(r'```[\s\S]*?```', save_code_block, text)

        # Save inline code (single backticks)
        text = re.sub(r'`[^`]+`', save_inline_code, text)

        # Now escape special characters in the remaining text
        text = self._escape_markdown(text)

        # Convert **bold** to *bold* (Telegram uses single asterisks)
        text = re.sub(r'\\\*\\\*(.+?)\\\*\\\*', r'*\1*', text)

        # Convert __underline__ to underline (already has single underscore in Telegram)
        text = re.sub(r'\\\_\\\_(.+?)\\\_\\\_', r'__\1__', text)

        # Convert headers ## to bold
        text = re.sub(r'^\\#\\#\s*(.+)$', r'*\1*', text, flags=re.MULTILINE)
        text = re.sub(r'^\\#\s*(.+)$', r'*\1*', text, flags=re.MULTILINE)

        # Restore code blocks and inline code
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)

        for i, code in enumerate(inline_codes):
            text = text.replace(f"__INLINE_CODE_{i}__", code)

        return text
