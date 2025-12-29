"""Slack message client adapter implementing MessageClient protocol."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from sleepless_agent.interfaces.message_client import (
    FormattedMessage,
    MessageClient,
    MessagePlatform,
    MessageResult,
)
from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


class SlackMessageClient:
    """Slack implementation of MessageClient protocol.

    Wraps the Slack WebClient and converts FormattedMessage to Block Kit format.
    """

    def __init__(self, web_client: WebClient):
        """Initialize with a Slack WebClient.

        Args:
            web_client: Configured Slack WebClient instance
        """
        self._client = web_client

    @property
    def platform(self) -> MessagePlatform:
        """Return Slack platform identifier."""
        return MessagePlatform.SLACK

    @property
    def raw_client(self) -> WebClient:
        """Access the underlying Slack WebClient for advanced operations."""
        return self._client

    def send_message(
        self,
        channel: str,
        message: FormattedMessage | str,
        thread_id: Optional[str] = None,
    ) -> MessageResult:
        """Send a message to a Slack channel.

        Args:
            channel: Slack channel ID
            message: FormattedMessage or plain text
            thread_id: Thread timestamp for threaded replies

        Returns:
            MessageResult with ts as message_id
        """
        try:
            if isinstance(message, str):
                text = message
                blocks = None
            else:
                text = message.text
                blocks = self._format_to_blocks(message)

            kwargs: Dict[str, Any] = {
                "channel": channel,
                "text": text,
                "mrkdwn": True,
            }
            if thread_id:
                kwargs["thread_ts"] = thread_id
            if blocks:
                kwargs["blocks"] = blocks

            response = self._client.chat_postMessage(**kwargs)
            return MessageResult(
                success=True,
                message_id=response.get("ts"),
            )

        except SlackApiError as e:
            logger.error("slack_client.send_message.failed", error=str(e))
            return MessageResult(success=False, error=str(e))
        except Exception as e:
            logger.error("slack_client.send_message.exception", error=str(e))
            return MessageResult(success=False, error=str(e))

    def delete_message(
        self,
        channel: str,
        message_id: str,
    ) -> bool:
        """Delete a Slack message.

        Args:
            channel: Slack channel ID
            message_id: Message timestamp (ts)

        Returns:
            True if deleted successfully
        """
        if not message_id:
            return False

        try:
            self._client.chat_delete(channel=channel, ts=message_id)
            return True
        except SlackApiError as e:
            logger.debug("slack_client.delete_message.failed", error=str(e))
            return False
        except Exception as e:
            logger.debug("slack_client.delete_message.exception", error=str(e))
            return False

    def add_reaction(
        self,
        channel: str,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Add an emoji reaction to a message.

        Args:
            channel: Slack channel ID
            message_id: Message timestamp (ts)
            reaction: Emoji name (without colons, e.g., "thumbsup")

        Returns:
            True if added successfully
        """
        try:
            self._client.reactions_add(
                channel=channel,
                timestamp=message_id,
                name=reaction,
            )
            return True
        except SlackApiError as e:
            logger.debug("slack_client.add_reaction.failed", error=str(e))
            return False
        except Exception as e:
            logger.debug("slack_client.add_reaction.exception", error=str(e))
            return False

    def remove_reaction(
        self,
        channel: str,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Remove an emoji reaction from a message.

        Args:
            channel: Slack channel ID
            message_id: Message timestamp (ts)
            reaction: Emoji name to remove

        Returns:
            True if removed successfully
        """
        try:
            self._client.reactions_remove(
                channel=channel,
                timestamp=message_id,
                name=reaction,
            )
            return True
        except SlackApiError as e:
            logger.debug("slack_client.remove_reaction.failed", error=str(e))
            return False
        except Exception as e:
            logger.debug("slack_client.remove_reaction.exception", error=str(e))
            return False

    def update_message(
        self,
        channel: str,
        message_id: str,
        message: FormattedMessage | str,
    ) -> MessageResult:
        """Update an existing Slack message.

        Args:
            channel: Slack channel ID
            message_id: Message timestamp (ts)
            message: New message content

        Returns:
            MessageResult with success status
        """
        try:
            if isinstance(message, str):
                text = message
                blocks = None
            else:
                text = message.text
                blocks = self._format_to_blocks(message)

            kwargs: Dict[str, Any] = {
                "channel": channel,
                "ts": message_id,
                "text": text,
            }
            if blocks:
                kwargs["blocks"] = blocks

            self._client.chat_update(**kwargs)
            return MessageResult(success=True, message_id=message_id)

        except SlackApiError as e:
            logger.error("slack_client.update_message.failed", error=str(e))
            return MessageResult(success=False, error=str(e))
        except Exception as e:
            logger.error("slack_client.update_message.exception", error=str(e))
            return MessageResult(success=False, error=str(e))

    def _format_to_blocks(self, message: FormattedMessage) -> List[Dict[str, Any]]:
        """Convert FormattedMessage to Slack Block Kit blocks.

        Args:
            message: Platform-agnostic message format

        Returns:
            List of Slack Block Kit blocks
        """
        blocks: List[Dict[str, Any]] = []

        # Header
        if message.header:
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": message.header[:150],  # Slack header limit
                },
            })

        # Fields (as section with fields)
        if message.fields:
            fields_list = []
            for key, value in message.fields.items():
                fields_list.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}",
                })
            # Slack allows max 10 fields per section
            for i in range(0, len(fields_list), 10):
                blocks.append({
                    "type": "section",
                    "fields": fields_list[i:i + 10],
                })

        # Sections
        for section in message.sections:
            formatted_text = self._convert_markdown_to_slack(section)
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": formatted_text[:3000],  # Slack text limit
                },
            })

        # Code blocks
        for language, code in message.code_blocks:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{code[:2900]}```",  # Leave room for backticks
                },
            })

        # Footer
        if message.footer:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": message.footer,
                    },
                ],
            })

        # Add divider at end if there are blocks
        if blocks:
            blocks.append({"type": "divider"})

        return blocks

    def _convert_markdown_to_slack(self, text: str) -> str:
        """Convert common Markdown to Slack mrkdwn format.

        Args:
            text: Markdown text

        Returns:
            Slack mrkdwn formatted text
        """
        # Convert **bold** to *bold* (Slack uses single asterisks)
        text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)

        # Convert headers ## to bold
        text = re.sub(r'^##\s*(.+)$', r'*\1*', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s*(.+)$', r'*\1*', text, flags=re.MULTILINE)

        return text
