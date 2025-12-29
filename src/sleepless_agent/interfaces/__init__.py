"""External interfaces - Slack bot, Telegram bot, CLI, and other integrations"""

from .bot import SlackBot
from .cli import main as cli_main
from .message_client import (
    FormattedMessage,
    MessageClient,
    MessagePlatform,
    MessageResult,
)
from .slack_client import SlackMessageClient

# TelegramBot and TelegramMessageClient are optional (require python-telegram-bot)
try:
    from .telegram_bot import TelegramBot
    from .telegram_client import TelegramMessageClient

    TELEGRAM_AVAILABLE = True
except ImportError:
    TelegramBot = None  # type: ignore
    TelegramMessageClient = None  # type: ignore
    TELEGRAM_AVAILABLE = False

__all__ = [
    "SlackBot",
    "TelegramBot",
    "cli_main",
    "FormattedMessage",
    "MessageClient",
    "MessagePlatform",
    "MessageResult",
    "SlackMessageClient",
    "TelegramMessageClient",
    "TELEGRAM_AVAILABLE",
]
