"""Telegram bot interface for task management.

This module provides a Telegram bot implementation that mirrors the SlackBot
functionality, allowing users to interact with sleepless-agent via Telegram.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.core.models import TaskPriority, TaskStatus
from sleepless_agent.core.queue import TaskQueue
from sleepless_agent.tasks.utils import prepare_task_creation, slugify_project
from sleepless_agent.utils.live_status import LiveStatusTracker
from sleepless_agent.monitoring.report_generator import ReportGenerator
from sleepless_agent.chat import ChatSessionManager, ChatExecutor, ChatHandler
from sleepless_agent.interfaces.telegram_client import TelegramMessageClient
from sleepless_agent.utils.display import format_age_seconds, format_duration, shorten

logger = get_logger(__name__)

# Import telegram at runtime to avoid dependency issues
try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )
    from telegram.error import TelegramError

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None  # type: ignore
    Bot = None  # type: ignore
    Application = None  # type: ignore
    CommandHandler = None  # type: ignore
    MessageHandler = None  # type: ignore
    ContextTypes = None  # type: ignore
    filters = None  # type: ignore
    TelegramError = Exception  # type: ignore


class TelegramBot:
    """Telegram bot for task management.

    Provides the same functionality as SlackBot but via Telegram interface:
    - /think: Add a task or thought
    - /chat: Start chat mode for a project
    - /check: View system status
    - /usage: View API usage
    - /cancel: Cancel a task
    - /report: View task reports
    - /trash: Manage trashed tasks
    - /help: Show available commands
    - /start: Welcome message
    """

    def __init__(
        self,
        bot_token: str,
        task_queue: TaskQueue,
        scheduler=None,
        monitor=None,
        report_generator=None,
        live_status_tracker: Optional[LiveStatusTracker] = None,
        workspace_root: str = "./workspace",
        allowed_chat_ids: Optional[Set[int]] = None,
    ):
        """Initialize Telegram bot.

        Args:
            bot_token: Telegram bot token from @BotFather
            task_queue: Task queue for managing tasks
            scheduler: Smart scheduler for budget/timing decisions
            monitor: Health monitor for system status
            report_generator: Report generator for viewing reports
            live_status_tracker: Live status tracker for real-time updates
            workspace_root: Root directory for workspace files
            allowed_chat_ids: Optional set of allowed chat IDs for security
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot>=21.0"
            )

        self.bot_token = bot_token
        self.task_queue = task_queue
        self.scheduler = scheduler
        self.monitor = monitor
        self.report_generator = report_generator
        self.live_status_tracker = live_status_tracker
        self.workspace_root = Path(workspace_root)
        self.allowed_chat_ids = allowed_chat_ids

        # Build the Application
        self.app = Application.builder().token(bot_token).build()

        # Initialize chat mode components
        chat_sessions_path = self.workspace_root / "data" / "chat_sessions.json"
        self.chat_session_manager = ChatSessionManager(storage_path=chat_sessions_path)
        self.chat_executor = ChatExecutor(workspace_root=str(self.workspace_root))

        # Message client will be created when bot starts (needs bot instance)
        self.message_client: Optional[TelegramMessageClient] = None
        self.chat_handler: Optional[ChatHandler] = None

        # Register handlers
        self._register_handlers()

        logger.info("telegram_bot.initialized")

    def _register_handlers(self):
        """Register all command and message handlers."""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("help", self._handle_help))
        self.app.add_handler(CommandHandler("think", self._handle_think))
        self.app.add_handler(CommandHandler("chat", self._handle_chat))
        self.app.add_handler(CommandHandler("check", self._handle_check))
        self.app.add_handler(CommandHandler("usage", self._handle_usage))
        self.app.add_handler(CommandHandler("cancel", self._handle_cancel))
        self.app.add_handler(CommandHandler("report", self._handle_report))
        self.app.add_handler(CommandHandler("trash", self._handle_trash))

        # Message handler for chat mode replies
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        logger.debug("telegram_bot.handlers.registered")

    async def _check_authorization(self, update: Update) -> bool:
        """Check if the user is authorized to use the bot.

        Args:
            update: Telegram update object

        Returns:
            True if authorized, False otherwise
        """
        if not self.allowed_chat_ids:
            return True  # No restrictions

        chat_id = update.effective_chat.id
        if chat_id not in self.allowed_chat_ids:
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            logger.warning(
                "telegram_bot.unauthorized",
                chat_id=chat_id,
                user=update.effective_user.username,
            )
            return False

        return True

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not await self._check_authorization(update):
            return

        welcome_text = (
            "*Welcome to Sleepless Agent\\!*\n\n"
            "I'm your 24/7 AI assistant for task management\\.\n\n"
            "*Available Commands:*\n"
            "`/think <description>` \\- Add a task or thought\n"
            "`/chat <project>` \\- Start chat mode\n"
            "`/check` \\- View system status\n"
            "`/usage` \\- View API usage\n"
            "`/cancel <id>` \\- Cancel a task\n"
            "`/report` \\- View reports\n"
            "`/trash` \\- Manage trashed tasks\n"
            "`/help` \\- Show this help\n\n"
            "_Type /help for more details\\._"
        )

        await update.message.reply_text(welcome_text, parse_mode="MarkdownV2")
        logger.info("telegram_bot.start", user=update.effective_user.username)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not await self._check_authorization(update):
            return

        help_text = (
            "*Sleepless Agent Commands*\n\n"
            "*Task Management:*\n"
            "`/think <description>` \\- Add a quick thought or task\n"
            "`/think <desc> \\-\\-project=<name>` \\- Add to a project\n"
            "`/cancel <task_id>` \\- Cancel a pending task\n"
            "`/cancel <project>` \\- Cancel all tasks in project\n\n"
            "*Chat Mode:*\n"
            "`/chat <project>` \\- Start interactive chat\n"
            "`/chat end` \\- End chat session\n"
            "`/chat status` \\- Check session status\n\n"
            "*Monitoring:*\n"
            "`/check` \\- View system status\n"
            "`/usage` \\- View API usage stats\n"
            "`/report` \\- View today's report\n"
            "`/report <date>` \\- View report for date\n\n"
            "*Trash Management:*\n"
            "`/trash list` \\- List trashed items\n"
            "`/trash restore <id>` \\- Restore item\n"
            "`/trash empty` \\- Empty trash\n"
        )

        await update.message.reply_text(help_text, parse_mode="MarkdownV2")

    async def _handle_think(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /think command - add a task or thought."""
        if not await self._check_authorization(update):
            return

        args = " ".join(context.args) if context.args else ""

        if not args:
            await update.message.reply_text(
                "Usage: `/think <description> [--project=<name>]`",
                parse_mode="MarkdownV2",
            )
            return

        # Parse task creation arguments
        (
            cleaned_description,
            project_name,
            project_id,
            note,
        ) = prepare_task_creation(args)

        if not cleaned_description.strip():
            await update.message.reply_text("Please provide a description\\.")
            return

        # Determine priority based on whether project is provided
        priority = TaskPriority.SERIOUS if project_id else TaskPriority.THOUGHT

        try:
            task = self.task_queue.add_task(
                description=cleaned_description.strip(),
                priority=priority,
                assigned_to=str(update.effective_user.id),
                project_id=project_id,
                project_name=project_name,
            )

            # Build response
            emoji = "💡" if priority == TaskPriority.THOUGHT else "📋"
            priority_label = "Thought" if priority == TaskPriority.THOUGHT else "Task"

            response = f"{emoji} *{priority_label} Added*\n\n"
            response += f"*ID:* `{task.id}`\n"
            response += f"*Description:* {self._escape_markdown(shorten(task.description, 100))}\n"

            if project_name:
                response += f"*Project:* {self._escape_markdown(project_name)}\n"

            queue_position = self.task_queue.get_queue_position(task.id)
            response += f"*Queue Position:* {queue_position}"

            await update.message.reply_text(
                response, parse_mode="MarkdownV2"
            )

            logger.info(
                "telegram_bot.think.added",
                task_id=task.id,
                priority=priority.value,
                project=project_id,
            )

        except Exception as e:
            logger.error("telegram_bot.think.error", error=str(e))
            await update.message.reply_text(f"Error adding task: {str(e)}")

    async def _handle_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chat command - start or manage chat mode."""
        if not await self._check_authorization(update):
            return

        if not self.chat_handler:
            await update.message.reply_text(
                "Chat mode is not available\\. Bot initialization incomplete\\.",
                parse_mode="MarkdownV2",
            )
            return

        args = " ".join(context.args) if context.args else ""
        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        # Handle subcommands
        args_lower = args.lower().strip()

        if args_lower in ("end", "exit", "quit"):
            session = self.chat_session_manager.end_session(user_id)
            if session:
                await update.message.reply_text(
                    f"Chat session for *{self._escape_markdown(session.project_name)}* ended\\.",
                    parse_mode="MarkdownV2",
                )
            else:
                await update.message.reply_text("No active chat session to end\\.")
            return

        if args_lower == "status":
            session = self.chat_session_manager.get_session(user_id)
            if session:
                await update.message.reply_text(
                    f"*Active Session:* {self._escape_markdown(session.project_name)}\n"
                    f"*Messages:* {len(session.conversation_history)}",
                    parse_mode="MarkdownV2",
                )
            else:
                await update.message.reply_text("No active chat session\\.")
            return

        if args_lower == "help":
            await update.message.reply_text(
                "*Chat Mode Commands:*\n\n"
                "`/chat <project>` \\- Start chat for project\n"
                "`/chat end` \\- End session\n"
                "`/chat status` \\- Check status\n\n"
                "_In chat mode, reply to the session message to chat with Claude\\._",
                parse_mode="MarkdownV2",
            )
            return

        if not args:
            await update.message.reply_text(
                "Usage: `/chat <project_name>`",
                parse_mode="MarkdownV2",
            )
            return

        # Check for existing session
        existing = self.chat_session_manager.get_session(user_id)
        if existing:
            await update.message.reply_text(
                f"You already have an active session for *{self._escape_markdown(existing.project_name)}*\\.\n"
                "Use `/chat end` first\\.",
                parse_mode="MarkdownV2",
            )
            return

        # Start new session
        project_name = args
        project_id = slugify_project(project_name)

        session = self.chat_session_manager.create_session(
            user_id=user_id,
            channel_id=chat_id,
            thread_ts=str(update.message.message_id),  # Use message_id as thread
            project_id=project_id,
            project_name=project_name,
            workspace_path=str(self.workspace_root / "projects" / project_id),
        )

        await update.message.reply_text(
            f"*Chat Mode Started*\n\n"
            f"*Project:* {self._escape_markdown(project_name)}\n\n"
            f"_Reply to this message to chat with Claude\\._\n"
            f"_Type `exit` or use `/chat end` to end the session\\._",
            parse_mode="MarkdownV2",
        )

        logger.info(
            "telegram_bot.chat.started",
            user=user_id,
            project=project_id,
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages - check for chat mode replies."""
        if not await self._check_authorization(update):
            return

        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)
        text = update.message.text

        # Check for chat session
        session = self.chat_session_manager.get_session(user_id)
        if not session:
            # Not in chat mode, ignore regular messages
            return

        # Check for exit command
        if text.lower().strip() in ("exit", "end", "quit"):
            self.chat_session_manager.end_session(user_id)
            await update.message.reply_text(
                f"Chat session for *{self._escape_markdown(session.project_name)}* ended\\.",
                parse_mode="MarkdownV2",
            )
            return

        # Process chat message
        processing_msg = await update.message.reply_text("🔄 Processing your message...")

        try:
            response, metrics = await self.chat_executor.execute_chat_turn_with_timeout(
                session, text, timeout_seconds=300
            )

            # Delete processing indicator
            await processing_msg.delete()

            # Send response (handle long messages)
            await self._send_long_message(update, response)

        except Exception as e:
            logger.error("telegram_bot.chat.error", error=str(e))
            await processing_msg.delete()
            await update.message.reply_text(f"Error: {str(e)}")

    async def _handle_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /check command - show system status."""
        if not await self._check_authorization(update):
            return

        try:
            # Get queue stats
            pending = len(self.task_queue.get_pending_tasks(limit=100))
            in_progress = len(self.task_queue.get_in_progress_tasks())
            completed_today = self.task_queue.get_completed_count_today()

            status_text = "*System Status*\n\n"
            status_text += f"*Pending Tasks:* {pending}\n"
            status_text += f"*In Progress:* {in_progress}\n"
            status_text += f"*Completed Today:* {completed_today}\n"

            # Add scheduler info if available
            if self.scheduler:
                can_execute = self.scheduler.should_execute_next_task()
                status_text += f"\n*Can Execute:* {'Yes' if can_execute else 'No'}\n"

            # Add monitor info if available
            if self.monitor:
                health = self.monitor.get_health_summary()
                status_text += f"\n*System Health:* {self._escape_markdown(health.get('status', 'Unknown'))}\n"

            await update.message.reply_text(status_text, parse_mode="MarkdownV2")

        except Exception as e:
            logger.error("telegram_bot.check.error", error=str(e))
            await update.message.reply_text(f"Error getting status: {str(e)}")

    async def _handle_usage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /usage command - show API usage."""
        if not await self._check_authorization(update):
            return

        try:
            from sleepless_agent.utils.zhipu_env import get_usage_checker

            checker = get_usage_checker()
            usage_percent, reset_time = checker.get_usage()

            usage_text = "*API Usage*\n\n"
            usage_text += f"*Current Usage:* {usage_percent:.1f}%\n"

            if reset_time:
                reset_str = reset_time.strftime("%H:%M")
                usage_text += f"*Resets At:* {self._escape_markdown(reset_str)}\n"

            # Add threshold info
            from sleepless_agent.utils.config import get_config
            config = get_config()

            from sleepless_agent.utils.zhipu_env import is_zhipu_enabled
            if is_zhipu_enabled():
                threshold = config.zhipu.threshold_day
            else:
                threshold = config.claude_code.threshold_day

            usage_text += f"*Threshold:* {threshold:.1f}%\n"

            if usage_percent >= threshold:
                usage_text += "\n⚠️ _Usage threshold exceeded\\. Tasks paused\\._"

            await update.message.reply_text(usage_text, parse_mode="MarkdownV2")

        except Exception as e:
            logger.error("telegram_bot.usage.error", error=str(e))
            await update.message.reply_text(f"Error getting usage: {str(e)}")

    async def _handle_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cancel command - cancel a task or project."""
        if not await self._check_authorization(update):
            return

        args = " ".join(context.args) if context.args else ""

        if not args:
            await update.message.reply_text(
                "Usage: `/cancel <task_id>` or `/cancel <project_name>`",
                parse_mode="MarkdownV2",
            )
            return

        try:
            # Try to parse as task ID
            try:
                task_id = int(args)
                task = self.task_queue.get_task(task_id)
                if task:
                    if task.status == TaskStatus.PENDING:
                        self.task_queue.cancel_task(task_id)
                        await update.message.reply_text(
                            f"Task `{task_id}` cancelled\\.",
                            parse_mode="MarkdownV2",
                        )
                    else:
                        await update.message.reply_text(
                            f"Task `{task_id}` is {task.status.value}, cannot cancel\\.",
                            parse_mode="MarkdownV2",
                        )
                else:
                    await update.message.reply_text(f"Task `{task_id}` not found\\.")
                return
            except ValueError:
                pass

            # Treat as project name
            project_id = slugify_project(args)
            count = self.task_queue.cancel_project_tasks(project_id)

            if count > 0:
                await update.message.reply_text(
                    f"Cancelled {count} task\\(s\\) for project *{self._escape_markdown(args)}*\\.",
                    parse_mode="MarkdownV2",
                )
            else:
                await update.message.reply_text(
                    f"No pending tasks found for project *{self._escape_markdown(args)}*\\.",
                    parse_mode="MarkdownV2",
                )

        except Exception as e:
            logger.error("telegram_bot.cancel.error", error=str(e))
            await update.message.reply_text(f"Error: {str(e)}")

    async def _handle_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report command - show task reports."""
        if not await self._check_authorization(update):
            return

        if not self.report_generator:
            await update.message.reply_text("Report generator not available\\.")
            return

        args = " ".join(context.args) if context.args else ""

        try:
            if args:
                # Try to parse as date
                try:
                    from datetime import datetime
                    date = datetime.strptime(args, "%Y-%m-%d").date()
                    report = self.report_generator.generate_daily_report(date)
                except ValueError:
                    # Treat as project name
                    project_id = slugify_project(args)
                    report = self.report_generator.generate_project_report(project_id)
            else:
                # Today's report
                report = self.report_generator.generate_daily_report()

            # Send report (may be long)
            await self._send_long_message(update, report)

        except Exception as e:
            logger.error("telegram_bot.report.error", error=str(e))
            await update.message.reply_text(f"Error generating report: {str(e)}")

    async def _handle_trash(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trash command - manage trashed tasks."""
        if not await self._check_authorization(update):
            return

        args = context.args if context.args else []

        if not args or args[0] == "list":
            # List trashed items
            trashed = self.task_queue.get_trashed_tasks(limit=10)
            if not trashed:
                await update.message.reply_text("Trash is empty\\.")
                return

            text = "*Trashed Tasks*\n\n"
            for task in trashed:
                text += f"`{task.id}` \\- {self._escape_markdown(shorten(task.description, 50))}\n"

            text += "\n_Use `/trash restore <id>` to restore\\._"
            await update.message.reply_text(text, parse_mode="MarkdownV2")

        elif args[0] == "restore" and len(args) > 1:
            try:
                task_id = int(args[1])
                if self.task_queue.restore_task(task_id):
                    await update.message.reply_text(f"Task `{task_id}` restored\\.")
                else:
                    await update.message.reply_text(f"Task `{task_id}` not found in trash\\.")
            except ValueError:
                await update.message.reply_text("Invalid task ID\\.")

        elif args[0] == "empty":
            count = self.task_queue.empty_trash()
            await update.message.reply_text(f"Deleted {count} task\\(s\\) permanently\\.")

        else:
            await update.message.reply_text(
                "Usage:\n"
                "`/trash list` \\- List trashed items\n"
                "`/trash restore <id>` \\- Restore item\n"
                "`/trash empty` \\- Empty trash",
                parse_mode="MarkdownV2",
            )

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4000):
        """Send a potentially long message, splitting if necessary."""
        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        # Split into chunks
        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > max_length:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line

        if current:
            chunks.append(current)

        for i, chunk in enumerate(chunks):
            prefix = f"[Part {i + 1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
            await update.message.reply_text(prefix + chunk)

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram Markdown V2."""
        import re
        special_chars = r'_*[]()~`>#+-=|{}.!'
        return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

    def start(self):
        """Start the bot (blocking).

        Note: When running in a background thread (as in daemon.py),
        we must disable signal handlers since they only work in the main thread.
        """
        logger.info("telegram_bot.starting")

        # Initialize message client with the bot instance
        self.message_client = TelegramMessageClient(self.app.bot)

        # Initialize chat handler
        self.chat_handler = ChatHandler(
            session_manager=self.chat_session_manager,
            chat_executor=self.chat_executor,
            task_queue=self.task_queue,
            message_client=self.message_client,
        )

        # Run the bot with polling
        # stop_signals=None disables signal handlers (required for background threads)
        self.app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            stop_signals=None,
        )

    async def start_async(self):
        """Start the bot asynchronously."""
        logger.info("telegram_bot.starting_async")

        # Initialize message client
        self.message_client = TelegramMessageClient(self.app.bot)

        # Initialize chat handler
        self.chat_handler = ChatHandler(
            session_manager=self.chat_session_manager,
            chat_executor=self.chat_executor,
            task_queue=self.task_queue,
            message_client=self.message_client,
        )

        # Start polling
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    def stop(self):
        """Stop the bot."""
        logger.info("telegram_bot.stopping")
        # Application cleanup is handled automatically

    async def stop_async(self):
        """Stop the bot asynchronously."""
        logger.info("telegram_bot.stopping_async")
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
