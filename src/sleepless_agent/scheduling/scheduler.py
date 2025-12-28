"""Smart task scheduler with usage tracking and time-based quotas"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from sleepless_agent.scheduling.time_utils import (
    current_period_start,
    get_time_label,
    is_nighttime,
)
from sleepless_agent.monitoring.logging import get_logger

from sleepless_agent.core.models import Task, TaskPriority, TaskStatus, UsageMetric
from sleepless_agent.core.queue import TaskQueue

logger = get_logger(__name__)


class BudgetManager:
    """Manage daily/monthly budgets with time-based allocation"""

    def __init__(
        self,
        session: Session,
        daily_budget_usd: float = 10.0,
        night_quota_percent: float = 90.0,
    ):
        """Initialize budget manager

        Args:
            session: Database session for querying usage
            daily_budget_usd: Daily budget in USD (default: $10)
            night_quota_percent: Percentage of daily budget for nighttime (default: 90%)
        """
        self.session = session
        self.daily_budget_usd = Decimal(str(daily_budget_usd))
        self.night_quota_percent = Decimal(str(night_quota_percent))
        self.day_quota_percent = Decimal("100") - self.night_quota_percent

    def get_usage_in_period(
        self, start_time: datetime, end_time: Optional[datetime] = None
    ) -> Decimal:
        """Get total usage in USD for a time period"""
        if end_time is None:
            end_time = datetime.now(timezone.utc).replace(tzinfo=None)

        metrics = (
            self.session.query(UsageMetric)
            .filter(
                UsageMetric.created_at >= start_time,
                UsageMetric.created_at < end_time,
            )
            .all()
        )

        total = Decimal("0")
        for metric in metrics:
            if metric.total_cost_usd:
                try:
                    total += Decimal(metric.total_cost_usd)
                except Exception as e:
                    logger.warning(
                        "budget.parse_cost_failed",
                        cost=metric.total_cost_usd,
                        error=str(e),
                    )

        return total

    def get_today_usage(self) -> Decimal:
        """Get total usage for today (UTC midnight to now)"""
        today_start = datetime.now(timezone.utc).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_usage_in_period(today_start)

    def get_current_time_period_usage(self) -> Decimal:
        """Get usage for current time period (night or day)"""
        period_start = current_period_start(datetime.now(timezone.utc).replace(tzinfo=None))
        return self.get_usage_in_period(period_start)

    def get_current_quota(self) -> Decimal:
        """Get budget quota for current time period"""
        if is_nighttime():
            quota = self.daily_budget_usd * (self.night_quota_percent / Decimal("100"))
        else:
            quota = self.daily_budget_usd * (self.day_quota_percent / Decimal("100"))

        return quota

    def get_remaining_budget(self) -> Decimal:
        """Get remaining budget for current time period"""
        quota = self.get_current_quota()
        usage = self.get_current_time_period_usage()
        remaining = quota - usage
        return max(Decimal("0"), remaining)

    def is_budget_available(self, estimated_cost: Decimal = Decimal("0.50")) -> bool:
        """Check if budget is available for a task

        Args:
            estimated_cost: Estimated cost in USD (default: $0.50 per task)

        Returns:
            True if budget available, False otherwise
        """
        remaining = self.get_remaining_budget()
        return remaining >= estimated_cost

    def get_usage_percent(self) -> int:
        """Get current usage as percentage of current quota (0-100)"""
        quota = self.get_current_quota()
        usage = self.get_current_time_period_usage()

        if quota == 0:
            return 0

        percent = (usage / quota) * 100
        return min(100, int(percent))

    def get_budget_status(self) -> dict:
        """Get comprehensive budget status"""
        is_night = is_nighttime()
        time_label = get_time_label()

        quota = self.get_current_quota()
        usage = self.get_current_time_period_usage()
        remaining = self.get_remaining_budget()
        today_usage = self.get_today_usage()

        return {
            "time_period": time_label,
            "is_nighttime": is_night,
            "daily_budget_usd": float(self.daily_budget_usd),
            "current_quota_usd": float(quota),
            "current_usage_usd": float(usage),
            "remaining_budget_usd": float(remaining),
            "today_total_usage_usd": float(today_usage),
            "quota_allocation": {
                "night_percent": float(self.night_quota_percent),
                "day_percent": float(self.day_quota_percent),
            },
        }


class CreditWindow:
    """Tracks credit usage in 5-hour windows (legacy, kept for backwards compatibility)"""

    WINDOW_SIZE_HOURS = 5

    def __init__(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        """Initialize credit window

        Args:
            start_time: Window start time (default: now)
            end_time: Window end time (default: start_time + 5 hours)
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc).replace(tzinfo=None)

        self.start_time = start_time

        if end_time is not None:
            self.end_time = end_time
        else:
            self.end_time = start_time + timedelta(hours=self.WINDOW_SIZE_HOURS)

        self.tasks_executed = 0
        self.estimated_credits_used = 0

    def is_active(self) -> bool:
        """Check if window is still active"""
        return datetime.now(timezone.utc).replace(tzinfo=None) < self.end_time

    def time_remaining_minutes(self) -> int:
        """Get minutes remaining in window"""
        remaining = (self.end_time - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds() / 60
        return max(0, int(remaining))

    def __repr__(self):
        return f"<CreditWindow({self.tasks_executed} tasks, {self.time_remaining_minutes()}m left)>"


class SmartScheduler:
    """Intelligent task scheduler with budget management and time-based quotas"""

    def __init__(
        self,
        task_queue: TaskQueue,
        max_parallel_tasks: int = 1,
        daily_budget_usd: float = 10.0,
        night_quota_percent: float = 90.0,
        usage_command: str = "/usage",
        threshold_day: float = 20.0,
        threshold_night: float = 80.0,
        night_start_hour: int = 20,
        night_end_hour: int = 8,
    ):
        """Initialize scheduler

        Args:
            task_queue: Task queue instance
            max_parallel_tasks: Maximum parallel tasks (default: 1)
            daily_budget_usd: Daily budget in USD (default: $10)
            night_quota_percent: Percentage for night usage (default: 90%)
            usage_command: CLI command to check usage (default: "/usage")
            threshold_day: Pause threshold during daytime (default: 20%)
            threshold_night: Pause threshold during nighttime (default: 80%)
            night_start_hour: Hour when night starts (default: 20 for 8 PM)
            night_end_hour: Hour when night ends (default: 8 for 8 AM)
        """
        self.task_queue = task_queue
        self.max_parallel_tasks = max_parallel_tasks
        self.usage_command = usage_command
        self.threshold_day = threshold_day
        self.threshold_night = threshold_night
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour

        # Budget management with time-based allocation
        session = self.task_queue.SessionLocal()
        self.budget_manager = BudgetManager(
            session=session,
            daily_budget_usd=daily_budget_usd,
            night_quota_percent=night_quota_percent,
        )

        # Usage checker (uses ZhipuUsageChecker if USE_ZHIPU=true, else ProPlanUsageChecker)
        try:
            from sleepless_agent.utils.zhipu_env import get_usage_checker
            self.usage_checker = get_usage_checker()
            logger.debug("scheduler.usage_checker.ready")
        except ImportError:
            logger.error("scheduler.usage_checker.unavailable")
            raise RuntimeError("Usage checker not available - required for scheduling")

        # Legacy credit window support
        self.active_windows: List[CreditWindow] = []
        self.current_window: Optional[CreditWindow] = None
        self._init_current_window()
        self._last_budget_exhausted_log: Optional[datetime] = None
        self._budget_exhausted_logged = False
        self.usage_pause_until: Optional[datetime] = None
        self._usage_pause_grace = timedelta(minutes=1)
        self._usage_pause_default = timedelta(minutes=5)

    def _init_current_window(self):
        """Initialize current credit window"""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Check if we need a new window
        if not self.current_window or not self.current_window.is_active():
            # Get actual reset time from usage checker
            reset_time = None
            try:
                _, reset_time = self.usage_checker.get_usage()
            except Exception as e:
                logger.debug("scheduler.credit_window.reset_time_unavailable", error=str(e))

            # Create window with actual reset time or fall back to 5-hour window
            self.current_window = CreditWindow(start_time=now, end_time=reset_time)
            self.active_windows.append(self.current_window)
            logger.debug(
                "scheduler.credit_window.new",
                window_start=self.current_window.start_time.isoformat(),
                window_end=self.current_window.end_time.isoformat(),
                minutes_left=self.current_window.time_remaining_minutes(),
            )

    def _check_scheduling_allowed(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if scheduling is allowed based on live usage monitoring

        Returns:
            Tuple of (should_schedule: bool, context: dict)
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Check if we're in a pause window
        if self.usage_pause_until:
            if now < self.usage_pause_until:
                remaining = self.usage_pause_until - now
                context = {
                    "event": "scheduler.pause.pending",
                    "reason": "usage_pause",
                    "resume_at": self.usage_pause_until.replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                    "detail": self._format_remaining(remaining),
                    "decision_logic": "Pausing: in pause window, waiting for resume time",
                }
                return False, context
            # Pause window has expired; resume normal checks.
            self.usage_pause_until = None

        # Check live usage (mandatory)
        try:
            usage_percent, reset_time = self.usage_checker.get_usage()
            effective_threshold = self._get_effective_threshold()

            if usage_percent >= effective_threshold:
                pause_base = (
                    reset_time
                    if reset_time and reset_time > now
                    else now + self._usage_pause_default
                )
                pause_until = pause_base + self._usage_pause_grace
                self.usage_pause_until = pause_until
                remaining = pause_until - now
                time_period = "nighttime" if is_nighttime() else "daytime"
                context = {
                    "event": "scheduler.pause.usage_threshold",
                    "reason": "usage_threshold",
                    "usage_percent": usage_percent,
                    "threshold_percent": effective_threshold,
                    "resume_at": pause_until.replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                    "detail": self._format_remaining(remaining),
                    "decision_logic": f"Pausing: usage {usage_percent}% >= threshold {effective_threshold}% ({time_period})",
                }
                return False, context
            else:
                self.usage_pause_until = None
                return True, {
                    "event": "scheduler.usage.ok",
                    "reason": "usage_ok",
                    "usage_percent": usage_percent,
                    "threshold_percent": effective_threshold,
                }

        except Exception as e:
            logger.error("scheduler.usage.check_failed", error=str(e))
            # No fallback - usage checking is mandatory
            context = {
                "event": "scheduler.pause.usage_check_failed",
                "reason": "usage_check_failed",
                "error": str(e),
            }
            return False, context

    def _get_effective_threshold(self) -> float:
        """Get threshold based on current time period

        Returns:
            Threshold percentage (0-100) from config:
            - Daytime: threshold_day
            - Nighttime: threshold_night
        """
        return self.threshold_night if is_nighttime(night_start_hour=self.night_start_hour, night_end_hour=self.night_end_hour) else self.threshold_day

    @staticmethod
    def _format_remaining(delta: timedelta) -> str:
        """Render a short human-readable remaining time string."""
        total_seconds = int(max(delta.total_seconds(), 0))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if not parts and seconds:
            parts.append(f"{seconds}s")
        return " ".join(parts) if parts else "0s"

    def get_pause_remaining_seconds(self) -> Optional[float]:
        """Return remaining pause duration in seconds if scheduling is halted."""
        if not self.usage_pause_until:
            return None
        remaining = (self.usage_pause_until - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        return remaining if remaining > 0 else None

    def get_next_tasks(self) -> List[Task]:
        """Get next tasks to execute respecting concurrency, priorities, and budget"""
        self._init_current_window()

        # Check if we should schedule tasks using live usage or budget
        should_schedule, context = self._check_scheduling_allowed()

        if not should_schedule:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            event = context.pop("event", "scheduler.pause")
            reason = context.get("reason")
            should_log = True
            if reason in {"budget_exhausted", "budget_insufficient"}:
                if (
                    self._budget_exhausted_logged
                    and self._last_budget_exhausted_log
                    and (now - self._last_budget_exhausted_log).total_seconds() < 60
                ):
                    should_log = False
                if should_log:
                    self._budget_exhausted_logged = True
                    self._last_budget_exhausted_log = now
            if should_log:
                logger.warning(event, **context)
            else:
                logger.debug(event, **context)
            return []
        else:
            if self._budget_exhausted_logged:
                logger.info("scheduler.resume", **{k: v for k, v in context.items() if k != "event"})
            self._budget_exhausted_logged = False
            self._last_budget_exhausted_log = None

        # Get in-progress tasks
        in_progress = self.task_queue.get_in_progress_tasks()
        available_slots = max(0, self.max_parallel_tasks - len(in_progress))

        if available_slots == 0:
            return []

        # Get pending tasks in priority order
        pending = self.task_queue.get_pending_tasks(limit=available_slots)

        # Enhanced dispatch log with detailed decision-making context
        if pending:
            # Get queue status
            queue_status = self.task_queue.get_queue_status()

            # Get time context
            time_label = get_time_label()
            is_night = is_nighttime()

            # Build comprehensive log payload explaining the scheduling decision
            payload: Dict[str, Any] = {
                "dispatching_tasks": len(pending),
                "time_period": time_label,
                "is_nighttime": is_night,
            }

            # Add usage information if available (live usage check)
            if context.get("usage_percent") is not None:
                effective_threshold = context.get("threshold_percent", self._get_effective_threshold())
                payload["usage_percent"] = context["usage_percent"]
                payload["threshold_percent"] = effective_threshold
                payload["decision_reason"] = f"usage {context['usage_percent']}% < threshold {effective_threshold}%"

            # Add budget information
            if self.budget_manager:
                quota = self.budget_manager.get_current_quota()
                usage = self.budget_manager.get_current_time_period_usage()
                remaining = self.budget_manager.get_remaining_budget()
                payload["quota_usd"] = float(quota)
                payload["usage_usd"] = float(usage)
                payload["remaining_usd"] = float(remaining)

                # Add budget allocation context
                if is_night:
                    payload["quota_allocation"] = f"night: {self.budget_manager.night_quota_percent}%"
                else:
                    payload["quota_allocation"] = f"daytime: {self.budget_manager.day_quota_percent}%"

            # Add queue status
            payload["queue_pending"] = queue_status.get("pending", 0)
            payload["queue_in_progress"] = queue_status.get("in_progress", 0)
            payload["available_slots"] = available_slots

            logger.info("scheduler.dispatch", **payload)

        # Filter out tasks that would conflict with currently executing tasks
        # (e.g., REFINE tasks targeting a workspace that's already in use)
        non_conflicting_tasks = self._filter_workspace_conflicts(in_progress, pending)

        if len(non_conflicting_tasks) < len(pending):
            filtered_count = len(pending) - len(non_conflicting_tasks)
            logger.debug(
                "scheduler.workspace_conflict",
                filtered_count=filtered_count,
                dispatching=len(non_conflicting_tasks)
            )

        return non_conflicting_tasks

    def _get_task_workspace_identifier(self, task: Task) -> str:
        """Get workspace identifier for a task

        Args:
            task: Task object

        Returns:
            Workspace identifier (e.g., "task:2", "project:myproject")
        """
        import json

        # For project tasks, use project workspace
        if task.project_id:
            return f"project:{task.project_id}"

        # Check if this task refines another task
        if task.context:
            try:
                context = json.loads(task.context)
                refines_task_id = context.get('refines_task_id')
                if refines_task_id is not None:
                    # REFINE task uses the target task's workspace
                    return f"task:{refines_task_id}"
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        # Regular task uses its own workspace
        return f"task:{task.id}"

    def _filter_workspace_conflicts(self, in_progress: List[Task], pending: List[Task]) -> List[Task]:
        """Filter pending tasks to avoid workspace conflicts

        Args:
            in_progress: Currently executing tasks
            pending: Pending tasks to potentially dispatch

        Returns:
            Filtered list of pending tasks that won't conflict with in-progress tasks
        """
        # Get workspaces currently in use
        busy_workspaces = {self._get_task_workspace_identifier(task) for task in in_progress}

        # Filter pending tasks to exclude those targeting busy workspaces
        non_conflicting = []
        for task in pending:
            task_workspace = self._get_task_workspace_identifier(task)
            if task_workspace not in busy_workspaces:
                non_conflicting.append(task)
                # Mark this workspace as busy for subsequent checks in this batch
                busy_workspaces.add(task_workspace)

        return non_conflicting

    def schedule_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.THOUGHT,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Task:
        """Schedule a new task"""
        task = self.task_queue.add_task(
            description=description,
            priority=priority,
            project_id=project_id,
            project_name=project_name,
        )

        # Log scheduling decision
        project_info = f" [Project: {project_name}]" if project_name else ""
        if priority == TaskPriority.SERIOUS:
            logger.info(
                "scheduler.task.scheduled",
                task_id=task.id,
                priority="serious",
                project=project_name,
            )
        elif priority == TaskPriority.THOUGHT:
            logger.info(
                "scheduler.task.scheduled",
                task_id=task.id,
                priority="thought",
                project=project_name,
            )
        else:
            logger.info(
                "scheduler.task.scheduled",
                task_id=task.id,
                priority="generated",
                project=project_name,
            )

        return task

    def record_task_usage(
        self,
        task_id: int,
        total_cost_usd: Optional[float] = None,
        duration_ms: Optional[int] = None,
        duration_api_ms: Optional[int] = None,
        num_turns: Optional[int] = None,
        project_id: Optional[str] = None,
    ):
        """Record API usage metrics for a completed task

        Args:
            task_id: Task ID
            total_cost_usd: Total cost in USD
            duration_ms: Total duration in milliseconds
            duration_api_ms: API call duration
            num_turns: Number of conversation turns
            project_id: Optional project ID for aggregation
        """
        session = self.task_queue.SessionLocal()
        try:
            usage = UsageMetric(
                task_id=task_id,
                total_cost_usd=str(total_cost_usd) if total_cost_usd is not None else None,
                duration_ms=duration_ms,
                duration_api_ms=duration_api_ms,
                num_turns=num_turns,
                project_id=project_id,
            )
            session.add(usage)
            session.commit()

            # Move to DEBUG - usage recording is an internal metric
            if total_cost_usd is not None:
                logger.debug(
                    "scheduler.usage.recorded",
                    task_id=task_id,
                    cost_usd=total_cost_usd,
                    turns=num_turns,
                    duration_ms=duration_ms,
                )
        except Exception as e:
            session.rollback()
            logger.error(
                "scheduler.usage.record_failed",
                task_id=task_id,
                error=str(e),
            )
        finally:
            session.close()

    def get_credit_status(self) -> dict:
        """Get current credit usage status with budget information"""
        self._init_current_window()

        # Get queue status
        status = self.task_queue.get_queue_status()

        # Get budget status
        budget_status = self.budget_manager.get_budget_status()

        return {
            "current_window": {
                "start_time": self.current_window.start_time.isoformat(),
                "end_time": self.current_window.end_time.isoformat(),
                "time_remaining_minutes": self.current_window.time_remaining_minutes(),
                "tasks_executed": self.current_window.tasks_executed,
            },
            "budget": budget_status,
            "queue": status,
            "max_parallel": self.max_parallel_tasks,
        }

    def get_execution_slots_available(self) -> int:
        """Get available execution slots"""
        in_progress = len(self.task_queue.get_in_progress_tasks())
        return max(0, self.max_parallel_tasks - in_progress)

    def should_backfill_with_random_thoughts(self) -> bool:
        """Determine if we should fill idle time with random thoughts"""
        slots = self.get_execution_slots_available()

        if slots == 0:
            return False

        pending_serious = self.task_queue.task_queue.filter(
            status=TaskStatus.PENDING,
            priority=TaskPriority.SERIOUS,
        )

        # If no serious tasks, fill with random thoughts
        return len(pending_serious) == 0

    def estimate_task_priority_score(self, task: Task) -> float:
        """Calculate priority score for task sorting"""
        score = 0.0

        # Priority multiplier
        if task.priority == TaskPriority.SERIOUS:
            score += 1000
        elif task.priority == TaskPriority.THOUGHT:
            score += 100
        elif task.priority == TaskPriority.GENERATED:
            score += 10

        # Age bonus (older tasks get higher score)
        age_minutes = (datetime.now(timezone.utc).replace(tzinfo=None) - task.created_at).total_seconds() / 60
        score += age_minutes * 0.1

        # Retry penalty (don't keep retrying failed tasks)
        score -= task.attempt_count * 50

        return score

    def get_scheduled_tasks_info(self) -> List[dict]:
        """Get info about all scheduled tasks"""
        queue_status = self.task_queue.get_queue_status()

        return [
            {
                "status": "pending",
                "count": queue_status["pending"],
            },
            {
                "status": "in_progress",
                "count": queue_status["in_progress"],
            },
            {
                "status": "completed",
                "count": queue_status["completed"],
            },
        ]

