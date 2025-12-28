"""Zhipu GLM Coding Plan usage monitoring"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


class ZhipuUsageChecker:
    """Check Zhipu GLM Coding Plan usage via API.

    Uses the same interface as ProPlanUsageChecker for drop-in replacement.
    Automatically detects region (CN/Global) from ZHIPU_BASE_URL.
    """

    def __init__(self):
        """Initialize usage checker with auto-detected endpoints."""
        from sleepless_agent.utils.zhipu_env import get_zhipu_config

        config = get_zhipu_config()
        self.auth_token = config.api_key

        if config.endpoints:
            # Use auto-detected endpoints
            self.usage_api_url = config.endpoints.usage_api
            self.region = config.endpoints.name
        else:
            # Fallback to default
            self.usage_api_url = "https://open.bigmodel.cn/api/monitor/usage/quota/limit"
            self.region = "zhipu_cn"

        # Cache
        self._cached_usage: Optional[float] = None
        self._cached_reset_time: Optional[datetime] = None
        self._cache_time: Optional[datetime] = None
        self.cache_duration_seconds = 60

        # For compatibility with ProPlanUsageChecker
        self.last_check_time: Optional[datetime] = None
        self.cached_usage: Optional[Tuple[float, Optional[datetime]]] = None

    def get_usage(self) -> Tuple[float, Optional[datetime]]:
        """Get usage percentage and reset time.

        Returns:
            Tuple of (usage_percent: float 0-100, reset_time: datetime or None)
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Check cache
        if self._cache_time:
            elapsed = (now - self._cache_time).total_seconds()
            if elapsed < self.cache_duration_seconds and self._cached_usage is not None:
                logger.debug("zhipu_usage.cache.hit", age_seconds=int(elapsed))
                return self._cached_usage, self._cached_reset_time

        if not self.auth_token:
            logger.error("zhipu_usage.no_auth_token")
            return 0.0, None

        try:
            headers = {
                "Authorization": self.auth_token,
                "Accept-Language": "en-US,en",
                "Content-Type": "application/json"
            }

            logger.debug("zhipu_usage.request", region=self.region, url=self.usage_api_url)

            response = requests.get(self.usage_api_url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            limits = data.get("data", {}).get("limits", [])

            # Find TOKENS_LIMIT
            usage_percent = 0.0
            for limit in limits:
                if limit.get("type") == "TOKENS_LIMIT":
                    usage_percent = float(limit.get("percentage", 0.0))
                    break

            reset_time = self._get_next_reset_time()

            # Update cache
            self._cached_usage = usage_percent
            self._cached_reset_time = reset_time
            self._cache_time = now

            # Also update ProPlanUsageChecker-compatible cache
            self.cached_usage = (usage_percent, reset_time)
            self.last_check_time = now

            if usage_percent >= 50.0:
                logger.info(
                    "zhipu_usage.snapshot",
                    usage_percent=usage_percent,
                )

            return usage_percent, reset_time

        except requests.exceptions.RequestException as e:
            logger.error("zhipu_usage.request_failed", error=str(e))
            # Return cached value if available
            if self._cached_usage is not None:
                return self._cached_usage, self._cached_reset_time
            return 0.0, None
        except Exception as e:
            logger.error("zhipu_usage.exception", error=str(e))
            return 0.0, None

    def _get_next_reset_time(self) -> datetime:
        """Calculate next 5-hour reset time.

        Zhipu resets quota every 5 hours starting from midnight.
        Reset hours: 0, 5, 10, 15, 20 (next day: 1, 6, 11, 16, 21)
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        current_hour = now.hour

        # Find next reset hour
        reset_hours = [0, 5, 10, 15, 20]
        next_reset_hour = None

        for h in reset_hours:
            if h > current_hour:
                next_reset_hour = h
                break

        if next_reset_hour is None:
            # Next reset is tomorrow at 0:00
            next_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
            next_reset += timedelta(days=1)
        else:
            next_reset = now.replace(
                hour=next_reset_hour,
                minute=0,
                second=0,
                microsecond=0
            )

        return next_reset

    def check_should_pause(
        self,
        threshold_percent: float = 85.0
    ) -> Tuple[bool, Optional[datetime]]:
        """Check if usage exceeds threshold.

        Args:
            threshold_percent: Pause if usage >= this percent (default 85%)

        Returns:
            Tuple of (should_pause: bool, reset_time: datetime or None)
        """
        try:
            usage_percent, reset_time = self.get_usage()
            should_pause = usage_percent >= threshold_percent

            if should_pause:
                logger.warning(
                    "zhipu_usage.threshold.exceeded",
                    usage_percent=usage_percent,
                    threshold_percent=threshold_percent,
                )

            return should_pause, reset_time

        except Exception as e:
            logger.error("zhipu_usage.threshold.error", error=str(e))
            return False, None
