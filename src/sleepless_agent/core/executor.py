"""Claude Code SDK executor for task processing"""

import asyncio
import re
import subprocess
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import shutil

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    CLINotFoundError,
    ProcessError,
    CLIJSONDecodeError,
    AssistantMessage,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
    TextBlock,
)
from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)

from sleepless_agent.utils.live_status import LiveStatusEntry, LiveStatusTracker


class ClaudeCodeExecutor:
    """Execute tasks using Claude Code CLI via Python Agent SDK"""

    def __init__(
        self,
        workspace_root: str = "./workspace",
        default_timeout: int = 3600,
        live_status_tracker: Optional[LiveStatusTracker] = None,
        default_model: str = "claude-sonnet-4-5-20250929",
    ):
        """Initialize Claude Code executor

        Args:
            workspace_root: Root directory for task workspaces
            default_timeout: Default timeout in seconds (not used by SDK directly)
            default_model: Default Claude model to use for all agents
        """
        self.workspace_root = Path(workspace_root)
        self.default_timeout = default_timeout
        self.default_model = default_model
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.live_status_tracker = live_status_tracker
        self._live_context: Dict[int, Dict[str, Optional[str]]] = {}

        # Create workspace subdirectories
        self.tasks_dir = self.workspace_root / "tasks"
        self.projects_dir = self.workspace_root / "projects"
        self.shared_dir = self.workspace_root / "shared"

        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        # Verify Claude Code is available
        self._verify_claude_cli()

        logger.info("executor.init", workspace=str(self.workspace_root))

    def _verify_claude_cli(self):
        """Verify Claude Code CLI is available"""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.debug("executor.cli.verified")
            else:
                raise CLINotFoundError()

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("executor.cli.not_found")
            raise RuntimeError(
                "Claude Code CLI not found. "
                "Please install with: npm install -g @anthropic-ai/claude-code"
            )
        except Exception as e:
            logger.warning("executor.cli.verify_failed", error=str(e))
            # Don't fail initialization - let it fail on actual execution if needed

    # ------------------------------------------------------------------ Live status helpers
    def _live_update(
        self,
        task_id: int,
        *,
        phase: str,
        prompt: Optional[str] = None,
        answer: Optional[str] = None,
        status: str = "running",
    ) -> None:
        """Publish live status updates if tracker is available."""
        if not self.live_status_tracker:
            return

        context = self._live_context.get(task_id, {})
        try:
            entry = LiveStatusEntry(
                task_id=task_id,
                description=context.get("description", ""),
                project_name=context.get("project_name"),
                phase=phase,
                prompt_preview=prompt or "",
                answer_preview=answer or "",
                status=status,
            )
            self.live_status_tracker.update(entry)
        except Exception as exc:  # pragma: no cover - debug aid
            logger.debug("executor.live_status.update_failed", task_id=task_id, error=str(exc))

    def _live_clear(self, task_id: int) -> None:
        """Remove live status tracking for the task."""
        if not self.live_status_tracker:
            return
        try:
            self.live_status_tracker.clear(task_id)
        except Exception as exc:  # pragma: no cover - debug aid
            logger.debug("executor.live_status.clear_failed", task_id=task_id, error=str(exc))

    def _live_phase_start(
        self,
        task_id: int,
        phase: str,
        prompt: str,
    ) -> None:
        """Helper for starting a new phase with live status update."""
        self._live_update(
            task_id,
            phase=phase,
            prompt=prompt[:500] if prompt else "",
            answer="",
            status="running",
        )

    def _live_phase_progress(
        self,
        task_id: int,
        phase: str,
        prompt: str,
        answer: str,
    ) -> None:
        """Helper for updating phase progress with live status."""
        self._live_update(
            task_id,
            phase=phase,
            prompt=prompt[:500] if prompt else "",
            answer=answer,
            status="running",
        )

    def _live_phase_complete(
        self,
        task_id: int,
        phase: str,
        prompt: str,
        answer: str,
    ) -> None:
        """Helper for completing a phase with live status update."""
        self._live_update(
            task_id,
            phase=phase,
            prompt=prompt[:500] if prompt else "",
            answer=answer,
            status="completed",
        )

    def _get_readme_template(self, template_type: str = "task") -> str:
        """Get README template content

        Args:
            template_type: 'task' or 'project'

        Returns:
            Template content string
        """
        # Templates were previously shipped as files; now we rely on lightweight defaults
        if template_type == "project":
            return (
                "# Project Workspace\n\n"
                "Project: {PROJECT_NAME}\n\n"
                "## Overview\n"
                "- Description: {TASK_DESCRIPTION}\n"
                "- Created: {CREATED_AT}\n"
            )

        return (
            "# Task Workspace\n\n"
            "Task #{TASK_ID}: {TASK_TITLE}\n\n"
            "## Summary\n"
            "- Priority: {PRIORITY_LABEL}\n"
            "- Project: {PROJECT_NAME}\n"
            "- Created: {CREATED_AT}\n\n"
            "## Description\n"
            "{TASK_DESCRIPTION}\n\n"
            "## Plan & Analysis\n"
            "(Generated by planner agent)\n\n"
            "## TODO List\n"
            "(Updated by worker agent)\n\n"
            "## Status: PENDING\n\n"
            "## Outstanding Items\n"
            "(Updated by evaluator)\n\n"
            "## Recommendations\n"
            "(Updated by evaluator)\n\n"
            "## Execution Summary\n"
        )

    def _ensure_readme_exists(self, workspace: Path, task_id: int, task_description: str,
                             project_id: Optional[str] = None, project_name: Optional[str] = None) -> Path:
        """Create README.md if it doesn't exist

        Args:
            workspace: Workspace path
            task_id: Task ID
            task_description: Task description
            project_id: Optional project ID
            project_name: Optional project name

        Returns:
            Path to README.md
        """
        readme_path = workspace / "README.md"

        if readme_path.exists():
            return readme_path

        try:
            template = self._get_readme_template("task")
            content = template.format(
                TASK_ID=task_id,
                TASK_TITLE=task_description[:50],
                TASK_DESCRIPTION=task_description,
                PRIORITY="serious" if project_id else "thought",
                PRIORITY_LABEL="SERIOUS" if project_id else "THOUGHT",
                PROJECT_NAME=project_name or "None",
                CREATED_AT=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            )

            readme_path.write_text(content)
            logger.debug("executor.readme.created", path=str(readme_path))
        except Exception as e:
            logger.warning("executor.readme.create_failed", error=str(e), path=str(readme_path))

        return readme_path

    def _update_readme_with_plan(self, workspace: Path, plan_content: str) -> None:
        """Update README.md with plan content from planner phase

        Args:
            workspace: Workspace path
            plan_content: Plan content from planner agent
        """
        readme_path = workspace / "README.md"

        if not readme_path.exists():
            logger.warning("executor.readme.missing", path=str(readme_path))
            return

        try:
            content = readme_path.read_text()

            # Replace the Plan & Analysis placeholder with actual plan
            content = re.sub(
                r'## Plan & Analysis\n\(Generated by planner agent\)',
                f'## Plan & Analysis\n{plan_content}',
                content
            )

            readme_path.write_text(content)
            logger.debug("executor.readme.plan_updated", path=str(readme_path))

        except Exception as e:
            logger.warning("executor.readme.plan_update_failed", error=str(e), path=str(readme_path))

    def _read_workspace_context(self, workspace: Path) -> str:
        """Read workspace context (README, existing files)

        Args:
            workspace: Workspace path

        Returns:
            Context string for planner
        """
        context_parts = []

        # Read README if exists
        readme_path = workspace / "README.md"
        if readme_path.exists():
            try:
                context_parts.append("## Project README\n" + readme_path.read_text())
            except Exception as e:
                logger.warning("executor.readme.read_failed", error=str(e), path=str(readme_path))

        # List main files/directories
        try:
            items = list(workspace.iterdir())
            file_list = [item.name for item in items if item.name not in {".git", ".gitignore", "__pycache__"}]
            if file_list:
                context_parts.append(f"\n## Workspace Contents\n- " + "\n- ".join(sorted(file_list)))
        except Exception as e:
            logger.warning("executor.workspace.list_failed", error=str(e), workspace=str(workspace))

        return "\n".join(context_parts) if context_parts else "Empty workspace"

    def _update_readme_task_history(self, workspace: Path, task_id: int,
                                    description: str, status: str,
                                    files_modified: int = 0,
                                    git_info: Optional[str] = None,
                                    execution_time: int = 0) -> None:
        """Update README.md with task completion history

        Args:
            workspace: Workspace path
            task_id: Task ID
            description: Task description
            status: 'completed' or 'failed'
            files_modified: Number of files modified
            git_info: Git commit/PR information
            execution_time: Execution time in seconds
        """
        readme_path = workspace / "README.md"

        try:
            if not readme_path.exists():
                return

            content = readme_path.read_text()

            # Add to execution summary
            status_icon = "✅" if status == "completed" else "❌"
            git_line = f"\n- Git: {git_info}" if git_info else ""

            update = f"\n\n### Execution {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')}\n"
            update += f"- Status: {status_icon} {status.upper()}\n"
            update += f"- Files Modified: {files_modified}\n"
            update += f"- Duration: {execution_time}s"
            update += git_line

            content = content.replace("## Execution Summary", f"## Execution Summary{update}\n\n## Execution Summary")
            readme_path.write_text(content)
            logger.debug("executor.readme.history_updated", path=str(readme_path))
        except Exception as e:
            logger.warning("executor.readme.history_failed", error=str(e), path=str(readme_path))

    def _extract_status_from_evaluation(self, evaluation_text: str) -> str:
        """Extract completion status from evaluator output

        Looks for keywords in evaluation:
        - "COMPLETE" or "completed successfully" → COMPLETE
        - "PARTIAL" or "partially complete" → PARTIAL
        - "INCOMPLETE" or "not completed" → INCOMPLETE
        - "FAILED" or "error" → FAILED

        Args:
            evaluation_text: Evaluator phase output

        Returns:
            Status string: COMPLETE | PARTIAL | INCOMPLETE | FAILED
        """
        text_lower = evaluation_text.lower()

        if "complete" in text_lower and "partial" not in text_lower:
            return "COMPLETE"
        elif "partial" in text_lower:
            return "PARTIAL"
        elif "incomplete" in text_lower or "not completed" in text_lower:
            return "INCOMPLETE"
        elif "failed" in text_lower or "error" in text_lower:
            return "FAILED"
        else:
            # Default to PARTIAL if unclear
            return "PARTIAL"

    def _extract_outstanding_items(self, evaluation_text: str) -> list:
        """Extract outstanding/incomplete items from evaluation

        Looks for sections like:
        - "Outstanding items:"
        - "Incomplete:"
        - "TODO:"

        Args:
            evaluation_text: Evaluator phase output

        Returns:
            List of outstanding items
        """
        items = []
        lines = evaluation_text.split('\n')
        in_outstanding = False

        for line in lines:
            if re.search(r'outstanding|incomplete|todo', line, re.IGNORECASE):
                in_outstanding = True
                continue

            if in_outstanding:
                # Check if line is a list item or checkbox
                if re.match(r'^\s*[-*❌✓]\s+', line) or re.match(r'^\s*\[\s*[x\s]\s*\]\s+', line):
                    items.append(line.strip())
                elif line.strip() == '':
                    in_outstanding = False

        return items

    def _extract_recommendations(self, evaluation_text: str) -> list:
        """Extract recommendations from evaluation

        Looks for "Recommendations:" section

        Args:
            evaluation_text: Evaluator phase output

        Returns:
            List of recommendations
        """
        items = []
        lines = evaluation_text.split('\n')
        in_recommendations = False

        for line in lines:
            if re.search(r'recommendation', line, re.IGNORECASE):
                in_recommendations = True
                continue

            if in_recommendations:
                # Check if line is a list item
                if re.match(r'^\s*[-*]\s+', line):
                    items.append(line.strip())
                elif line.strip() == '' or re.match(r'^##', line):
                    in_recommendations = False

        return items

    def _update_readme_with_evaluation(
        self,
        workspace: Path,
        status: str,
        outstanding_items: list,
        recommendations: list,
    ) -> None:
        """Update README.md with evaluation results

        Replaces sections:
        - ## Status: PENDING → ## Status: PARTIAL
        - ## Outstanding Items
        - ## Recommendations

        Args:
            workspace: Workspace path
            status: Completion status
            outstanding_items: List of outstanding items
            recommendations: List of recommendations
        """
        readme_path = workspace / "README.md"

        if not readme_path.exists():
            logger.warning("executor.readme.missing", path=str(readme_path))
            return

        try:
            content = readme_path.read_text()

            # Update status heading
            content = re.sub(
                r'## Status: \w+',
                f'## Status: {status}',
                content
            )

            # Update outstanding items
            if outstanding_items:
                items_text = '\n'.join(outstanding_items)
                content = re.sub(
                    r'## Outstanding Items\n(.*?)(?=##)',
                    f'## Outstanding Items\n{items_text}\n\n',
                    content,
                    flags=re.DOTALL
                )
            else:
                # Clear items if none
                content = re.sub(
                    r'## Outstanding Items\n(.*?)(?=##)',
                    f'## Outstanding Items\n(None)\n\n',
                    content,
                    flags=re.DOTALL
                )

            # Update recommendations
            if recommendations:
                rec_text = '\n'.join(recommendations)
                content = re.sub(
                    r'## Recommendations\n(.*?)(?=##)',
                    f'## Recommendations\n{rec_text}\n\n',
                    content,
                    flags=re.DOTALL
                )
            else:
                # Clear recommendations if none
                content = re.sub(
                    r'## Recommendations\n(.*?)(?=##)',
                    f'## Recommendations\n(None)\n\n',
                    content,
                    flags=re.DOTALL
                )

            readme_path.write_text(content)
            logger.debug("executor.readme.status_updated", status=status, path=str(readme_path))

        except Exception as e:
            logger.warning("executor.readme.evaluation_failed", error=str(e), path=str(readme_path))

    def _generate_task_name_slug(self, description: str) -> str:
        """Generate a slug from task description

        Args:
            description: Task description

        Returns:
            Slugified name from first few words (max 30 chars)
        """
        # Get first 3-4 words and join them
        words = description.split()[:4]
        slug = '-'.join(words)

        # Remove non-alphanumeric chars except hyphens
        slug = re.sub(r'[^a-z0-9-]', '', slug.lower())

        # Remove multiple hyphens
        slug = re.sub(r'-+', '-', slug)

        # Remove leading/trailing hyphens
        slug = slug.strip('-')

        # Truncate to 30 chars
        slug = slug[:30]

        # If slug is empty, use 'task'
        return slug if slug else 'task'

    def _get_allowed_directories(
        self,
        workspace: Path,
        workspace_task_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> list[str]:
        """Compute allowed directories for task execution.

        Workspace isolation is always enabled to restrict file access to:
        - The task's own workspace (set via cwd)
        - The shared directory (always allowed for cross-task resources)
        - Project workspace (for project-based tasks only)

        Args:
            workspace: Task workspace path
            workspace_task_type: Task type ("new" or "refine")
            project_id: Optional project ID

        Returns:
            List of absolute paths to additional allowed directories
        """
        allowed = []

        # Always allow shared directory (hardcoded for cross-task resources)
        shared_dir = self.shared_dir
        if shared_dir.exists():
            allowed.append(str(shared_dir.resolve()))

        # For project tasks, allow access to the project workspace
        # (different from task workspace which is in tasks/)
        if project_id:
            project_workspace = self.projects_dir / project_id
            if project_workspace.exists() and project_workspace != workspace:
                allowed.append(str(project_workspace.resolve()))

        logger.debug(
            "executor.access_control",
            workspace=str(workspace),
            add_dirs=allowed,
            task_type=workspace_task_type,
        )

        return allowed

    async def _execute_planner_phase(
        self,
        task_id: int,
        workspace: Path,
        description: str,
        context: str,
        config_max_turns: int = 10,
        workspace_task_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> tuple[str, dict]:
        """Execute planner agent phase

        Args:
            workspace: Workspace path
            description: Task description
            context: Workspace context (README, files)
            config_max_turns: Maximum turns for this phase
            workspace_task_type: Task type ("new" or "refine")
            project_id: Optional project ID for access control

        Returns:
            Tuple of (plan_text, usage_metrics)
        """
        # Add workspace task type context
        task_type_note = ""
        if workspace_task_type == "refine":
            task_type_note = """
## 🔧 REFINE TASK
This workspace contains a copy of the sleepless-agent source code. Your goal is to IMPROVE the existing codebase:
- Analyze and understand the current implementation
- Refactor, optimize, or enhance existing code
- Fix bugs or issues
- Improve code quality, tests, or documentation
- Add missing features to existing modules

The changes you make should enhance the existing codebase and can be merged back to the main project.
"""
        elif workspace_task_type == "new":
            task_type_note = """
## 🆕 NEW TASK
This is a fresh workspace. Your goal is to BUILD new functionality from scratch:
- Design and implement new features
- Create new modules or tools
- Build standalone projects or prototypes
- Experiment with new ideas
"""

        planner_prompt = f"""You are a planning expert. Analyze the task and workspace context, then create a structured plan.
{task_type_note}
## Task
{description}

## Workspace Context
{context}

## Your Task
1. Analyze the task requirements and workspace
2. Identify what needs to be done
3. Create a detailed TODO list with specific, actionable items
4. Note any dependencies between tasks
5. Estimate effort level for each TODO item

Output should be:
- Executive summary (2-3 sentences)
- Analysis of the task
- Structured TODO list (numbered, with clear descriptions)
- Notes on approach and strategy
- Any assumptions or potential blockers
"""

        prompt_preview = " ".join(planner_prompt.split())

        usage_metrics = {
            "planner_cost_usd": None,
            "planner_duration_ms": None,
            "planner_turns": None,
        }

        try:
            output_parts = []
            start_time = time.time()

            self._live_update(
                task_id,
                phase="planner",
                prompt=prompt_preview,
                answer="",
                status="running",
            )

            # Get allowed directories for workspace isolation
            allowed_dirs = self._get_allowed_directories(
                workspace=workspace,
                workspace_task_type=workspace_task_type,
                project_id=project_id,
            )

            # Get MCP servers for planner phase (search/reader, no vision)
            from sleepless_agent.utils.config import get_config
            from sleepless_agent.utils.mcp_config import get_mcp_servers_for_phase
            config = get_config()
            mcp_servers = get_mcp_servers_for_phase(config, phase="planner")

            options = ClaudeAgentOptions(
                cwd=str(workspace),
                add_dirs=allowed_dirs,
                allowed_tools=["Read", "Glob", "Grep"],
                permission_mode="acceptEdits",
                max_turns=config_max_turns,
                model=self.default_model,
                mcp_servers=mcp_servers if mcp_servers else None,
            )

            async for message in query(prompt=planner_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text.strip()
                            if text:
                                output_parts.append(text)
                                self._live_update(
                                    task_id,
                                    phase="planner",
                                    prompt=prompt_preview,
                                    answer=text,
                                    status="running",
                                )

                elif isinstance(message, ResultMessage):
                    usage_metrics["planner_cost_usd"] = message.total_cost_usd
                    usage_metrics["planner_duration_ms"] = message.duration_ms
                    usage_metrics["planner_turns"] = message.num_turns

                    logger.debug(
                        "executor.planner.turn_metrics",
                        duration_ms=message.duration_ms,
                        turns=message.num_turns,
                        cost_usd=message.total_cost_usd,
                    )

            plan_text = "\n".join(output_parts)
            execution_time = int(time.time() - start_time)

            self._live_update(
                task_id,
                phase="planner",
                prompt=prompt_preview,
                answer=plan_text,
                status="completed",
            )

            return plan_text, usage_metrics

        except Exception as e:
            logger.error("executor.planner.failed", error=str(e))
            raise

    async def _execute_worker_phase(
        self,
        task_id: int,
        workspace: Path,
        description: str,
        plan_text: str,
        config_max_turns: int = 30,
        workspace_task_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> tuple[str, set, list, int, dict]:
        """Execute worker agent phase

        Args:
            workspace: Workspace path
            description: Task description
            plan_text: Plan from planner phase
            config_max_turns: Maximum turns for this phase
            workspace_task_type: Task type ("new" or "refine")
            project_id: Optional project ID for access control

        Returns:
            Tuple of (output_text, files_modified, commands_executed, exit_code, usage_metrics)
        """
        worker_prompt = f"""You are an expert developer/engineer. Execute the plan below to complete the task.

## Task
{description}

## Plan to Execute
{plan_text}

## Instructions
1. Execute the TODO items from the plan
2. Use TodoWrite to track progress on each item
3. Make changes using available tools (Read, Write, Edit, Bash)
4. Test your changes as needed
5. Provide a summary of what you completed

Please work through the plan systematically and update TodoWrite as you complete each item.
"""

        prompt_preview = " ".join(worker_prompt.split())

        files_modified = set()
        commands_executed = []
        output_parts = []
        success = True
        usage_metrics = {
            "worker_cost_usd": None,
            "worker_duration_ms": None,
            "worker_turns": None,
        }
        tool_usage_counts: "OrderedDict[str, int]" = OrderedDict()

        try:
            files_before = self._get_workspace_files(workspace)
            start_time = time.time()

            self._live_update(
                task_id,
                phase="worker",
                prompt=prompt_preview,
                answer="",
                status="running",
            )

            # Get allowed directories for workspace isolation
            allowed_dirs = self._get_allowed_directories(
                workspace=workspace,
                workspace_task_type=workspace_task_type,
                project_id=project_id,
            )

            # Get MCP servers for worker phase (all capabilities)
            from sleepless_agent.utils.config import get_config
            from sleepless_agent.utils.mcp_config import get_mcp_servers_for_phase
            config = get_config()
            mcp_servers = get_mcp_servers_for_phase(config, phase="worker")

            options = ClaudeAgentOptions(
                cwd=str(workspace),
                add_dirs=allowed_dirs,
                allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite"],
                permission_mode="acceptEdits",
                max_turns=config_max_turns,
                model=self.default_model,
                mcp_servers=mcp_servers if mcp_servers else None,
            )

            async for message in query(prompt=worker_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text.strip()
                            if text:
                                output_parts.append(text)
                                self._live_update(
                                    task_id,
                                    phase="worker",
                                    prompt=prompt_preview,
                                    answer=text,
                                    status="running",
                                )
                        elif isinstance(block, ToolUseBlock):
                            tool_name = block.name
                            tool_usage_counts.setdefault(tool_name, 0)
                            tool_usage_counts[tool_name] += 1

                            if tool_name in ["Write", "Edit"]:
                                file_path = block.input.get("file_path", "")
                                if file_path:
                                    files_modified.add(file_path)

                            elif tool_name == "Bash":
                                command = block.input.get("command", "")
                                if command:
                                    commands_executed.append(command)
                                self._live_update(
                                    task_id,
                                    phase="worker",
                                    prompt=prompt_preview,
                                    answer=f"[Bash] {command}",
                                    status="running",
                                )

                elif isinstance(message, ResultMessage):
                    success = not message.is_error
                    if message.result:
                        output_parts.append(f"\n[Result: {message.result}]")
                        self._live_update(
                            task_id,
                            phase="worker",
                            prompt=prompt_preview,
                            answer=message.result,
                            status="running",
                        )

                    usage_metrics["worker_cost_usd"] = message.total_cost_usd
                    usage_metrics["worker_duration_ms"] = message.duration_ms
                    usage_metrics["worker_turns"] = message.num_turns

                    logger.debug(
                        "executor.worker.turn_metrics",
                        duration_ms=message.duration_ms,
                        turns=message.num_turns,
                        cost_usd=message.total_cost_usd,
                    )

            output_text = "\n".join(output_parts)
            files_after = self._get_workspace_files(workspace)
            new_files = files_after - files_before
            all_modified_files = files_modified.union(new_files)

            if tool_usage_counts:
                summary = ", ".join(
                    f"{name} x{count}" for name, count in tool_usage_counts.items()
                )
                logger.debug("executor.worker.tools_summary", summary=summary)

            exit_code = 0 if success else 1

            # Update final worker status
            self._live_update(
                task_id,
                phase="worker",
                prompt=prompt_preview,
                answer=output_text,
                status="completed" if exit_code == 0 else "error",
            )

            return output_text, all_modified_files, commands_executed, exit_code, usage_metrics

        except Exception as e:
            logger.error("executor.worker.failed", error=str(e))
            raise

    async def _execute_evaluator_phase(
        self,
        task_id: int,
        workspace: Path,
        description: str,
        plan_text: str,
        worker_output: str,
        files_modified: set,
        commands_executed: list,
        config_max_turns: int = 10,
        workspace_task_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> tuple[str, str, list, list, dict]:
        """Execute evaluator agent phase

        Args:
            workspace: Workspace path
            description: Task description
            plan_text: Plan from planner phase
            worker_output: Output from worker phase
            files_modified: Files modified by worker
            commands_executed: Commands executed by worker
            config_max_turns: Maximum turns for this phase
            workspace_task_type: Task type ("new" or "refine")
            project_id: Optional project ID for access control

        Returns:
            Tuple of (evaluation_text, status, outstanding_items, recommendations, usage_metrics)
        """
        evaluator_prompt = f"""You are a quality assurance expert. Evaluate whether the task was completed successfully.

## Task
{description}

## Original Plan
{plan_text}

## Worker Output
{worker_output}

## Changes Made
- Files Modified: {len(files_modified)}
- Commands Executed: {len(commands_executed)}

## Your Task
1. Review the worker output against the original plan
2. Verify each TODO item was addressed
3. Check if the task objectives were met
4. Identify any incomplete items or issues
5. Provide a comprehensive evaluation summary

Output should include:
- Completion status (COMPLETE / INCOMPLETE / PARTIAL)
- Items successfully completed
- Any outstanding items
- Quality assessment
- Recommendations (if any)
"""

        usage_metrics = {
            "evaluator_cost_usd": None,
            "evaluator_duration_ms": None,
            "evaluator_turns": None,
        }

        try:
            output_parts = []
            start_time = time.time()

            prompt_preview = " ".join(evaluator_prompt.split())

            self._live_update(
                task_id,
                phase="evaluator",
                prompt=prompt_preview,
                answer="",
                status="running",
            )

            # Get allowed directories for workspace isolation
            allowed_dirs = self._get_allowed_directories(
                workspace=workspace,
                workspace_task_type=workspace_task_type,
                project_id=project_id,
            )

            options = ClaudeAgentOptions(
                cwd=str(workspace),
                add_dirs=allowed_dirs,
                allowed_tools=["Read", "Glob"],
                permission_mode="acceptEdits",
                max_turns=config_max_turns,
                model=self.default_model,
            )

            async for message in query(prompt=evaluator_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text.strip()
                            if text:
                                output_parts.append(text)
                                self._live_update(
                                    task_id,
                                    phase="evaluator",
                                    prompt=prompt_preview,
                                    answer=text,
                                    status="running",
                                )

                elif isinstance(message, ResultMessage):
                    usage_metrics["evaluator_cost_usd"] = message.total_cost_usd
                    usage_metrics["evaluator_duration_ms"] = message.duration_ms
                    usage_metrics["evaluator_turns"] = message.num_turns

                    logger.debug(
                        "executor.evaluator.turn_metrics",
                        duration_ms=message.duration_ms,
                        turns=message.num_turns,
                        cost_usd=message.total_cost_usd,
                    )

            evaluation_text = "\n".join(output_parts)
            execution_time = int(time.time() - start_time)

            self._live_update(
                task_id,
                phase="evaluator",
                prompt=prompt_preview,
                answer=evaluation_text,
                status="completed",
            )

            # Extract evaluation status, outstanding items, and recommendations
            status = self._extract_status_from_evaluation(evaluation_text)
            outstanding_items = self._extract_outstanding_items(evaluation_text)
            recommendations = self._extract_recommendations(evaluation_text)

            logger.debug("executor.evaluator.status", status=status)
            if outstanding_items:
                logger.debug("executor.evaluator.outstanding", count=len(outstanding_items))
            if recommendations:
                logger.debug("executor.evaluator.recommendations", count=len(recommendations))

            # Update README with evaluation results
            # Note: We'll do this right before the usage check so we have access to task_id, project_id, etc.
            # This is done in the execute_task orchestration method, not here

            # Check Pro plan usage after evaluation (mandatory)
            try:
                from sleepless_agent.utils.config import get_config
                from sleepless_agent.utils.zhipu_env import get_usage_checker
                from sleepless_agent.utils.exceptions import PauseException
                from sleepless_agent.scheduling.time_utils import is_nighttime

                config = get_config()
                logger.debug("executor.usage.checking")
                checker = get_usage_checker()

                # Use time-based threshold
                threshold = config.claude_code.threshold_night if is_nighttime(night_start_hour=config.claude_code.night_start_hour, night_end_hour=config.claude_code.night_end_hour) else config.claude_code.threshold_day
                should_pause, reset_time = checker.check_should_pause(
                    threshold_percent=threshold
                )

                if should_pause:
                    usage_percent, _ = checker.get_usage()
                    logger.critical(
                        "executor.usage.threshold_reached",
                        usage_percent=usage_percent,
                        threshold_percent=threshold,
                    )
                    raise PauseException(
                        message=f"Pro plan usage limit reached at {usage_percent:.1f}%",
                        reset_time=reset_time,
                        usage_percent=usage_percent,
                    )
                else:
                    logger.debug("executor.usage.ready")

            except PauseException:
                # Re-raise PauseException to be caught by caller
                raise
            except ImportError:
                logger.debug("executor.usage.monitoring_unavailable")
            except Exception as e:
                logger.warning("executor.usage.check_error", error=str(e))
                # Don't fail the task due to usage check errors

            return evaluation_text, status, outstanding_items, recommendations, usage_metrics

        except Exception as e:
            logger.error("executor.evaluator.failed", error=str(e))
            raise

    def _copy_source_to_workspace(
        self,
        workspace: Path,
        task_id: int,
    ) -> None:
        """Copy sleepless-agent source code to workspace for REFINE tasks

        Args:
            workspace: Target workspace directory
            task_id: Task ID (for logging)
        """
        # Source paths to copy
        SOURCE_PATHS = ["src/", "pyproject.toml", "README.md", ".gitignore"]

        # Patterns to exclude during copy
        EXCLUDE_PATTERNS = (
            "__pycache__",
            "*.pyc",
            ".git",
            ".venv",
            "venv",
            "*.egg-info",
            "dist",
            "build",
            "workspace",
        )

        try:
            # Get project root (parent of workspace_root)
            project_root = self.workspace_root.parent

            logger.debug(
                "executor.copy_source.start",
                task_id=task_id,
                project_root=str(project_root),
                workspace=str(workspace)
            )

            copied_count = 0

            # Copy each source path
            for source_path_str in SOURCE_PATHS:
                source = project_root / source_path_str
                dest = workspace / source_path_str

                if not source.exists():
                    logger.warning(
                        "executor.copy_source.not_found",
                        source=str(source),
                        task_id=task_id
                    )
                    continue

                if source.is_dir():
                    # Copy directory, excluding patterns
                    shutil.copytree(
                        source,
                        dest,
                        ignore=shutil.ignore_patterns(*EXCLUDE_PATTERNS),
                        dirs_exist_ok=True
                    )
                    copied_count += 1
                else:
                    # Copy file
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    copied_count += 1

            logger.info(
                "executor.copy_source.complete",
                task_id=task_id,
                copied_items=copied_count,
                workspace=str(workspace)
            )

        except Exception as e:
            logger.error(
                "executor.copy_source.failed",
                task_id=task_id,
                error=str(e)
            )
            # Don't fail task creation due to copy errors - workspace is still usable

    def create_task_workspace(
        self,
        task_id: int,
        task_description: str = "",
        init_git: bool = False,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        task_type: Optional[str] = None,
        task_context: Optional[dict] = None,
    ) -> Path:
        """Create workspace for task - project-based if project_id provided, else task-specific

        Args:
            task_id: Task ID
            task_description: Task description (for generating slug)
            init_git: Whether to initialize git repo in workspace
            project_id: Optional project ID for shared workspace
            project_name: Optional project name (for logging)
            task_type: Task type ("new" or "refine") - if "refine", copy source code
            task_context: Optional task context dict (may contain refines_task_id)

        Returns:
            Path to created workspace
        """
        # Check if this is a REFINE task with a specific target task to refine
        refines_task_id = None
        if task_context:
            refines_task_id = task_context.get('refines_task_id')

        # If this task refines another task, reuse that task's workspace
        if refines_task_id is not None:
            target_workspace = self._find_task_workspace(refines_task_id)
            if target_workspace and target_workspace.exists():
                logger.info(
                    "workspace.reuse",
                    task_id=task_id,
                    refines=refines_task_id,
                    workspace=str(target_workspace)
                )
                return target_workspace
            else:
                logger.warning(
                    "workspace.refine_target_missing",
                    task_id=task_id,
                    refines=refines_task_id
                )
                # Fall through to create new workspace

        # Use project-based workspace if project_id provided, else task-specific
        if project_id:
            workspace = self.projects_dir / project_id
            workspace_type = f"project workspace '{project_name or project_id}'"
        else:
            # Generate slug from task description
            task_slug = self._generate_task_name_slug(task_description)
            workspace = self.tasks_dir / f"{task_id}_{task_slug}"
            workspace_type = f"task workspace {task_id}"

        workspace.mkdir(parents=True, exist_ok=True)

        # For REFINE tasks without specific target, copy source code to workspace
        if task_type == "refine" and refines_task_id is None:
            logger.info(
                "workspace.refine_task",
                task_id=task_id,
                workspace=str(workspace)
            )
            self._copy_source_to_workspace(workspace, task_id)

        # Move workspace setup to DEBUG - internal detail
        logger.debug(
            "workspace.ready",
            task_id=task_id,
            workspace=str(workspace),
            workspace_type=workspace_type,
            task_type=task_type or "new"
        )
        return workspace

    async def execute_task(
        self,
        task_id: int,
        description: str,
        task_type: str = "general",
        priority: str = "thought",
        timeout: Optional[int] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        workspace_task_type: Optional[str] = None,
        task_context: Optional[dict] = None,
    ) -> Tuple[str, List[str], List[str], int, Dict, Optional[str]]:
        """Execute task with Claude Code SDK

        Args:
            task_id: Task ID
            description: Task description/prompt
            task_type: Type of task (code, research, brainstorm, etc.)
            priority: Task priority (thought or serious)
            timeout: Timeout in seconds (not directly supported by SDK)
            project_id: Optional project ID for shared workspace
            project_name: Optional project name (for logging)
            workspace_task_type: Workspace task type ("new" or "refine") - for workspace initialization
            task_context: Optional task context dict (may contain refines_task_id)

        Returns:
            Tuple of (output_text, files_modified, commands_executed, exit_code, usage_metrics, eval_status)
            eval_status can be: "COMPLETE", "PARTIAL", "INCOMPLETE", "FAILED", or None if evaluator disabled
        """
        timeout = timeout or self.default_timeout

        self._live_context[task_id] = {
            "description": description,
            "project_name": project_name,
        }
        kickoff_label = f"{task_type.replace('_', ' ').title()} task kickoff"
        self._live_update(
            task_id,
            phase="initializing",
            prompt=kickoff_label,
            answer="",
            status="running",
        )

        try:
            # Create workspace (project-based if project_id provided)
            init_git = (priority == "serious")
            workspace = self.create_task_workspace(
                task_id=task_id,
                task_description=description,
                init_git=init_git,
                project_id=project_id,
                project_name=project_name,
                task_type=workspace_task_type,
                task_context=task_context,
            )

            # Multi-agent workflow orchestration
            # Build context dict with only non-None values to reduce log noise
            context = {
                "task_id": task_id,
                "priority": priority,
            }
            if project_id:
                context["project_id"] = project_id
            if project_name:
                context["project_name"] = project_name
            task_log = logger.bind(**context)
            # Move workflow start to DEBUG - reduces log noise
            workspace_str = str(workspace)
            task_log.debug(
                "task.workflow.start",
                workspace=workspace_str,
                task_type=task_type,
            )
            start_time = time.time()

            # Get configuration
            from sleepless_agent.utils.config import get_config
            config = get_config()
            multi_agent_config = config.multi_agent_workflow

            # Initialize combined metrics
            combined_metrics = {
                "total_cost_usd": 0.0,
                "duration_ms": 0,
                "duration_api_ms": 0,
                "num_turns": 0,
                "planner_cost_usd": None,
                "planner_duration_ms": None,
                "planner_turns": None,
                "worker_cost_usd": None,
                "worker_duration_ms": None,
                "worker_turns": None,
                "evaluator_cost_usd": None,
                "evaluator_duration_ms": None,
                "evaluator_turns": None,
            }

            all_output_parts = []
            all_files_modified = set()
            all_commands_executed = []
            final_exit_code = 0
            evaluation_summary = ""

            # Ensure README exists (mandatory)
            self._ensure_readme_exists(workspace, task_id, description, project_id, project_name)

            # Read workspace context for planner
            workspace_context = self._read_workspace_context(workspace)

            # Phase 1: Planner
            if multi_agent_config.planner.enabled:
                phase_log = task_log.bind(phase="planner")
                # Move phase start to DEBUG - internal workflow detail
                phase_log.debug(
                    "task.phase.start",
                    max_turns=multi_agent_config.planner.max_turns,
                )
                try:
                    plan_text, planner_metrics = await self._execute_planner_phase(
                        task_id=task_id,
                        workspace=workspace,
                        description=description,
                        context=workspace_context,
                        config_max_turns=multi_agent_config.planner.max_turns,
                        workspace_task_type=workspace_task_type,
                        project_id=project_id,
                    )
                    all_output_parts.append(f"## Planner Output\n{plan_text}")

                    # Update combined metrics
                    combined_metrics["planner_cost_usd"] = planner_metrics.get("planner_cost_usd")
                    combined_metrics["planner_duration_ms"] = planner_metrics.get("planner_duration_ms")
                    combined_metrics["planner_turns"] = planner_metrics.get("planner_turns")
                    if planner_metrics.get("planner_cost_usd"):
                        combined_metrics["total_cost_usd"] += planner_metrics["planner_cost_usd"]
                    if planner_metrics.get("planner_duration_ms"):
                        combined_metrics["duration_api_ms"] += planner_metrics["planner_duration_ms"]
                    if planner_metrics.get("planner_turns"):
                        combined_metrics["num_turns"] += planner_metrics["planner_turns"]

                    # Move phase done to DEBUG - verbose internal metrics
                    phase_log.debug(
                        "task.phase.done",
                        turns=planner_metrics.get("planner_turns"),
                        cost_usd=planner_metrics.get("planner_cost_usd"),
                        duration_ms=planner_metrics.get("planner_duration_ms"),
                    )

                    # Update README with plan
                    self._update_readme_with_plan(workspace, plan_text)

                except Exception as e:
                    phase_log.error("task.phase.failed", error=str(e))
                    plan_text = f"[Planner phase failed: {str(e)}]"
                    final_exit_code = 1
            else:
                plan_text = "[Planner phase disabled]"

            # Phase 2: Worker
            if multi_agent_config.worker.enabled and final_exit_code == 0:
                phase_log = task_log.bind(phase="worker")
                # Move phase start to DEBUG - internal workflow detail
                phase_log.debug(
                    "task.phase.start",
                    max_turns=multi_agent_config.worker.max_turns,
                )
                try:
                    worker_output, files_modified, commands_executed, exit_code, worker_metrics = await self._execute_worker_phase(
                        task_id=task_id,
                        workspace=workspace,
                        description=description,
                        plan_text=plan_text,
                        config_max_turns=multi_agent_config.worker.max_turns,
                        workspace_task_type=workspace_task_type,
                        project_id=project_id,
                    )
                    all_output_parts.append(f"## Worker Output\n{worker_output}")
                    all_files_modified = files_modified
                    all_commands_executed = commands_executed
                    final_exit_code = exit_code

                    # Update combined metrics
                    combined_metrics["worker_cost_usd"] = worker_metrics.get("worker_cost_usd")
                    combined_metrics["worker_duration_ms"] = worker_metrics.get("worker_duration_ms")
                    combined_metrics["worker_turns"] = worker_metrics.get("worker_turns")
                    if worker_metrics.get("worker_cost_usd"):
                        combined_metrics["total_cost_usd"] += worker_metrics["worker_cost_usd"]
                    if worker_metrics.get("worker_duration_ms"):
                        combined_metrics["duration_api_ms"] += worker_metrics["worker_duration_ms"]
                    if worker_metrics.get("worker_turns"):
                        combined_metrics["num_turns"] += worker_metrics["worker_turns"]

                    # Move phase done to DEBUG - verbose internal metrics
                    phase_log.debug(
                        "task.phase.done",
                        turns=worker_metrics.get("worker_turns"),
                        cost_usd=worker_metrics.get("worker_cost_usd"),
                        duration_ms=worker_metrics.get("worker_duration_ms"),
                        exit_code=exit_code,
                        files=len(all_files_modified),
                        commands=len(all_commands_executed),
                    )

                except Exception as e:
                    phase_log.error("task.phase.failed", error=str(e))
                    all_output_parts.append(f"## Worker Output\n[Worker phase failed: {str(e)}]")
                    final_exit_code = 1
            else:
                all_output_parts.append("## Worker Output\n[Worker phase disabled or skipped due to planner failure]")

            # Phase 3: Evaluator
            eval_status = None
            eval_outstanding = []
            eval_recommendations = []

            if multi_agent_config.evaluator.enabled:
                phase_log = task_log.bind(phase="evaluator")
                # Move phase start to DEBUG - internal workflow detail
                phase_log.debug(
                    "task.phase.start",
                    max_turns=multi_agent_config.evaluator.max_turns,
                )
                try:
                    evaluation_summary, eval_status, eval_outstanding, eval_recommendations, evaluator_metrics = await self._execute_evaluator_phase(
                        task_id=task_id,
                        workspace=workspace,
                        description=description,
                        plan_text=plan_text,
                        worker_output="\n".join(all_output_parts),
                        files_modified=all_files_modified,
                        commands_executed=all_commands_executed,
                        config_max_turns=multi_agent_config.evaluator.max_turns,
                        workspace_task_type=workspace_task_type,
                        project_id=project_id,
                    )
                    all_output_parts.append(f"## Evaluator Output\n{evaluation_summary}")

                    # Update README with evaluation results (mandatory)
                    if eval_status:
                        self._update_readme_with_evaluation(
                            workspace=workspace,
                            status=eval_status,
                            outstanding_items=eval_outstanding,
                            recommendations=eval_recommendations,
                        )

                    # Update combined metrics
                    combined_metrics["evaluator_cost_usd"] = evaluator_metrics.get("evaluator_cost_usd")
                    combined_metrics["evaluator_duration_ms"] = evaluator_metrics.get("evaluator_duration_ms")
                    combined_metrics["evaluator_turns"] = evaluator_metrics.get("evaluator_turns")
                    if evaluator_metrics.get("evaluator_cost_usd"):
                        combined_metrics["total_cost_usd"] += evaluator_metrics["evaluator_cost_usd"]
                    if evaluator_metrics.get("evaluator_duration_ms"):
                        combined_metrics["duration_api_ms"] += evaluator_metrics["evaluator_duration_ms"]
                    if evaluator_metrics.get("evaluator_turns"):
                        combined_metrics["num_turns"] += evaluator_metrics["evaluator_turns"]

                    # Move phase done to DEBUG - verbose internal metrics
                    phase_log.debug(
                        "task.phase.done",
                        turns=evaluator_metrics.get("evaluator_turns"),
                        cost_usd=evaluator_metrics.get("evaluator_cost_usd"),
                        duration_ms=evaluator_metrics.get("evaluator_duration_ms"),
                        status=eval_status,
                    )

                except Exception as e:
                    phase_log.error("task.phase.failed", error=str(e))
                    all_output_parts.append(f"## Evaluator Output\n[Evaluator phase failed: {str(e)}]")
                    eval_status = "failed"
                    final_exit_code = 1
            else:
                all_output_parts.append("## Evaluator Output\n[Evaluator phase disabled]")

            # Finalize execution
            execution_time = int(time.time() - start_time)
            combined_metrics["duration_ms"] = execution_time * 1000

            # Combine output
            output_text = "\n".join(all_output_parts)

            # Update README with execution history (mandatory)
            status = "completed" if final_exit_code == 0 else "failed"
            git_info = None  # Could be set by caller if needed
            self._update_readme_task_history(
                workspace,
                task_id,
                description,
                status,
                files_modified=len(all_files_modified),
                git_info=git_info,
                execution_time=execution_time,
            )

            # Convert sets to sorted lists for return
            all_modified_files = sorted(list(all_files_modified))

            last_section = all_output_parts[-1] if all_output_parts else ""
            self._live_update(
                task_id,
                phase="completed",
                prompt=description,
                answer=last_section,
                status="completed" if final_exit_code == 0 else "error",
            )

            # Simplified workflow complete log - only essential info
            task_log.info(
                "task.workflow.complete",
                duration_ms=combined_metrics.get("duration_ms"),
                files=len(all_modified_files),
                commands=len(all_commands_executed),
            )

            return output_text, all_modified_files, all_commands_executed, final_exit_code, combined_metrics, eval_status

        except CLINotFoundError:
            self._live_update(
                task_id,
                phase="error",
                prompt=description,
                answer="Claude Code CLI not found",
                status="error",
            )
            logger.error("executor.cli.not_found")
            raise RuntimeError(
                "Claude Code CLI not found. "
                "Please install with: npm install -g @anthropic-ai/claude-code"
            )
        except ProcessError as e:
            self._live_update(
                task_id,
                phase="error",
                prompt=description,
                answer=str(e),
                status="error",
            )
            logger.error("executor.cli.process_error", error=str(e))
            raise RuntimeError(f"Claude Code process failed: {e}")
        except CLIJSONDecodeError as e:
            self._live_update(
                task_id,
                phase="error",
                prompt=description,
                answer=str(e),
                status="error",
            )
            logger.error("executor.cli.parse_failed", error=str(e))
            raise RuntimeError(f"Failed to parse Claude Code output: {e}")
        except asyncio.CancelledError:
            self._live_update(
                task_id,
                phase="error",
                prompt=description,
                answer="Execution cancelled (timeout or shutdown)",
                status="error",
            )
            logger.warning("executor.task.cancelled", task_id=task_id)
            raise
        except Exception as e:
            self._live_update(
                task_id,
                phase="error",
                prompt=description,
                answer=str(e),
                status="error",
            )
            logger.error("executor.task.failed", task_id=task_id, error=str(e))
            raise
        finally:
            self._live_context.pop(task_id, None)


    def _get_workspace_files(self, workspace: Path) -> set:
        """Get set of all files in workspace (excluding metadata and .git)

        Args:
            workspace: Workspace path

        Returns:
            Set of relative file paths
        """
        files = set()
        exclude_patterns = {
            ".git",
            ".gitignore",
            "__pycache__",
            ".DS_Store",
            "node_modules",
        }

        try:
            for path in workspace.rglob("*"):
                if path.is_file():
                    # Check if any parent or the file itself should be excluded
                    relative_path = path.relative_to(workspace)
                    parts = set(relative_path.parts)

                    # Check for excluded patterns
                    should_exclude = False
                    for pattern in exclude_patterns:
                        if pattern in parts or relative_path.name == pattern:
                            should_exclude = True
                            break
                        # Check for .git directory in path
                        if any(part.startswith(".git") for part in parts):
                            should_exclude = True
                            break

                    if not should_exclude:
                        files.add(str(relative_path))
        except Exception as e:
            logger.warning("executor.workspace.scan_failed", error=str(e), workspace=str(workspace))

        return files

    def list_workspace_files(self, workspace: Path) -> set:
        """Public helper to list workspace files (excludes caches and metadata)."""
        return self._get_workspace_files(workspace)

    def cleanup_workspace_caches(self, workspace: Path):
        """Remove Python cache directories from workspace."""
        try:
            for cache_dir in workspace.rglob("__pycache__"):
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as exc:
            logger.debug("executor.workspace.cache_cleanup_failed", workspace=str(workspace), error=str(exc))

    def _find_task_workspace(self, task_id: int) -> Optional[Path]:
        """Find task workspace by ID (searches tasks/ directory)

        Args:
            task_id: Task ID

        Returns:
            Path to workspace if found, None otherwise
        """
        try:
            for item in self.tasks_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"{task_id}_"):
                    return item
        except Exception as e:
            logger.debug("executor.workspace.lookup_failed", task_id=task_id, error=str(e))
        return None

    def cleanup_workspace(self, task_id: int, force: bool = False) -> bool:
        """Clean up task workspace

        Args:
            task_id: Task ID
            force: Force cleanup even if files exist (default: False)
        Returns:
            True if workspace was removed, False otherwise.
        """
        workspace = self._find_task_workspace(task_id)

        if workspace is None or not workspace.exists():
            logger.debug("executor.workspace.missing", task_id=task_id)
            return False

        try:
            # Check if workspace is empty or force cleanup
            contents = list(workspace.iterdir())
            if force or len(contents) == 0:
                shutil.rmtree(workspace)
                logger.debug("executor.workspace.cleaned", workspace=str(workspace))
                return True
            logger.debug("executor.workspace.skip_cleanup", workspace=str(workspace))
            return False
        except Exception as e:
            logger.error("executor.workspace.cleanup_failed", workspace=str(workspace), error=str(e))
            return False

    def get_workspace_path(self, task_id: int, project_id: Optional[str] = None) -> Optional[Path]:
        """Get path to task workspace

        Args:
            task_id: Task ID
            project_id: Optional project identifier

        Returns:
            Path to workspace if found, None otherwise
        """
        if project_id:
            project_workspace = self.projects_dir / project_id
            return project_workspace if project_workspace.exists() else None

        return self._find_task_workspace(task_id)

    def workspace_exists(self, task_id: int) -> bool:
        """Check if workspace exists for task

        Args:
            task_id: Task ID

        Returns:
            True if workspace exists
        """
        return self._find_task_workspace(task_id) is not None
