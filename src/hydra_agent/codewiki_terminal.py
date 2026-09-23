"""Human-readable CodeWikiBench progress; detailed payloads stay in JSONL traces."""

import math
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from .terminal import terminal_text


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class CodeWikiTerminal:
    def __init__(self, *, plain=False, console=None, clock=time.monotonic):
        self.console = console or Console(highlight=False, color_system=None if plain else "auto")
        self.clock = clock
        self.started = clock()
        self.secrets = ()
        self.repo = self.stage = ""
        self.stage_started = self.started
        self.progress_marks = {}
        self.live = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", markup=False),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            disable=plain or not self.console.is_terminal or self.console.is_dumb_terminal,
        )
        self.task = None

    def clean(self, value):
        value = str(value)
        for secret in self.secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return " ".join(terminal_text(value).split())

    def log(self, message, *, style="", contextual=True):
        line = Text(f"[{duration(self.clock() - self.started)}] ", style="dim")
        if contextual and self.repo:
            line.append(f"{self.repo} / {self.stage}  ", style="cyan")
        line.append(self.clean(message), style=style)
        self.console.print(line)

    def header(self, names, stages, output):
        self.console.print(
            Text("CodeWikiBench · HydraDB documentation pipeline", style="bold cyan")
        )
        self.log(f"Repositories ({len(names)}): {', '.join(names)}", contextual=False)
        self.log(f"Stages: {' → '.join(stages)}", contextual=False)
        self.log(f"Artifacts: {output}", contextual=False)

    def start_stage(self, repo, stage, number, total):
        self.stop()
        self.repo, self.stage = repo, stage
        self.stage_started = self.clock()
        self.progress_marks.clear()
        self.log(f"START {number}/{total}", style="bold")
        self.task = self.live.add_task(f"{repo} / {stage}", total=None)
        self.live.start()

    def activity(self, message, *, log=True):
        if self.task is not None:
            self.live.update(
                self.task, description=self.clean(f"{self.repo} / {self.stage} · {message}")
            )
        if log:
            self.log(message)

    def stop(self):
        self.live.stop()
        if self.task is not None:
            self.live.remove_task(self.task)
            self.task = None

    def complete_stage(self):
        self.stop()
        self.log(f"DONE in {duration(self.clock() - self.stage_started)}", style="green")

    def failed_stage(self, message, *, interrupted=False):
        self.stop()
        status = "INTERRUPTED" if interrupted else "FAILED"
        self.log(
            f"{status} after {duration(self.clock() - self.stage_started)} · {message}",
            style="yellow" if interrupted else "red",
        )

    def progress(self, key, completed, total, message, *, force=False):
        percent = 100 * completed / total if total else 0
        # Reserve 100% for completion, even when just one of many sources remains.
        percentage = "100%" if total and completed == total else f"{min(percent, 99.9):.1f}%"
        text = f"{message} · {completed:,}/{total:,} ({percentage})"
        self.activity(text, log=False)
        now = self.clock()
        previous = self.progress_marks.get(key)
        if (
            force
            or previous is None
            or completed - previous[0] >= max(1, math.ceil(total / 10))
            or (completed == total and previous[0] != total)
            or now - previous[1] >= 30
        ):
            self.log(text)
            self.progress_marks[key] = (completed, now)

    def emit(self, kind, **data):
        self.observe({"event": kind, **data})

    def observe(self, event):
        kind = event["event"]
        if kind == "codewiki_database":
            self.activity(event["message"])
        elif kind == "codewiki_index_workers":
            self.log(f"Index requests: {event['workers']} concurrent · batches of 20 sources")
            self.log(f"Upload limit: {event.get('max_attempts', 2)} attempts/source across resumes")
        elif kind == "codewiki_index_resume":
            self.activity(f"Revalidating {event['sources']:,} previously uploaded sources")
        elif kind == "codewiki_upload":
            self.progress(kind, event["uploaded"], event["total"], "Sources uploaded")
        elif kind == "codewiki_indexing":
            states = ", ".join(
                f"{count} {state.replace('_', ' ')}"
                for state, count in sorted(event.get("states", {}).items())
            )
            message = "Sources indexed" + (f" · waiting: {states}" if states else "")
            self.progress(kind, event["completed"], event["total"], message)
        elif kind == "codewiki_index_stalled":
            self.log(
                f"No new completed sources for {duration(event['seconds'])} · "
                f"{event['remaining']} remaining · waiting on HydraDB processing",
                style="yellow",
            )
            for source in event["sources"]:
                self.log(f"{source['path']} · {source['status']}", style="yellow")
        elif kind == "codewiki_retry":
            self.log(f"Retrying {event['sources']:,} failed sources", style="yellow")
        elif kind == "hydradb_request":
            if event["status"] in (429, 500, 502, 503, 504) and event["retry"] < 2:
                self.log(
                    f"HydraDB HTTP {event['status']} · retry {event['retry'] + 1}/2",
                    style="yellow",
                )
        elif kind == "hydradb_reused":
            self.activity(f"Existing index verified · {event['source_count']:,} sources")
        elif kind == "outline_ready":
            verb = "Reusing" if event["resumed"] else "Planned"
            self.activity(f"{verb} outline · {event['pages']} pages")
        elif kind in ("exploration_started", "exploration_completed"):
            verb = "Exploring" if kind == "exploration_started" else "Explored"
            self.activity(
                f"{verb} {event['path']} · depth {event['depth']} · "
                f"{event['completed']} topics saved · {event['pending']} queued"
            )
        elif kind == "exploration_reused":
            self.activity(f"Reusing exploration · {event['completed']} saved topics")
        elif kind == "exploration_upgraded":
            self.log(
                f"Applied exploration validation fix · retained {event['completed']} saved topics · "
                "original checkpoint backed up"
            )
        elif kind == "exploration_finished":
            self.log(
                f"Exploration {event['status']} · {event['modules']} modules named · "
                f"{', '.join(event['stop_reasons']) or 'survey finished'}"
            )
        elif kind == "retrieval_cache_hit":
            self.activity(f"Reusing retrieval · {event['hits']} evidence chunks")
        elif kind == "artifact_retry":
            self.log(f"{event['label']} · correcting artifact · {event['error']}", style="yellow")
        elif kind in ("page_started", "page_completed", "page_reused"):
            verb = {"page_started": "Writing", "page_completed": "Saved", "page_reused": "Reusing"}[
                kind
            ]
            self.activity(f"{verb} page {event['number']}/{event['total']} · {event['title']}")
        elif kind == "agent_task":
            if event["label"] == "outline":
                self.activity("Planning wiki outline")
        elif kind == "retrieval_start":
            self.activity("Searching repository with HydraDB")
        elif kind == "initial_retrieval":
            self.activity(f"Retrieved {len(event['hits'])} evidence chunks")
        elif kind == "model_request":
            counted = f"{event['tokens']:,} tokens"
            if event["limit"]:
                counted = f"{counted}/{event['limit']:,}"
            self.activity(
                f"{event['label']} · model step {event['step'] + 1}/{event['max_steps']} · "
                f"tokens used {counted}"
            )
        elif kind == "model":
            # Summarize tool names only: model responses contain full articles and source text.
            response = event["response"]
            calls = response["choices"][0]["message"].get("tool_calls", [])
            names = list(dict.fromkeys(call["function"]["name"] for call in calls))
            action = ", ".join(names) if names else "article returned"
            self.activity(f"{event['label']} · model response · {action}")
        elif kind == "tool":
            result = event["result"]
            if isinstance(result, dict) and "error" in result:
                self.log(f"{event['name']} · {result['error']}", style="yellow")
        elif kind == "evaluation_started":
            self.activity(f"Judging {event['total']:,} criteria · {event['workers']} workers")
        elif kind == "criterion_judged":
            message = (
                f"Criteria processed · {event['judged']:,} judged · "
                f"{event['reused']:,} reused · {event['errors']:,} errors"
            )
            if event["status"] != "completed":
                self.log(
                    f"Criterion {event['path']} unscored · "
                    f"{event.get('error') or event['error_type']}",
                    style="yellow",
                )
            self.progress(
                kind,
                event["completed"],
                event["total"],
                message,
                force=event["status"] != "completed",
            )

    def summary(self, records, report_path):
        self.stop()
        self.console.print()
        table = Table(box=None, padding=(0, 2))
        table.add_column("Repository", style="cyan")
        table.add_column("Status")
        table.add_column("Score", justify="right")
        for record in records:
            status = record["status"]
            style = "green" if status == "completed" or status.endswith("_completed") else "yellow"
            if status == "failed":
                style = "red"
            score = record.get("score")
            table.add_row(
                Text(self.clean(record["repo"])),
                Text(self.clean(status), style=style),
                Text(f"{score:.3f}" if score is not None else "—"),
            )
        self.console.print(table)
        self.log(f"Report: {report_path}", contextual=False)
