"""Presentation only: never changes model context, tool execution, or retrieval policy."""

import io
import re
import unicodedata

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


def terminal_text(value: str) -> str:
    value = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", value)
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return "".join(c for c in value if c in "\n\t" or unicodedata.category(c) not in {"Cc", "Cf"})


class ChatTerminal:
    def __init__(self, *, write=print, secrets=(), console=None):
        self.write = write
        self.secrets = secrets
        self.console = console or Console(highlight=False)
        self.expanded = False
        self.last_output = "No tool output yet."

    def clean(self, value):
        text = str(value)
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return terminal_text(text)

    def render(self, value):
        if self.write is print:
            self.console.print(value)
        else:
            stream = io.StringIO()
            Console(file=stream, width=88, color_system=None, highlight=False).print(value)
            self.write(stream.getvalue().rstrip())

    def notice(self, text):
        self.render(Text(self.clean(text), style="dim"))

    def panel(self, title, text, *, color="cyan"):
        self.render(
            Panel(
                Text(self.clean(text)),
                title=Text(title, style="bold"),
                border_style=color,
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def header(self, workspace, model, memory, output):
        backend = getattr(workspace, "backend", "docker")
        config = getattr(model, "config", None)
        deployment = getattr(config, "deployment", "configured model")
        rows = [
            ("Repository", str(getattr(workspace, "repo", "repository"))),
            ("Snapshot", getattr(workspace, "base_commit", "")[:12]),
            ("Model", deployment),
            (
                "Execution",
                "Docker · isolated · network disabled"
                if backend == "docker"
                else "LOCAL SHELL · NOT ISOLATED",
            ),
            (
                "Memory",
                "HydraDB connected · queries optional" if memory else "Off · workspace only",
            ),
        ]
        if memory:
            rows.append(("Collection", getattr(memory, "collection", "configured collection")))
            rows.append(
                (
                    "Index",
                    "Reusing existing graph · no uploads"
                    if getattr(memory, "retrieval_only", False)
                    else "Prepared for this session",
                )
            )
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(overflow="fold")
        for label, value in rows:
            table.add_row(Text(label), Text(self.clean(value)))
        self.render(
            Panel(
                Group(
                    Text("HYDRA / CODE", style="bold cyan"),
                    Text("A conversation with your repository", style="dim"),
                    Text(""),
                    table,
                    Text(""),
                    Text(
                        "/help  commands    /diff  changes    /graph  memory    /exit  save & close",
                        style="dim",
                    ),
                ),
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self.notice(
            f"Session: {output}\nEdits are saved as patches; your checkout stays unchanged."
        )

    def begin(self, number):
        self.render(Text(f"\n── Turn {number} · agent working ──", style="bold cyan"))

    def activity(self, name, argument):
        label = "GRAPH SEARCH" if name == "memory_search" else "SHELL"
        self.render(
            Text(
                f"  {label}  {self.clean(argument)[:500]}",
                style="cyan" if name == "memory_search" else "dim",
            )
        )

    def tool_result(self, result):
        if "output" in result:
            self.last_output = self.clean(result["output"])
            lines = self.last_output.splitlines()
            preview = self.last_output if self.expanded else "\n".join(lines[:5])[:600]
            exit_code = result.get("exit_code", "?")
            color = "green" if exit_code == 0 else "red"
            self.render(
                Text(
                    f"  ↳ exit {exit_code}" + (" · timed out" if result.get("timed_out") else ""),
                    style=color,
                )
            )
            if preview.strip():
                self.render(Text("    " + preview.replace("\n", "\n    "), style="dim"))
            if len(preview) < len(self.last_output):
                self.notice(
                    "    Preview collapsed · /last to inspect · /tools to expand future output"
                )
            if result.get("truncated"):
                self.notice(
                    "    Tool output also reached the harness limit; use a narrower command."
                )
        elif "hits" in result:
            paths = list(dict.fromkeys(hit.get("path", "unknown") for hit in result["hits"]))
            self.render(Text(f"  ↳ {len(result['hits'])} evidence chunks", style="cyan"))
            for path in paths[:4]:
                self.notice("    " + path)
        elif "error" in result:
            self.render(Text("  ↳ " + self.clean(result["error"]), style="red"))

    def answer(self, text):
        self.render(
            Panel(
                Markdown(self.clean(text), hyperlinks=False),
                title=Text("HYDRA", style="bold cyan"),
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def diff(self, patch):
        if not patch:
            self.notice("No changes.")
            return
        self.render(
            Panel(
                Syntax(self.clean(patch), "diff", theme="ansi_dark", word_wrap=True),
                title="Cumulative patch",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

    def graph(self, memory, usage=None):
        if memory is None:
            self.panel("Memory", "Off · this session uses workspace tools only.", color="yellow")
            return
        text = f"Collection: {getattr(memory, 'collection', 'configured collection')}\n"
        text += f"Session queries: {getattr(memory, 'search_calls', 0)}/40\n"
        text += (
            "Retrieval is available, not enforced. A connected graph does not mean it was queried."
        )
        if usage:
            text += f"\nLast turn: {usage['searches']} tool calls · {usage['requests']} HTTP query attempts · {usage['evidence_chunks']} chunks · {usage['errors']} errors"
        self.panel("HydraDB memory", text)

    def footer(self, result, total, budget, patch_bytes):
        usage = result["graph_usage"]
        if not usage["enabled"]:
            graph = "Memory off"
        elif not usage["searches"]:
            graph = "Graph NOT queried this turn"
        elif not usage["requests"]:
            graph = f"Graph tool called · no HTTP query sent · {usage['errors']} errors"
        else:
            graph = f"Graph: {usage['requests']} HTTP query attempts · {usage['evidence_chunks']} chunks · {usage['errors']} errors"
        status = {"submitted": "Turn complete", "model_stopped": "Answered"}.get(
            result["status"], result["status"]
        )
        self.render(
            Text(
                f"{status}  ·  {result['total_tokens']:,} tokens  ·  {result.get('elapsed_seconds', 0):.1f}s  ·  patch {patch_bytes:,} B",
                style="dim",
            )
        )
        self.render(
            Text(graph, style="yellow" if usage["enabled"] and not usage["searches"] else "cyan")
        )
        self.notice(
            f"Session {total:,}/{budget:,} tokens · /diff to review · /exit to save & close"
        )
