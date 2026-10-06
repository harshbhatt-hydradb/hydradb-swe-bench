import fcntl
import io
import os
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from rich.console import Console

from hydra_agent import codewiki
from hydra_agent.bench_data import atomic_json, read_json
from hydra_agent.codewiki_terminal import CodeWikiTerminal


def test_judge_failure_shows_actionable_reason():
    stream = io.StringIO()
    terminal = CodeWikiTerminal(console=Console(file=stream, width=240))
    terminal.emit(
        "criterion_judged",
        path="0.0.2",
        status="error",
        completed=3,
        total=96,
        judged=2,
        reused=0,
        errors=1,
        error_type="BudgetExceeded",
        error="Model token budget exhausted before request; raise --max-judge-tokens",
    )
    text = stream.getvalue()
    assert "Criterion 0.0.2 unscored" in text and "raise --max-judge-tokens" in text


def test_evaluate_uses_the_paper_judges(tmp_path, monkeypatch, capsys):
    atomic_json(tmp_path / "svelte/inference/task.json", {})
    atomic_json(
        tmp_path / "svelte/evaluation-paper-panel/result.json",
        {"status": "completed", "score_percent": 50.0},
    )
    monkeypatch.setattr(codewiki, "get_corpus", lambda *args: None)
    monkeypatch.setattr(codewiki, "load_dotenv", lambda *a, **kw: None)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        descriptor = int(command[command.index("--campaign-lock-fd") + 1])
        assert kwargs["pass_fds"] == (descriptor,)
        assert os.fstat(descriptor).st_ino == (tmp_path / ".lock").stat().st_ino
        with (tmp_path / ".lock").open("a") as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(codewiki.subprocess, "run", run)
    assert (
        codewiki.main(["evaluate", "--repos", "svelte", "--output", str(tmp_path), "--plain"]) == 0
    )
    assert commands and commands[0][1].endswith("run_codewiki_paper_panel.py")
    assert "evaluate" in commands[0] and "svelte" in commands[0]
    text = capsys.readouterr().out
    assert "Gemini 2.5 Flash" in text and "50.000/100" in text


def test_paper_panel_failure_gives_evaluate_only_resume_guidance(tmp_path, monkeypatch, capsys):
    atomic_json(tmp_path / "svelte/inference/task.json", {})
    monkeypatch.setattr(codewiki, "get_corpus", lambda *args: None)
    monkeypatch.setattr(codewiki, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(codewiki, "run_paper_judges", lambda *a, **kw: 1)
    assert (
        codewiki.main(["evaluate", "--repos", "svelte", "--output", str(tmp_path), "--plain"]) == 1
    )
    text = " ".join(capsys.readouterr().out.split())
    assert "Resume with the evaluate stage" in text
    assert "paper panel fixes its judge models" in text
    assert "--max-judge-tokens" not in text and "same --output, --repos and judge flags" not in text


def test_redirected_progress_is_readable_throttled_and_sanitized():
    stream = io.StringIO()
    now = [0.0]
    terminal = CodeWikiTerminal(console=Console(file=stream, width=160), clock=lambda: now[0])
    terminal.secrets = ("secret-key",)
    terminal.start_stage("Chart.js", "index", 2, 4)
    for _ in range(30):
        terminal.emit("codewiki_indexing", completed=20, total=100)
    assert stream.getvalue().count("Sources indexed") == 1
    now[0] = 30
    terminal.emit("codewiki_indexing", completed=20, total=100)
    terminal.emit("codewiki_indexing", completed=100, total=100)
    terminal.emit("codewiki_indexing", completed=100, total=100)
    terminal.log("[bold]secret-key[/bold]\x1b[2J\nspoofed line")
    terminal.complete_stage()
    text = stream.getvalue()
    assert text.count("Sources indexed") == 3
    assert "100/100 (100%)" in text
    assert "[00:30] Chart.js / index" in text
    assert "[bold][REDACTED][/bold] spoofed line" in text
    assert "secret-key" not in text and "\x1b" not in text and "\r" not in text
    assert "DONE in 00:30" in text
    assert not terminal.live.live.is_started


def test_resume_shows_checked_sources_separately_from_ready_sources_and_read_retries():
    stream = io.StringIO()
    terminal = CodeWikiTerminal(console=Console(file=stream, width=160), clock=lambda: 0)
    terminal.start_stage("svelte", "index", 2, 4)
    terminal.emit("codewiki_index_resume", sources=100)
    for _ in range(30):
        terminal.emit("codewiki_index_revalidated", checked=20, total=100, completed=18)
    terminal.emit(
        "hydradb_transport_error",
        method="GET",
        path="/context/status",
        error_type="ReadTimeout",
        retry=0,
        retrying=True,
        delay_seconds=1,
    )
    terminal.emit("codewiki_index_revalidated", checked=100, total=100, completed=98)
    terminal.emit("codewiki_indexing", completed=98, total=100)
    terminal.failed_stage("stop fixture")
    text = stream.getvalue()
    assert "Revalidating 100 previously uploaded sources" in text
    assert text.count("Sources revalidated") == 2
    assert "Sources revalidated · 98 ready" in text and "100/100 (100%)" in text
    assert "HydraDB GET /context/status · ReadTimeout · retry 1/2 in 1s" in text
    assert "Sources indexed" in text and "98/100 (98.0%)" in text
    assert not terminal.live.live.is_started


def test_final_verification_shows_parallel_workers_and_progress():
    stream = io.StringIO()
    terminal = CodeWikiTerminal(console=Console(file=stream, width=160), clock=lambda: 0)
    terminal.start_stage("svelte", "generate", 3, 4)
    terminal.emit("codewiki_verify_workers", workers=8, sources=100)
    terminal.emit("codewiki_index_verified", checked=20, total=100)
    terminal.emit("codewiki_index_verified", checked=100, total=100)
    terminal.emit("hydradb_reused", source_count=100)
    terminal.complete_stage()
    text = stream.getvalue()
    assert "Verifying 100 sources · 8 concurrent requests" in text
    assert "Sources verified · 100/100 (100%)" in text
    assert "Existing index verified · 100 sources" in text


@pytest.mark.parametrize("stage", ["index", "generate"])
@pytest.mark.parametrize("workers", [None, 16])
def test_cli_passes_worker_count_to_indexing_and_verification(
    tmp_path, monkeypatch, stage, workers
):
    from hydra_agent.hydradb import HydraError

    atomic_json(tmp_path / "svelte/inference/task.json", {})
    monkeypatch.setattr(codewiki, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(codewiki, "get_corpus", lambda *a: SimpleNamespace())
    monkeypatch.setattr(codewiki.HydraConfig, "from_env", lambda: SimpleNamespace(api_key="key"))
    monkeypatch.setattr(
        codewiki.AzureConfig,
        "from_env",
        lambda: SimpleNamespace(api_key="key", deployment="fixture"),
    )
    seen = []

    def check(*args, **kwargs):
        seen.append(kwargs["workers"])
        raise HydraError("stop after checking CLI configuration")

    monkeypatch.setattr(codewiki, "index_corpus" if stage == "index" else "open_index", check)
    args = [stage, "--repos", "svelte", "--output", str(tmp_path), "--plain"]
    if workers is not None:
        args += ["--index-workers", str(workers)]
    assert codewiki.main(args) == 1
    assert seen == [workers if workers is not None else 8]
    assert read_json(tmp_path / "svelte/run.json")["index_workers"] == seen[0]


@pytest.mark.parametrize("workers", ["0", "17", "-1"])
def test_cli_rejects_worker_count_outside_supported_range(workers):
    with pytest.raises(SystemExit) as exc:
        codewiki.parser().parse_args(["--index-workers", workers])
    assert exc.value.code == 2


@pytest.mark.parametrize("width", [40, 80, 120])
def test_terminal_summary_and_activity_fit_and_stop_on_failure(width):
    stream = io.StringIO()
    terminal = CodeWikiTerminal(console=Console(file=stream, width=width, force_terminal=True))
    terminal.start_stage("Chart.js", "generate", 3, 4)
    terminal.activity("Writing page 1/6 · Architecture and public API")
    terminal.failed_stage("Provider unavailable")
    assert not terminal.live.live.is_started
    assert terminal.task is None
    # Inspect the static summary without ANSI control sequences.
    output = io.StringIO()
    terminal.console = Console(file=output, width=width, color_system=None)
    terminal.summary([{"repo": "Chart.js", "status": "failed", "score": None}], "/tmp/report.md")
    assert "Chart.js" in output.getvalue() and "failed" in output.getvalue()
    assert max(cell_len(line) for line in output.getvalue().splitlines()) <= width


def test_model_payloads_stay_out_of_terminal_logs():
    stream = io.StringIO()
    terminal = CodeWikiTerminal(console=Console(file=stream, width=160))
    terminal.emit(
        "model",
        label="overview",
        response={
            "choices": [
                {
                    "message": {
                        "content": "PRIVATE ARTICLE",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": "PRIVATE SOURCE"}},
                        ],
                    }
                }
            ]
        },
    )
    terminal.emit("initial_retrieval", hits=[{"text": "PRIVATE EVIDENCE"}])
    assert "read_file" in stream.getvalue()
    assert "Retrieved 1 evidence chunks" in stream.getvalue()
    assert "PRIVATE" not in stream.getvalue()


@pytest.mark.parametrize(
    "failure,exit_code", [(None, 0), (ValueError, 1), (KeyboardInterrupt, 130)]
)
def test_cli_stage_outcomes_and_resume_guidance(tmp_path, monkeypatch, capsys, failure, exit_code):
    calls = []

    def prepare(root, name, cache):
        calls.append(name)
        if name == "Chart.js" and failure:
            raise failure("safe failure with secret-key")
        return {"commit_id": "a" * 40}

    monkeypatch.setattr(codewiki, "prepare_record", prepare)
    monkeypatch.setattr(
        codewiki, "get_corpus", lambda *args: SimpleNamespace(sources=[SimpleNamespace(size=42)])
    )
    monkeypatch.setattr(codewiki, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "secret-key")
    output = tmp_path / "run"
    assert (
        codewiki.main(
            [
                "prepare",
                "--repos",
                "Chart.js",
                "graphrag",
                "--output",
                str(output),
                "--plain",
            ]
        )
        == exit_code
    )
    text = capsys.readouterr().out
    assert "CodeWikiBench" in text and "Report:" in text
    assert "\x1b" not in text and '"event":' not in text and "secret-key" not in text
    status = read_json(output / "Chart.js/run.json")["status"]
    if failure:
        assert status == ("interrupted" if failure is KeyboardInterrupt else "failed")
        assert "Resume by rerunning" in text and "DONE" not in text.split("graphrag / prepare")[0]
        assert "INTERRUPTED" in text if failure is KeyboardInterrupt else "FAILED" in text
    else:
        assert status == "prepare_completed"
        assert "1 eligible files" in text and "DONE in" in text
    assert calls == (["Chart.js"] if failure is KeyboardInterrupt else ["Chart.js", "graphrag"])


def test_cli_resumes_one_saved_repository_without_rewriting_campaign(tmp_path, monkeypatch, capsys):
    calls = []

    def prepare(root, name, cache):
        calls.append(name)
        return {"commit_id": "a" * 40}

    monkeypatch.setattr(codewiki, "prepare_record", prepare)
    monkeypatch.setattr(
        codewiki, "get_corpus", lambda *args: SimpleNamespace(sources=[SimpleNamespace(size=42)])
    )
    monkeypatch.setattr(codewiki, "load_dotenv", lambda *a, **kw: None)
    common = ["prepare", "--output", str(tmp_path), "--plain"]
    assert codewiki.main([*common, "--repos", "graphrag", "svelte"]) == 0
    campaign = (tmp_path / "campaign.json").read_bytes()
    graph_run = (tmp_path / "graphrag/run.json").read_bytes()
    atomic_json(tmp_path / "svelte/index-manifest.json", {"source_count": 3854, "statuses": {}})
    index = (tmp_path / "svelte/index-manifest.json").read_bytes()
    calls.clear()

    assert codewiki.main([*common, "--repos", "svelte"]) == 0
    assert calls == ["svelte"]
    assert (tmp_path / "campaign.json").read_bytes() == campaign
    assert (tmp_path / "graphrag/run.json").read_bytes() == graph_run
    assert (tmp_path / "svelte/index-manifest.json").read_bytes() == index
    assert [r["repo"] for r in read_json(tmp_path / "report.json")["repositories"]] == [
        "graphrag",
        "svelte",
    ]
    calls.clear()
    with pytest.raises(SystemExit, match="outside this saved campaign"):
        codewiki.main([*common, "--repos", "Chart.js"])
    assert calls == []
    modified = read_json(tmp_path / "campaign.json")
    modified["dataset_revision"] = "changed"
    atomic_json(tmp_path / "campaign.json", modified)
    with pytest.raises(SystemExit, match="pins changed"):
        codewiki.main([*common, "--repos", "svelte"])
