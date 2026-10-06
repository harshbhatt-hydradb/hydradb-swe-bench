"""Reproducible CodeWikiBench pipeline: prepare, index, generate, evaluate, report."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from .agent import Trace
from .bench_data import atomic_json, digest, read_json
from .codewiki_agent import DocumentationAgent, UsageBudget, generate
from .codewiki_data import (
    DATASET,
    DATASET_REVISION,
    EVALUATOR_REVISION,
    POLICY,
    REPOSITORIES,
    code_corpus,
    generation_fingerprint,
    prepare_record,
)
from .codewiki_explore import ExplorationConfig
from .codewiki_memory import (
    DEFAULT_INDEX_WORKERS,
    MAX_INDEX_WORKERS,
    IndexRetryExhausted,
    index_corpus,
    open_index,
)
from .codewiki_terminal import CodeWikiTerminal
from .config import AzureConfig, HydraConfig, openrouter_config
from .memory import MemoryScope
from .model import AzureModel

# Fixed run shape. Token totals are recorded; they do not stop a run.
PAGES = 6
AGENT_STEPS = 20
INDEX_TIMEOUT = 24 * 60 * 60
INDEX_ATTEMPTS = 2
SURVEY = ExplorationConfig()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        nargs="?",
        choices=["run", "prepare", "index", "generate", "evaluate", "report"],
        default="run",
    )
    p.add_argument(
        "--repos",
        nargs="+",
        default=None,
        help="Repository names; resume a subset of a saved campaign, or use 'all' for all 22",
    )
    p.add_argument("--output", type=Path, default=Path("runs/codewiki"))
    p.add_argument("--cache", type=Path, default=Path("targets/codewiki"))
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument(
        "--index-workers",
        type=int,
        choices=range(1, MAX_INDEX_WORKERS + 1),
        metavar=f"1-{MAX_INDEX_WORKERS}",
        default=DEFAULT_INDEX_WORKERS,
        help=f"Concurrent upload/status requests, including final verification (default: {DEFAULT_INDEX_WORKERS})",
    )
    p.add_argument(
        "--plain", action="store_true", help="Plain terminal logs without colors or animation"
    )
    p.add_argument(
        "--agent-provider",
        choices=["azure", "openrouter"],
        default="azure",
        help="Generation backend; 'openrouter' uses OPEN_ROUTER_API_KEY for the agent model",
    )
    p.add_argument(
        "--agent-model",
        default="openai/gpt-6-astra",
        help="OpenRouter agent model id (only used when --agent-provider openrouter)",
    )
    return p


def get_corpus(root, metadata):
    path = root / "inference" / "corpus.json"
    if path.exists():
        saved = read_json(path)
        scope = MemoryScope(**saved["scope"])
        if (scope.repository, scope.base_commit) != (metadata["repo_url"], metadata["commit_id"]):
            raise ValueError("Corpus belongs to a different repository/commit")
    else:
        saved = None
        scope = MemoryScope(
            metadata["repo_url"],
            metadata["commit_id"],
            "codewikibench:" + metadata["repo_name"],
            "codewiki_" + uuid.uuid4().hex,
        )
    corpus = code_corpus(root / "inference" / "snapshot.tar", scope)
    if saved is not None and saved != corpus.manifest():
        raise ValueError("Corpus policy or snapshot changed; choose a new output directory")
    atomic_json(path, corpus.manifest())
    return corpus


def report(root, campaign):
    lines = [
        "# CodeWikiBench — HydraDB documentation agent",
        "",
        f"Dataset: `{DATASET}` at `{DATASET_REVISION}`.",
        f"Published judge source: `{EVALUATOR_REVISION}`.",
        "",
        (
            "This is a selected-repository run. Evaluation uses the CodeWikiBench paper judges "
            "(Gemini 2.5 Flash, GPT-OSS 120B, and Kimi K2), averaged with the published weights. "
            "It is not a full-suite result unless all 22 repositories finish."
        ),
        "",
        "| Repository | Indexed sources | Wiki pages | Judged criteria | Coverage / 100 | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    records = []
    for name in campaign["repos"]:
        repo_root = root / name

        def optional(relative, repo_root=repo_root):
            p = repo_root / relative
            return read_json(p) if p.exists() else {}

        index = optional("index-manifest.json")
        wiki = optional("wiki/generation.json")
        evaluation = optional("evaluation-paper-panel/result.json") or optional(
            "evaluation/result.json"
        )
        run = optional("run.json")
        completed = sum(
            v.get("indexing_status") == "completed" for v in index.get("statuses", {}).values()
        )
        status = evaluation.get("status", run.get("status", "not_started"))
        score = evaluation.get("score_percent")
        judged = evaluation.get("judged", 0)
        criteria = evaluation.get("criteria", 0)
        if evaluation.get("judges"):
            leaf_count = next(iter(evaluation["judges"].values())).get("criteria", 0)
            criteria = leaf_count * len(evaluation["judges"])
            judged = evaluation.get("judgments", judged)
        records.append({"repo": name, "score": score, "status": status})
        lines.append(
            f"| {name} | {completed}/{index.get('source_count', 0)} | "
            f"{len(wiki.get('completed_pages', {}))} | "
            f"{judged}/{criteria} | "
            f"{score if score is not None else '—'} | {status} |"
        )
        skipped = index.get("skipped", {})
        if skipped:
            paths = ", ".join(f"`{e['path']}` ({e['error_code']})" for e in skipped.values())
            lines += [
                "",
                (
                    f"{name}: {len(skipped)} source(s) excluded from retrieval after failing "
                    f"indexing: {paths}. These remain readable to the agent but are absent from "
                    "HydraDB search results."
                ),
            ]
        adapters = evaluation.get("judgment_adapters", {})
        if len(adapters) > 1:
            versions = ", ".join(f"`{version}`: {count}" for version, count in adapters.items())
            lines += ["", f"{name}: resumed evaluation includes judgments from {versions}."]
        exploration = wiki.get("exploration")
        if exploration:
            records[-1]["exploration"] = exploration
            coverage = exploration["coverage"]
            lines += [
                "",
                (
                    f"{name}: exploration {exploration['status']} · {coverage['topics_explored']} topics · "
                    f"{coverage['cited_source_files']}/{coverage['eligible_source_files']} source files cited · "
                    f"{exploration['deferred_dependencies']} deferred dependencies · "
                    f"{coverage['unresolved_questions']} unresolved questions. "
                    f"[Exploration notes]({name}/wiki/exploration/README.md). "
                    "Dependency labels are inferred from inspected source, not a verified call graph."
                ),
            ]
        if run.get("error"):
            lines += ["", f"{name} stopped: `{run['error']}`."]
    complete = [r for r in records if r["score"] is not None and r["status"] == "completed"]
    lines += ["", f"Completed repositories: {len(complete)}/{len(records)}."]
    if len(complete) == len(records):
        lines += [
            f"Selected-repository macro-average: {sum(r['score'] for r in complete) / len(complete):.3f}/100."
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "Scores measure whether generated documentation covers the published criteria. "
            "They do not prove claims or graph edges are correct. Source-link validation checks "
            "paths and line ranges only. Mermaid blocks are exported but not renderer-validated."
        ),
        "",
        (
            "Generation sees only eligible code/build/test sources. Reference docs, benchmark "
            "rubrics, prose files, visual fixtures and lockfiles are excluded from agent inputs. "
            "The original snapshot and reference material stay in controller/evaluator artifacts; "
            "the agent has no filesystem or shell tool. Source comments remain available."
        ),
        "",
        (
            "Generator and judge deployments are recorded per repository. Using the same model "
            "for both introduces possible self-evaluation bias. Model token usage is recorded; "
            "HydraDB internal token usage and dollar cost are not exposed by this harness."
        ),
        "",
        (
            "Judge errors remain unscored and block an overall score. Re-running the same command "
            "resumes index status checks, completed pages and completed judgments."
        ),
    ]
    for name in campaign["repos"]:
        lines += [
            "",
            f"- [{name} generated wiki]({name}/wiki/README.md)",
            f"- [{name} evaluation]({name}/evaluation/result.json)",
            f"- [{name} index manifest]({name}/index-manifest.json)",
        ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        root / "report.json",
        {"repositories": records, "completed": len(complete), "assigned": len(records)},
    )


def run_paper_judges(
    output: Path, repo: str, env_file: Path, plain: bool, *, campaign_lock_fd: int | None = None
) -> int:
    """Score one repository with the CodeWikiBench paper's three judges."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_codewiki_paper_panel.py"
    command = [
        sys.executable,
        str(script),
        "evaluate",
        "--output",
        str(output),
        "--repos",
        repo,
        "--env-file",
        str(env_file),
    ]
    if plain:
        command.append("--plain")
    if campaign_lock_fd is not None:
        command += ["--campaign-lock-fd", str(campaign_lock_fd)]
    return subprocess.run(
        command,
        check=False,
        pass_fds=() if campaign_lock_fd is None else (campaign_lock_fd,),
    ).returncode


def main(argv=None):
    args = parser().parse_args(argv)
    terminal = CodeWikiTerminal(plain=args.plain)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A process lock prevents concurrent upload/upsert or checkpoint corruption.
    import fcntl

    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This CodeWikiBench run is already active") from None
        campaign_path = root / "campaign.json"
        campaign = read_json(campaign_path) if campaign_path.exists() else None
        names = args.repos or (campaign["repos"] if campaign else ["Chart.js"])
        if names == ["all"]:
            names = list(REPOSITORIES)
        if not names or len(set(names)) != len(names) or set(names) - set(REPOSITORIES):
            raise SystemExit("Select unique CodeWikiBench repository names, or 'all'")
        if campaign and set(names) - set(campaign["repos"]):
            raise SystemExit(
                "Repositories are outside this saved campaign; choose a new --output directory"
            )
        expected = {
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "evaluator_revision": EVALUATOR_REVISION,
            # A resume selection controls execution, without rewriting the campaign
            # membership or removing other repositories from the saved report.
            "repos": campaign["repos"] if campaign else names,
            "corpus_policy": POLICY,
        }
        if campaign and campaign != expected:
            raise SystemExit("Campaign selection/pins changed; choose a new --output directory")
        campaign = expected
        atomic_json(campaign_path, campaign)
        if args.stage == "report":
            report(root, campaign)
            terminal.summary(read_json(root / "report.json")["repositories"], root / "report.md")
            return 0
        load_dotenv(args.env_file, override=False)
        stages = (
            ["prepare", "index", "generate", "evaluate"] if args.stage == "run" else [args.stage]
        )
        failed = False
        interrupted = False
        records = []
        terminal.header(names, stages, root)
        for repo_number, name in enumerate(names, 1):
            repo_root = root / name
            repo_root.mkdir(parents=True, exist_ok=True)
            state = {
                "status": "running",
                "started_at": time.time(),
                "stage": stages[0],
                "index_workers": args.index_workers,
            }
            state_path = repo_root / "run.json"
            score = None
            terminal.log(f"Repository {repo_number}/{len(names)}: {name}", contextual=False)
            try:
                for stage_number, stage in enumerate(stages, 1):
                    state["stage"] = stage
                    atomic_json(state_path, state)
                    terminal.start_stage(name, stage, stage_number, len(stages))
                    if stage == "prepare":
                        terminal.activity("Loading pinned record and source archive")
                        metadata = prepare_record(repo_root, name, args.cache.resolve())
                        corpus = get_corpus(repo_root, metadata)
                        terminal.log(
                            f"{len(corpus.sources):,} eligible files · "
                            f"{sum(s.size for s in corpus.sources):,} bytes · "
                            f"commit {metadata['commit_id'][:12]}"
                        )
                        terminal.complete_stage()
                        continue
                    metadata = read_json(repo_root / "inference" / "task.json")
                    corpus = get_corpus(repo_root, metadata)
                    if stage in ("index", "generate"):
                        hydra = HydraConfig.from_env()
                        terminal.secrets = (hydra.api_key,)
                        trace = Trace(
                            repo_root
                            / (
                                "index-events.jsonl"
                                if stage == "index"
                                else "generation-events.jsonl"
                            ),
                            secrets=(hydra.api_key,),
                            observer=terminal.observe,
                        )
                        if stage == "index":
                            memory = index_corpus(
                                hydra,
                                corpus,
                                repo_root / "index-manifest.json",
                                timeout=INDEX_TIMEOUT,
                                workers=args.index_workers,
                                max_attempts=INDEX_ATTEMPTS,
                                max_failures=0,
                                report=trace.emit,
                            )
                            memory.close()
                        else:
                            if args.agent_provider == "openrouter":
                                config = openrouter_config(args.agent_model)
                            else:
                                config = AzureConfig.from_env()
                            trace.secrets = (hydra.api_key, config.api_key)
                            terminal.secrets = trace.secrets
                            terminal.log(
                                f"Model: {config.deployment} ({args.agent_provider}) · "
                                f"{PAGES} pages · {AGENT_STEPS} steps/session"
                            )
                            terminal.activity("Verifying existing HydraDB index")
                            memory = open_index(
                                hydra,
                                corpus,
                                repo_root / "index-manifest.json",
                                trace.emit,
                                workers=args.index_workers,
                            )
                            model = AzureModel(config)
                            try:
                                identity = {
                                    "metadata": metadata,
                                    "corpus": digest(corpus.manifest()),
                                    "model": config.deployment,
                                    "endpoint": config.endpoint,
                                    "reasoning": config.reasoning_effort,
                                    "pages": PAGES,
                                    "steps": AGENT_STEPS,
                                    "max_tokens": 0,
                                    "code": generation_fingerprint(),
                                    "policy": POLICY,
                                    "exploration": {
                                        "max_depth": SURVEY.max_depth,
                                        "max_modules": SURVEY.max_modules,
                                        "max_branches": SURVEY.max_branches,
                                    },
                                }
                                budget = UsageBudget(
                                    repo_root / "generation-usage.json",
                                    0,
                                    context_bytes=0,
                                )
                                # Migrate query accounting if resuming a pre-cache generation attempt.
                                cache_path = repo_root / "retrieval-cache.json"
                                if not cache_path.exists() and trace.path.exists():
                                    memory.search_calls = sum(
                                        json.loads(line).get("event") == "retrieval_start"
                                        for line in trace.path.read_text().splitlines()
                                    )
                                memory.configure_search(cache_path, max_queries=0)
                                terminal.log(
                                    f"Exploration: one repository survey · "
                                    f"up to {SURVEY.max_branches} named modules"
                                )
                                agent = DocumentationAgent(
                                    model, memory, corpus, budget, trace, max_steps=AGENT_STEPS
                                )
                                generation = generate(
                                    agent,
                                    repo_root / "wiki",
                                    metadata,
                                    max_pages=PAGES,
                                    identity=identity,
                                    exploration_config=SURVEY,
                                )
                                terminal.log(
                                    f"Wiki complete · {len(generation['completed_pages'])} pages · "
                                    f"{budget.data['total_tokens']:,} tokens used"
                                )
                            finally:
                                model.close()
                                memory.close()
                    elif stage == "evaluate":
                        terminal.log("Judges: Gemini 2.5 Flash, GPT-OSS 120B, Kimi K2 · averaged")
                        if run_paper_judges(
                            root, name, args.env_file, args.plain, campaign_lock_fd=lock.fileno()
                        ):
                            raise RuntimeError("Paper judge panel did not finish")
                        result = read_json(repo_root / "evaluation-paper-panel" / "result.json")
                        if result.get("status") != "completed":
                            missing = result.get("criteria", 0) - result.get("judged", 0)
                            raise RuntimeError(
                                f"{missing or 'some'} criteria remain unscored; resume evaluation"
                            )
                        score = result["score_percent"]
                        terminal.log(f"Rubric coverage: {score:.3f}/100")
                    terminal.complete_stage()
                state["status"] = "completed" if args.stage == "run" else stages[-1] + "_completed"
            except KeyboardInterrupt:
                interrupted = True
                state["status"] = "interrupted"
                terminal.failed_stage("Checkpointed artifacts retained", interrupted=True)
            except Exception as exc:  # noqa: BLE001 -- preserve resumable state and sanitize errors
                failed = True
                state["status"] = "failed"
                if isinstance(exc, IndexRetryExhausted):
                    state["index_retries_exhausted"] = True
                    state["index_max_attempts"] = INDEX_ATTEMPTS
                # Network/provider exceptions may include credentials or request payloads.
                state["error"] = (
                    str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
                )
                for config_type in (AzureConfig, HydraConfig):
                    try:
                        state["error"] = state["error"].replace(
                            config_type.from_env().api_key, "[REDACTED]"
                        )
                    except ValueError:
                        pass
                router_key = os.environ.get("OPEN_ROUTER_API_KEY", "").strip()
                if router_key:
                    state["error"] = state["error"].replace(router_key, "[REDACTED]")
                terminal.failed_stage(state["error"])
            finally:
                terminal.stop()
                state["elapsed_seconds"] = round(time.time() - state["started_at"], 3)
                atomic_json(state_path, state)
                report(root, campaign)
            records.append({"repo": name, "status": state["status"], "score": score})
            if state["status"] in {"failed", "interrupted"}:
                terminal.log(f"Details: {state_path} · traces: {repo_root}")
                if state.get("index_retries_exhausted"):
                    terminal.log(
                        "Upload retries exhausted. Repeating this command will not re-upload "
                        "the failed sources. Resolve the HydraDB error or explicitly raise "
                        "--index-max-attempts to allow another attempt; completed sources are retained."
                    )
                elif state["stage"] == "evaluate":
                    terminal.log(
                        "Resume with the evaluate stage and the same --output and --repos; "
                        "saved wiki pages and successful judgments are reused. "
                        "The paper panel fixes its judge models and protocol limits."
                    )
                else:
                    terminal.log(
                        "Resume by rerunning the same command; completed artifacts are retained"
                    )
            if interrupted:
                break
        if interrupted:
            records.extend(
                {"repo": name, "status": "not_started", "score": None}
                for name in names[len(records) :]
            )
        terminal.summary(records, root / "report.md")
        if interrupted:
            return 130
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
