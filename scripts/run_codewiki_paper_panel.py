"""Generate one CodeWikiBench repository and judge it with the paper's three-model panel.

The wrapper uses the project environment for generation and the original pinned
environment for judging. Judges, prompts, temperature, and scoring stay those of the
Svelte paper protocol. Another repository uses its own pinned rubric. `evaluate`
resumes only the panel; --check-panel validates an existing wiki without calling models.
Original results are never imported as scores.
"""

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import math
import runpy
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hydra_agent.bench_data import atomic_json, digest, read_json
from hydra_agent.codewiki import parser as pipeline_parser
from hydra_agent.codewiki_data import REPOSITORIES, validate_rubrics

MODEL_IDS = ("google/gemini-2.5-flash", "openai/gpt-oss-120b", "moonshotai/kimi-k2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = pipeline_parser()
    parser.description = __doc__
    parser.set_defaults(repos=["svelte"])
    parser.add_argument("--protocol-run", type=Path, default=PROJECT / "runs/codewiki-svelte-paper")
    parser.add_argument(
        "--check-panel",
        action="store_true",
        help="Validate an existing wiki and panel without model calls",
    )
    parser.add_argument("--panel-worker", action="store_true", help=argparse.SUPPRESS)
    argv = sys.argv[1:] if argv is None else argv
    forbidden = {
        "--judge-provider",
        "--judge-model",
        "--judge-deployment",
        "--max-judge-tokens",
        "--max-judge-context-bytes",
        "--max-judge-output-tokens",
    }
    if any(a.split("=")[0] in forbidden for a in argv):
        parser.error(
            "The paper panel fixes its three models and protocol limits; remove single-judge flags"
        )
    args = parser.parse_args(argv)
    if (
        len(args.repos) != 1
        or args.repos[0] not in REPOSITORIES
        or args.stage
        not in (
            "run",
            "evaluate",
        )
    ):
        parser.error("Choose one CodeWikiBench repository; stages are run or evaluate")
    args.output, args.protocol_run = args.output.resolve(), args.protocol_run.resolve()
    if args.output == args.protocol_run or args.protocol_run in args.output.parents:
        parser.error("Use a separate output directory to preserve the protocol run")
    return args


def generation_args(args):
    skip = {"stage", "protocol_run", "check_panel", "panel_worker"}
    flags = []
    for name, value in vars(args).items():
        if name in skip or "judge" in name or value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        else:
            flags += (
                [flag, *[str(v) for v in value]] if isinstance(value, list) else [flag, str(value)]
            )
    return flags


def load_wiki(repo, protocol, rubric, *, strict_protocol=True):
    docs = read_json(repo / "wiki/structured_docs.json")
    tree = read_json(repo / "wiki/docs_tree.json")
    generation = read_json(repo / "wiki/generation.json")
    metadata = read_json(repo / "inference/task.json")
    candidate = read_json(repo / "evaluation/rubrics.json")
    candidate = candidate.get("rubrics", candidate) if isinstance(candidate, dict) else candidate
    if strict_protocol:
        if metadata != protocol["repository"] or candidate != rubric:
            raise ValueError("Repository commit or rubric differs from the previous paper run")
    else:
        validate_rubrics(candidate)
        if metadata.get("repo_name") != protocol.get("repo_name"):
            raise ValueError("Wiki repository does not match the selected CodeWikiBench repository")
    if generation.get("status") != "completed" or generation.get("docs_digest") != digest(docs):
        raise ValueError("Generation is incomplete or the generated documentation changed")
    if docs.get("metadata", {}).get("commit") != metadata["commit_id"]:
        raise ValueError("Wiki commit differs from the pinned repository")
    expected_tree = {k: v for k, v in docs.items() if k != "metadata"}
    expected_tree["subpages"] = [
        {**page, "content": {"markdown": "<detail_content>"}} for page in docs["subpages"]
    ]
    if tree != expected_tree:
        raise ValueError("Documentation tree does not match the generated pages")
    return docs, tree


def resume_identity(saved, current):
    """Reuse verified judgments when only the panel scripts changed.

    The wiki, rubric, models, navigation, and judgment parser still have to match.
    Saved checkpoints are checked again. A script change only retries leaves that
    never received a verified score.
    """

    if saved == current:
        return saved
    ignored = {"wrapper", "adapter"}
    if {k: v for k, v in saved.items() if k not in ignored} != {
        k: v for k, v in current.items() if k not in ignored
    }:
        raise ValueError("Wiki or evaluator changed; use a new output directory")
    return current


def use_repository_rubric(runner, rubric):
    """Score this repository's pinned rubric with the paper judges.

    ``runpy`` returns a copy of the runner namespace. ``audit`` and the trace
    recorder look up ``rubrics`` on the real module globals, so both mappings
    have to see the replacement.
    """

    def rubrics():
        return rubric

    runner["audit"].__globals__["rubrics"] = rubrics
    runner["rubrics"] = rubrics


def fresh_selection(runner, models):
    return {
        m["openrouter_id"]: {
            path: {"rerun": True, "reason": "New wiki; fresh judgment required"}
            for path in runner["leaves"](runner["rubrics"]())
        }
        for m in models
    }


def json_baseline(repo, result, leaves):
    """Published nlohmann/json scores from Table 4 of arXiv:2510.24428."""

    if repo != "json" or "score_percent" not in result:
        return ""
    score = result["score_percent"]
    spread = result["propagated_std_percent"]
    satisfied = result.get("leaves_unanimous", 0)
    return (
        "\n## Comparison on nlohmann/json\n\n"
        "These rows are the nlohmann/json results only. "
        "They use this repository's 57-leaf rubric.\n\n"
        "| System | Score | Leaves satisfied |\n"
        "| --- | ---: | ---: |\n"
        f"| This run | **{score:.2f} ± {spread:.2f}** | **{satisfied} / {leaves}** |\n"
        "| DeepWiki | 66.06 ± 3.08 | 33 / 57 |\n"
        "| CodeWiki, Kimi K2 | 61.28 ± 2.35 | 30 / 57 |\n\n"
        "A leaf counts as satisfied when all three judges score it 1. "
        f"This run is {score - 66.06:+.1f} over DeepWiki and {score - 61.28:+.1f} over CodeWiki "
        "on this repository. The paper did not publish a Claude Sonnet 4 score for nlohmann/json.\n"
    )


def aggregate(output, manifest, runner, adapter, docs):
    from judge import combine_evaluations, judge

    rubric = runner["rubrics"]()
    paths = list(runner["leaves"](rubric))
    all_scores, judges, missing = [], {}, []
    for name in MODEL_IDS:
        stem = name.replace("/", "_")
        evaluations = {}
        for path in paths:
            checkpoint = output / "judgments" / stem / (path + ".json")
            trace = output / "traces" / stem / (path + ".jsonl")
            if not adapter.verified_checkpoint(checkpoint, trace, docs):
                missing.append([name, path])
                continue
            value = read_json(checkpoint)
            evaluations[path] = {k: value[k] for k in ("score", "reasoning", "evidence")}
        if len(evaluations) != len(paths):
            continue
        scored = judge.calculate_scores_bottom_up(rubric, evaluations)
        target = output / "evaluation_results" / (stem + ".json")
        atomic_json(target, scored)
        if runner["audit"](target)["status"] != "completed":
            raise ValueError(f"Rubric/score audit failed for {name}")
        all_scores.append(evaluations)
        judges[name] = {
            "criteria": len(paths),
            "score_percent": 100
            * sum(n["score"] * n["weight"] for n in scored)
            / sum(n["weight"] for n in scored),
        }
    result = {
        "status": "incomplete" if missing else "completed",
        "missing": missing,
        "judges": judges,
        "judgments": len(MODEL_IDS) * len(paths) - len(missing),
        "usage": adapter.usage(output),
        "elapsed_seconds": time.time() - manifest["started_at"],
    }
    if not missing:
        combined = combine_evaluations.calculate_scores_bottom_up(
            rubric, combine_evaluations.combine_leaf_evaluations(all_scores, "average")
        )
        score = (
            100
            * sum(n["score"] * n["weight"] for n in combined)
            / sum(n["weight"] for n in combined)
        )
        if not math.isclose(
            score, sum(j["score_percent"] for j in judges.values()) / 3, abs_tol=1e-10
        ):
            raise ValueError("Panel aggregation mismatch")
        for path, node in runner["leaves"](combined).items():
            if not math.isclose(
                node["score"], sum(e[path]["score"] for e in all_scores) / 3, abs_tol=1e-12
            ):
                raise ValueError(f"Combined leaf mismatch: {path}")
        atomic_json(output / "evaluation_results/combined.json", combined)
        std = combine_evaluations.combine_std_weighted(
            [n["std"] for n in combined], [n["weight"] for n in combined]
        )
        result.update(score_percent=score, propagated_std_percent=100 * std)
        result["leaves_unanimous"] = sum(
            all(evaluation[path]["score"] == 1 for evaluation in all_scores) for path in paths
        )
    atomic_json(output / "result.json", result)
    score_text = (
        f"{result['score_percent']:.3f}/100"
        if not missing
        else "unscored until all criteria finish"
    )
    rows = [f"| {name} | {value['score_percent']:.3f} |" for name, value in judges.items()]
    (output / "report.md").write_text(
        f"# {docs['title']} — paper judge panel\n\n"
        f"Status: **{result['status']}**. Coverage: **{score_text}**.\n\n"
        f"Judgments with verified content access: {result['judgments']}/{len(paths) * 3}.\n\n"
        "| Judge | Coverage / 100 |\n| --- | ---: |\n" + "\n".join(rows) + "\n\n"
        "All judgments evaluate this wiki. Resumes reuse only this panel's verified checkpoints. "
        "The models, prompt, temperature 0, 36,000-token response/tool limits and weighted scoring "
        "match the previous repaired panel. Current OpenRouter hosting and the repaired navigation "
        "adapter remain differences from historical paper execution.\n\n"
        f"API-reported usage, including retries: {result['usage']['total_tokens']:,} tokens; "
        f"${result['usage']['reported_cost_usd']:.4f}. Failed/interrupted calls may have unreported charges.\n"
        + json_baseline(output.parent.name, result, len(paths))
        + "\n[Result](result.json) · [Manifest](manifest.json) · [Traces](traces/)\n"
    )
    return result


def panel_worker(args, runner, protocol):
    from dotenv import load_dotenv

    load_dotenv(args.env_file, override=False)
    runner["configure_environment"]()
    runner["apply_provider_compatibility"]()
    sys.path.insert(0, str(args.protocol_run / "upstream/src"))
    spec = importlib.util.spec_from_file_location(
        "codewiki_panel_adapter", PROJECT / "scripts/repair_codewiki_judge_access.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    name = args.repos[0]
    repo = args.output / name
    docs, tree = load_wiki(
        repo,
        protocol if name == "svelte" else {**protocol, "repo_name": name},
        runner["rubrics"](),
        strict_protocol=name == "svelte",
    )
    if name != "svelte":
        stored = read_json(repo / "evaluation/rubrics.json")
        stored = stored.get("rubrics", stored) if isinstance(stored, dict) else stored
        use_repository_rubric(runner, stored)
    output = repo / "evaluation-paper-panel"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [
        repo / p
        for p in (
            "wiki/structured_docs.json",
            "wiki/docs_tree.json",
            "wiki/generation.json",
            "evaluation/rubrics.json",
            "inference/task.json",
        )
    ]
    identity = {
        "inputs": {str(p): sha(p) for p in inputs},
        "models": list(MODEL_IDS),
        "protocol_manifest": sha(args.protocol_run / "manifest.json"),
        "adapter": sha(PROJECT / "scripts/repair_codewiki_judge_access.py"),
        "navigation": sha(PROJECT / "src/hydra_agent/codewiki_navigation.py"),
        "judgment_parser": sha(PROJECT / "src/hydra_agent/codewiki_eval.py"),
        "wrapper": sha(Path(__file__)),
    }
    path = output / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        manifest["identity"] = resume_identity(manifest["identity"], identity)
    else:
        manifest = {
            "identity": identity,
            "models": protocol["models"],
            "settings": {
                "temperature": 0,
                "max_tokens_per_response": 36000,
                "max_tokens_per_tool_response": 36000,
                "timeout_seconds": 300,
                "combination_method": "average",
                "validation_retries": 2,
                "attempts_per_criterion_per_invocation": 2,
                "request_starts_per_minute_per_model": 40,
            },
            "status": "prepared",
            "started_at": time.time(),
            "selection": fresh_selection(runner, protocol["models"]),
            "protocol": "Same repaired paper panel; every criterion starts fresh for this wiki",
        }
        for source in inputs:
            atomic_json(output / "inputs" / source.name, read_json(source))
        atomic_json(path, manifest)
    count = sum(len(v) for v in manifest["selection"].values())
    print(
        f"Paper panel: {', '.join(MODEL_IDS)}\n{count} judgments · 4 workers/model",
        flush=True,
    )
    if args.check_panel:
        # Exercise the exact adapter's recorded Svelte regressions, then validate new-wiki inputs above.
        asyncio.run(
            adapter.self_check(
                read_json(args.protocol_run / "upstream/data/svelte/hydra/structured_docs.json")
            )
        )
        print("New wiki and panel inputs verified; no model calls made.", flush=True)
        return 0
    manifest.update(status="running", workers_per_model=4)
    atomic_json(path, manifest)
    try:
        asyncio.run(
            adapter.execute(
                argparse.Namespace(output=output, workers=4),
                manifest,
                runner,
                docs,
                tree,
            )
        )
    except KeyboardInterrupt:
        manifest.update(status="interrupted")
        atomic_json(path, manifest)
        return 130
    for file, expected in identity["inputs"].items():
        if sha(Path(file)) != expected:
            raise ValueError("Wiki inputs changed during judging")
    runner["verify"]()
    result = aggregate(output, manifest, runner, adapter, docs)
    manifest.update(status=result["status"], finished_at=time.time())
    atomic_json(path, manifest)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Panel report: {output / 'report.md'}", flush=True)
    return 0 if result["status"] == "completed" else 1


def main(argv=None):
    args = parse_args(argv)
    runner = runpy.run_path(str(args.protocol_run / "runner.py"))
    protocol = runner["verify"]()
    if tuple(m["openrouter_id"] for m in protocol["models"]) != MODEL_IDS:
        raise ValueError("Protocol does not use the expected three judge models")
    if args.panel_worker:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return panel_worker(args, runner, protocol)
    if args.stage == "run" and not args.check_panel:
        for stage in ("prepare", "index", "generate"):
            completed = subprocess.run(
                [sys.executable, "-m", "hydra_agent.codewiki", stage, *generation_args(args)],
                check=False,
            )
            if completed.returncode:
                return completed.returncode
    forwarded = list(sys.argv[1:] if argv is None else argv)
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.12",
        "--with-requirements",
        str(args.protocol_run / "upstream/requirements.txt"),
        "--with",
        "tiktoken==0.11.0",
        "python",
        str(Path(__file__).resolve()),
        *forwarded,
        "--panel-worker",
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
