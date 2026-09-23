"""Audit and repair a saved three-judge paper run without changing its artifacts.

Run with the source run's pinned requirements and tiktoken==0.11.0. --check
prepares the repair manifest and exercises content guards without model calls.
"""

import argparse
import asyncio
import contextvars
import functools
import hashlib
import json
import math
import os
import runpy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from pydantic_ai import Agent, ModelRetry, RunContext, Tool

from hydra_agent.codewiki_eval import JudgmentFormatError, parse_judgment
from hydra_agent.codewiki_judge_reply import promote_reasoning_channel
from hydra_agent.codewiki_navigation import (
    contains_documentation,
    documentation_paths,
    navigate,
    navigation_succeeded,
)

ACTIVE = contextvars.ContextVar("trace", default=None)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(event, **fields):
    context = ACTIVE.get()
    if context is None:
        return
    path, model, criterion = context
    value = json.dumps(
        {"event": event, "time": time.time(), "model": model, "criterion": criterion, **fields},
        default=str,
    )
    key = os.environ.get("API_KEY")
    if key:
        value = value.replace(key, "[REDACTED]")
    with path.open("a") as stream:
        stream.write(value + "\n")


@dataclass
class Evidence:
    docs: dict
    round_step: int = -1
    round_valid: bool = False
    successful_paths: int = 0
    failed_paths: int = 0
    rounds: list = field(default_factory=list)
    sent_contents: set = field(default_factory=set)


async def guarded_navigator(
    ctx: RunContext[Evidence],
    paths: list[list[Any]],
    query: str | None = None,
    search: str | None = None,
) -> str:
    """Read page content using paths such as [['subpages', 0, 'content', 'markdown']].

    Errors mean the read failed, not that a feature is undocumented. Correct errors
    using the returned available_paths before scoring. Request narrower paths if
    the response exceeds the documentation tool's token budget. Requests for page
    titles or descriptions include the owning page's actual content.
    Optional query/search hints also return literal, case-insensitive line matches
    within the requested content. Page content is still returned on its first read.
    """
    import config
    from utils import enc

    deps = ctx.deps
    if deps.round_step != ctx.run_step:
        deps.round_step = ctx.run_step
        deps.round_valid = True
    try:
        results = navigate(deps.docs, paths, max_bytes=None)
    except (ValueError, TypeError) as exc:
        results = {"error": str(exc), "available_paths": documentation_paths(deps.docs)}
    if isinstance(results, list):
        for entry in results:
            if "content" in entry:
                payload = json.dumps(entry["content"], indent=2)
                entry["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
                entry["cached"] = entry["content_sha256"] in deps.sent_contents
                hint = query if query is not None else search
                if hint is not None:
                    entry["literal_search"] = literal_search(entry["content"], hint)
        text = "".join(
            "--------------------------------\n"
            f"Path: {entry['path']}\n"
            + (
                (
                    f"Content already read in this conversation (SHA256 {entry['content_sha256']}). "
                    "Use the earlier complete tool output; the page has not changed.\n"
                    if entry["cached"]
                    else f"Content: \n{json.dumps(entry['content'], indent=2)}\n"
                )
                if "content" in entry
                else f"READ ERROR: {json.dumps(entry)}\n"
            )
            + (
                f"Literal search: {json.dumps(entry['literal_search'])}\n"
                if "literal_search" in entry
                else ""
            )
            + "--------------------------------\n"
            for entry in results
        )
    else:
        text = json.dumps(results)
    if len(enc.encode(text)) > config.MAX_TOKENS_PER_TOOL_RESPONSE:
        results = {
            "error": "Requested pages exceed the 36,000-token tool budget; request fewer pages",
            "available_paths": documentation_paths(deps.docs),
        }
        text = json.dumps(results)
    valid = navigation_succeeded(results)
    if valid:
        deps.sent_contents.update(entry["content_sha256"] for entry in results)
    deps.round_valid = deps.round_valid and valid
    if isinstance(results, list):
        deps.successful_paths += sum(navigation_succeeded([entry]) for entry in results)
        deps.failed_paths += sum(not navigation_succeeded([entry]) for entry in results)
    else:
        deps.failed_paths += 1
    deps.rounds.append({"step": ctx.run_step, "valid": valid})
    emit(
        "navigation",
        step=ctx.run_step,
        results=results,
        delivered=text,
        valid=valid,
        query=query,
        search=search,
    )
    if not valid:
        raise ModelRetry(text)
    return text


def literal_search(content, query):
    matches = []

    def visit(value):
        if isinstance(value, str):
            for number, line in enumerate(value.splitlines(), 1):
                if query and query.casefold() in line.casefold():
                    matches.append({"line": number, "text": line[:600]})
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(content)
    return {
        "query": query,
        "matches": matches[:20],
        "total_matches": len(matches),
        "note": "Literal matches in these requested pages only; this is not a coverage score.",
    }


def last_navigation_batch_succeeded(messages):
    def field(value, name, default=None):
        return (
            value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
        )

    for message in reversed(messages):
        parts = [
            part
            for part in field(message, "parts", [])
            if field(part, "tool_name") == "docs_navigator"
            and field(part, "part_kind") in ("tool-return", "retry-prompt")
        ]
        if parts:
            return all(field(part, "part_kind") == "tool-return" for part in parts)
    return False


def validate_output(ctx: RunContext[Evidence], output: str) -> str:
    if (
        not ctx.deps.successful_paths
        or not ctx.deps.round_valid
        or not last_navigation_batch_succeeded(ctx.messages)
    ):
        raise ModelRetry(
            "Read actual documentation before scoring. Correct failed reads with the listed "
            "content paths. A failed read is not evidence of missing documentation."
        )
    try:
        parse_judgment(output)
    except JudgmentFormatError as exc:
        raise ModelRetry(
            f"Return one valid JSON judgment, optionally in one JSON fence: {exc}"
        ) from exc
    return output


def instrument():
    from openai.resources.chat.completions import AsyncCompletions

    original = AsyncCompletions.create
    locks = {}
    next_start = {}

    @functools.wraps(original)
    async def create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        lock = locks.setdefault(model, asyncio.Lock())
        async with lock:
            await asyncio.sleep(max(0, next_start.get(model, 0) - time.monotonic()))
            next_start[model] = time.monotonic() + 1.5  # <=40 requests/minute/model
        settings = {k: kwargs.get(k) for k in ("model", "temperature", "max_completion_tokens")}
        if settings["temperature"] != 0 or settings["max_completion_tokens"] != 36000:
            raise ValueError("Request no longer matches paper temperature/output limits")
        emit("request", settings=settings)
        started = time.monotonic()
        try:
            result = await original(self, *args, **kwargs)
        except Exception as exc:
            emit(
                "request_error",
                error_type=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
            )
            raise
        emit(
            "response", response=result.model_dump(mode="json"), seconds=time.monotonic() - started
        )
        try:
            promote_reasoning_channel(result.choices[0].message)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return result

    AsyncCompletions.create = create


def select_attempts(source, runner, models, docs):
    """Replay the old tool, not the corrected resolver, to find affected scores."""
    from tools import docs_navigator_tool
    from tools.docs_navigator import DocsNavigator

    original_nav = DocsNavigator(
        str(source / "upstream/data/svelte/hydra/docs_tree.json"),
        str(source / "upstream/data/svelte/hydra/structured_docs.json"),
    )
    selection = {}
    for model in models:
        name = model["openrouter_id"]
        stem = name.replace("/", "_")
        scored = runner["leaves"](read(runner["RESULTS"] / (stem + ".json")))
        pending = {}
        matches = {}
        for line in (source / "traces" / (stem + ".jsonl")).read_text().splitlines():
            event = json.loads(line)
            path = event.get("criterion")
            if event["event"] == "response":
                for choice in event["response"]["choices"]:
                    for call in choice["message"].get("tool_calls") or []:
                        if call["function"]["name"] != "docs_navigator":
                            continue
                        try:
                            arguments = docs_navigator_tool.function_schema.validator.validate_json(
                                call["function"]["arguments"]
                            )
                            for requested in arguments["paths"]:
                                result = original_nav.get_content(requested)
                                valid = "error" not in result and contains_documentation(
                                    result.get("content"), requested
                                )
                                pending.setdefault(path, []).append(
                                    {
                                        "path": requested,
                                        "valid": valid,
                                        "error": result.get("error"),
                                    }
                                )
                        except (ValueError, TypeError, KeyError):
                            pending.setdefault(path, []).append(
                                {"valid": False, "error": "Malformed call"}
                            )
            elif event["event"] == "agent_error":
                pending[path] = []
            elif event["event"] == "agent_result":
                try:
                    judgment = parse_judgment(event["output"])
                except JudgmentFormatError:
                    judgment = None
                if (
                    path in scored
                    and judgment
                    == {
                        k: scored[path]["evaluation"][k] for k in ("score", "reasoning", "evidence")
                    }
                    and path not in matches
                ):
                    calls = pending.get(path, [])
                    matches[path] = {
                        "rerun": not calls or not all(c["valid"] for c in calls),
                        "successful_paths": sum(c["valid"] for c in calls),
                        "failed_paths": sum(not c["valid"] for c in calls),
                        "calls": calls,
                    }
                pending[path] = []
        for path in scored:
            if path not in matches:
                matches[path] = {"rerun": True, "reason": "No matching audited answer in trace"}
        selection[name] = matches
    return selection


async def self_check(docs):
    context = SimpleNamespace(
        deps=Evidence(docs),
        run_step=1,
        messages=[{"parts": [{"part_kind": "tool-return", "tool_name": "docs_navigator"}]}],
    )
    text = await guarded_navigator(context, [[["subpages", 1, "content", "markdown"]]])
    assert "<svelte:options>" in text and context.deps.round_valid
    validate_output(context, '{"score":1,"reasoning":"Present","evidence":"Page"}')
    context.run_step = 2
    try:
        await guarded_navigator(context, [["nonexistent"]])
    except ModelRetry as exc:
        assert "READ ERROR" in str(exc) and "available_paths" in str(exc)
    else:
        raise AssertionError("Invalid paths must trigger a bounded tool retry")
    assert not context.deps.round_valid
    try:
        validate_output(context, '{"score":0,"reasoning":"Missing","evidence":"None"}')
    except ModelRetry:
        pass
    else:
        raise AssertionError("A failed read was accepted as a zero score")
    context.run_step = 3
    await guarded_navigator(context, [["subpages", 1, "content", "markdown"]])
    assert context.deps.round_valid
    context.run_step = 4
    text = await guarded_navigator(context, [["subpages", 0, "description"]])
    assert "Architecture and Public Component API" in text and context.deps.round_valid
    context.run_step = 5
    text = await guarded_navigator(context, [["subpages", 0, "description"]], search="component")
    assert '"total_matches": 0' not in text and "Content already read" in text
    assert literal_search("A component\nOther text", "COMPONENT")["total_matches"] == 1
    assert literal_search("A component", "nonexistent")["matches"] == []
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    steps = iter(
        [
            ToolCallPart(
                "docs_navigator",
                {"paths": [["subpages", 1, "content", "markdown"]], "query": "options"},
            ),
            ToolCallPart("docs_navigator", {"paths": [42]}),
            TextPart('{"score":0,"reasoning":"Missing","evidence":"Failed read"}'),
            ToolCallPart("docs_navigator", {"paths": [[["subpages", 1, "content", "markdown"]]]}),
            TextPart('{"score":1,"reasoning":"Present","evidence":"Verified page"}'),
        ]
    )

    def fake_model(messages, info):
        return ModelResponse(parts=[next(steps)])

    agent = Agent(
        model=FunctionModel(fake_model),
        deps_type=Evidence,
        tools=[Tool(guarded_navigator, name="docs_navigator")],
        retries=2,
    )
    agent.output_validator(validate_output)
    result = await agent.run("Check content", deps=Evidence(docs))
    assert parse_judgment(result.output)["score"] == 1
    print(
        "Content-access checks passed: saved malformed path repaired, failure explicit, scoring gated."
    )


def usage(output):
    totals = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reported_cost_usd": 0.0,
        "requests": 0,
        "request_errors": 0,
        "responses": 0,
        "responses_without_cost": 0,
    }
    for path in (output / "traces").glob("*/*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "request":
                totals["requests"] += 1
            elif event["event"] == "request_error":
                totals["request_errors"] += 1
            elif event["event"] == "response":
                totals["responses"] += 1
                item = event["response"].get("usage") or {}
                if item.get("cost") is None:
                    totals["responses_without_cost"] += 1
                for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
                    totals[key] += item.get(key, 0)
                totals["reported_cost_usd"] += item.get("cost") or 0
    totals["unreturned_requests"] = (
        totals["requests"] - totals["responses"] - totals["request_errors"]
    )
    return totals


def verified_checkpoint(checkpoint, trace, docs):
    """Prove the saved answer followed real page reads delivered to the model."""
    if not checkpoint.exists() or not trace.exists():
        return False
    saved = read(checkpoint)
    if saved.get("status") != "completed":
        return False
    expected = {k: saved.get(k) for k in ("score", "reasoning", "evidence")}
    step, round_valid, delivered, verified = None, False, [], False
    sent_contents = set()
    try:
        for line in trace.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "attempt_started":
                step, round_valid, delivered = None, False, []
                sent_contents = set()
            elif event["event"] == "navigation":
                if step != event["step"]:
                    step, round_valid = event["step"], True
                valid = navigation_succeeded(event["results"])
                if valid != event["valid"]:
                    return False
                round_valid = round_valid and valid
                if valid:
                    for entry in event["results"]:
                        actual = navigate(docs, [entry["path"]], max_bytes=None)[0]
                        if actual.get("content") != entry["content"]:
                            return False
                        payload = json.dumps(entry["content"], indent=2)
                        content_hash = hashlib.sha256(payload.encode()).hexdigest()
                        if entry.get("cached"):
                            if (
                                content_hash not in sent_contents
                                or content_hash not in event["delivered"]
                            ):
                                return False
                        elif payload not in event["delivered"]:
                            return False
                        if "literal_search" in entry:
                            expected_search = literal_search(
                                entry["content"], entry["literal_search"]["query"]
                            )
                            if (
                                expected_search != entry["literal_search"]
                                or json.dumps(expected_search) not in event["delivered"]
                            ):
                                return False
                        sent_contents.add(content_hash)
                    delivered.append(event["delivered"])
            elif event["event"] == "agent_result":
                returns = [
                    part.get("content")
                    for message in event["messages"]
                    for part in message.get("parts", [])
                    if part.get("part_kind") == "tool-return"
                    and part.get("tool_name") == "docs_navigator"
                ]
                if parse_judgment(event["output"]) == expected:
                    verified = (
                        bool(delivered)
                        and round_valid
                        and last_navigation_batch_succeeded(event["messages"])
                        and all(text in returns for text in delivered)
                    )
    except (ValueError, TypeError, KeyError, IndexError):
        return False
    return verified


async def execute(args, manifest, runner, docs, tree):
    import logfire
    from judge import judge

    logfire.configure(console=False, send_to_logfire=False)
    instrument()
    leaves = judge.collect_leaf_requirements(runner["rubrics"]())
    semaphores = {m["openrouter_id"]: asyncio.Semaphore(args.workers) for m in manifest["models"]}
    total = sum(x["rerun"] for values in manifest["selection"].values() for x in values.values())
    done = 0

    async def one(name, leaf):
        nonlocal done
        stem = name.replace("/", "_")
        destination = args.output / "judgments" / stem / (leaf["path"] + ".json")
        trace = args.output / "traces" / stem / (leaf["path"] + ".jsonl")
        if verified_checkpoint(destination, trace, docs):
            done += 1
            return
        if destination.exists() and read(destination).get("status") == "completed":
            write(
                args.output
                / "invalidated"
                / stem
                / (leaf["path"] + "-" + sha(destination)[:12] + ".json"),
                read(destination),
            )
        trace.parent.mkdir(parents=True, exist_ok=True)
        async with semaphores[name]:
            token = ACTIVE.set((trace, name, leaf["path"]))
            try:
                for attempt in range(1, 3):
                    emit("attempt_started", attempt=attempt)
                    deps = Evidence(docs)
                    agent = Agent(
                        model=judge.get_llm(name),
                        deps_type=Evidence,
                        system_prompt=judge.EVALUATION_SYSTEM_PROMPT,
                        tools=[Tool(guarded_navigator, name="docs_navigator")],
                        retries=2,
                    )
                    agent.output_validator(validate_output)
                    prompt = (
                        "Evaluate this criteria against the documentation:\n\n"
                        f'Criteria: "{leaf["requirement"]}"\n\nDocumentation tree:\n```json\n'
                        f"{json.dumps(tree, indent=2)}\n```\n\n"
                        "First, you need to find the relevant documentation section that covers this criteria through `docs_navigator` tool.\n"
                        "Then, you need to evaluate if the criteria is mentioned. Respond with the exact JSON format specified."
                    )
                    try:
                        result = await agent.run(prompt, deps=deps)
                        judgment = parse_judgment(result.output)
                        assert deps.successful_paths and deps.round_valid
                        emit(
                            "agent_result",
                            output=result.output,
                            messages=json.loads(result.all_messages_json()),
                        )
                        write(
                            destination,
                            {
                                **judgment,
                                "status": "completed",
                                "model": name,
                                "criterion": leaf["path"],
                                "attempt": attempt,
                                "content_access": {
                                    "verified": True,
                                    "successful_paths": deps.successful_paths,
                                    "failed_paths": deps.failed_paths,
                                    "rounds": deps.rounds,
                                },
                            },
                        )
                        done += 1
                        print(
                            f"[{done}/{total}] {name} · {leaf['path']} · scored after verified content access",
                            flush=True,
                        )
                        return
                    except Exception as exc:  # noqa: BLE001 -- provider failures remain unscored
                        emit("agent_error", error_type=type(exc).__name__, attempt=attempt)
                        print(
                            f"{name} · {leaf['path']} · attempt {attempt} failed: {type(exc).__name__}",
                            flush=True,
                        )
                        write(
                            destination,
                            {
                                "status": "error",
                                "model": name,
                                "criterion": leaf["path"],
                                "error_type": type(exc).__name__,
                                "score": None,
                            },
                        )
                        if attempt < 2:
                            await asyncio.sleep(10)
            finally:
                ACTIVE.reset(token)

    await asyncio.gather(
        *(
            one(model["openrouter_id"], leaf)
            for model in manifest["models"]
            for leaf in leaves
            if manifest["selection"][model["openrouter_id"]][leaf["path"]]["rerun"]
        )
    )


def aggregate(args, manifest, runner):
    from judge import combine_evaluations, judge

    docs = read(args.source_run / "upstream/data/svelte/hydra/structured_docs.json")
    models = []
    reports = {}
    missing = []
    for model in manifest["models"]:
        name = model["openrouter_id"]
        stem = name.replace("/", "_")
        original = read(runner["RESULTS"] / (stem + ".json"))
        leaves = runner["leaves"](original)
        evaluations = {}
        for path, node in leaves.items():
            if manifest["selection"][name][path]["rerun"]:
                checkpoint = args.output / "judgments" / stem / (path + ".json")
                value = read(checkpoint) if checkpoint.exists() else {}
                if not verified_checkpoint(
                    checkpoint, args.output / "traces" / stem / (path + ".jsonl"), docs
                ):
                    missing.append([name, path])
                    continue
                evaluations[path] = {k: value[k] for k in ("score", "reasoning", "evidence")}
                evaluations[path]["tokens"] = {"input": 0, "output": 0}
            else:
                evaluations[path] = node["evaluation"]
        if len(evaluations) != 96:
            continue
        scored = judge.calculate_scores_bottom_up(runner["rubrics"](), evaluations)
        models.append(evaluations)
        target = args.output / "evaluation_results" / (stem + ".json")
        write(target, scored)
        assert runner["audit"](target)["status"] == "completed"
        rerun = sum(x["rerun"] for x in manifest["selection"][name].values())
        reports[name] = {
            "score_percent": 100
            * sum(n["score"] * n["weight"] for n in scored)
            / sum(n["weight"] for n in scored),
            "rejudged": rerun,
            "reused": 96 - rerun,
        }
    result = {
        "status": "incomplete" if missing else "completed",
        "missing": missing,
        "judges": reports,
        "added_usage": usage(args.output),
    }
    if not missing:
        combined_leaves = combine_evaluations.combine_leaf_evaluations(models, "average")
        combined = combine_evaluations.calculate_scores_bottom_up(
            runner["rubrics"](), combined_leaves
        )
        score = sum(n["score"] * n["weight"] for n in combined) / sum(n["weight"] for n in combined)
        std = combine_evaluations.combine_std_weighted(
            [n["std"] for n in combined], [n["weight"] for n in combined]
        )
        assert math.isclose(
            100 * score, sum(r["score_percent"] for r in reports.values()) / 3, abs_tol=1e-10
        )
        for path, node in runner["leaves"](combined).items():
            assert math.isclose(
                node["score"], sum(m[path]["score"] for m in models) / 3, abs_tol=1e-12
            )
        write(args.output / "evaluation_results/combined.json", combined)
        result.update(score_percent=100 * score, propagated_std_percent=100 * std, judgments=288)
    for relative, expected in manifest["source_artifacts_sha256"].items():
        assert sha(args.source_run / relative) == expected, f"Original artifact changed: {relative}"
    runner["verify"]()
    result["original_artifacts_unchanged"] = True
    result["elapsed_seconds"] = time.time() - manifest["started_at"]
    write(args.output / "result.json", result)
    print(json.dumps(result, indent=2))
    return not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=PROJECT / "runs/codewiki-svelte-paper")
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "runs/codewiki-svelte-paper-access-fixed"
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--upgrade-adapter",
        action="store_true",
        help="Record an adapter upgrade; reverify completed checkpoints before reuse",
    )
    args = parser.parse_args()
    args.source_run = args.source_run.resolve()
    args.output = args.output.resolve()
    if args.output == args.source_run or args.source_run in args.output.parents:
        raise ValueError("Use a separate output directory to preserve the old run")
    if not 1 <= args.workers <= 4:
        raise ValueError("Choose 1–4 workers per model")
    runner = runpy.run_path(str(args.source_run / "runner.py"))
    source_manifest = runner["verify"]()
    runner["configure_environment"]()
    runner["apply_provider_compatibility"]()
    sys.path.insert(0, str(args.source_run / "upstream/src"))
    docs = read(args.source_run / "upstream/data/svelte/hydra/structured_docs.json")
    tree = read(args.source_run / "upstream/data/svelte/hydra/docs_tree.json")
    manifest_path = args.output / "manifest.json"
    identity = {
        "source": str(args.source_run),
        "docs": sha(args.source_run / "upstream/data/svelte/hydra/structured_docs.json"),
        "navigation": sha(PROJECT / "src/hydra_agent/codewiki_navigation.py"),
        "judgment_parser": sha(PROJECT / "src/hydra_agent/codewiki_eval.py"),
        "script": sha(Path(__file__)),
    }
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["identity"] != identity:
            same_inputs = all(manifest["identity"][k] == identity[k] for k in ("source", "docs"))
            if not args.upgrade_adapter or not same_inputs or manifest["status"] == "running":
                raise ValueError(
                    "Inputs/adapter changed; use a new directory or explicitly upgrade a stopped run"
                )
            manifest.setdefault("adapter_history", []).append(manifest["identity"])
            manifest.setdefault("selection_history", []).append(manifest["selection"])
            manifest["selection"] = select_attempts(
                args.source_run, runner, source_manifest["models"], docs
            )
            manifest["identity"] = identity
            manifest["adapter_upgrade"] = (
                "Metadata requests include owning content; query/search hints use literal page search; repeat bodies reference prior delivered text. Original tool schemas determine rerun selection; retained checkpoints are reverified against delivered messages and source docs."
            )
            write(manifest_path, manifest)
    else:
        selection = select_attempts(args.source_run, runner, source_manifest["models"], docs)
        manifest = {
            "identity": identity,
            "models": source_manifest["models"],
            "selection": selection,
            "status": "prepared",
            "started_at": time.time(),
            "workers_per_model": args.workers,
            "protocol": "Original system/user rubric prompts, temperature 0, 36000 output/tool tokens, 300s requests, original scoring. Corrected navigation/errors/content guard; two validation retries and two attempts per affected criterion. Cached judgments retained only after old-tool replay; selection is independent of score.",
            "source_artifacts_sha256": {
                str(p.relative_to(args.source_run)): sha(p)
                for p in args.source_run.rglob("*")
                if p.is_file()
                and p.suffix in (".json", ".jsonl", ".md", ".py", ".txt")
                and "__pycache__" not in p.parts
            },
        }
        write(manifest_path, manifest)
    for name, values in manifest["selection"].items():
        print(
            f"{name}: {sum(x['rerun'] for x in values.values())} affected, {sum(not x['rerun'] for x in values.values())} retained",
            flush=True,
        )
    if args.check:
        asyncio.run(self_check(docs))
        return 0
    manifest["status"] = "running"
    manifest.setdefault("execution_started_at", time.time())
    write(manifest_path, manifest)
    try:
        asyncio.run(execute(args, manifest, runner, docs, tree))
    except KeyboardInterrupt:
        manifest.update(status="interrupted", interrupted_at=time.time())
        write(manifest_path, manifest)
        return 130
    complete = aggregate(args, manifest, runner)
    manifest.update(status="completed" if complete else "incomplete", finished_at=time.time())
    write(manifest_path, manifest)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
