"""Azure adapter for the pinned CodeWikiBench prompt and hierarchical scoring."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import Trace, function
from .bench_data import atomic_json, digest, read_json
from .codewiki_agent import AgentFailure, BudgetExceeded, UsageBudget
from .codewiki_data import EVALUATOR_REVISION, validate_rubrics
from .codewiki_navigation import documentation_paths, navigate, navigation_succeeded
from .model import AzureModel

LEGACY_ADAPTER = "azure_docs_navigator_strict_errors_v1"
PREVIOUS_ADAPTER = "docs_navigator_paths_v2"
RESPONSE_ADAPTER = "docs_navigator_responses_v3"
ADAPTER = "docs_navigator_content_access_v4"
DEFAULT_OUTPUT_TOKENS = 8192

NAVIGATOR = function(
    "docs_navigator",
    "Read generated documentation. Use JSON key/index paths, e.g. "
    '[["subpages", 0, "content", "markdown"]] for the first page. '
    'Exact page-title paths also work, e.g. [["Architecture"]]. '
    "Request relevant pages, not the entire repository at once.",
    {
        "paths": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                },
            },
        }
    },
)


def validate_judgment(value):
    if type(value.get("score")) is not int or value["score"] not in (0, 1):
        raise ValueError("Judge must return a binary integer score")
    for key in ("reasoning", "evidence"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError("Judge must provide reasoning and evidence")
    return {key: value[key] for key in ("score", "reasoning", "evidence")}


class JudgmentFormatError(AgentFailure):
    pass


def parse_judgment(text):
    """Accept one JSON object, including a single fenced object with surrounding prose.

    Never guess missing fields, repair escaping, or choose between multiple answers.
    Malformed responses must be corrected by the judge and validated again.
    """
    if not isinstance(text, str):
        raise JudgmentFormatError("Judge returned no JSON text")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Count fence lines, not backticks inside JSON evidence strings. A
        # preceding source-code example is not a second JSON judgment.
        blocks = re.findall(
            r"^[ \t]*```json[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
            text,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        if not blocks:
            blocks = re.findall(
                r"^[ \t]*```[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
                text,
                re.DOTALL | re.MULTILINE,
            )
        if len(blocks) != 1:
            raise JudgmentFormatError("Judge must return a single complete JSON object") from None
        try:
            value = json.loads(blocks[0])
        except json.JSONDecodeError as exc:
            raise JudgmentFormatError(
                f"Invalid judge JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from None
    if not isinstance(value, dict):
        raise JudgmentFormatError("Judge must return a JSON object")
    try:
        return validate_judgment(value)
    except ValueError as exc:
        raise JudgmentFormatError(str(exc)) from None


def cached_content_access(trace_path, saved):
    """Only reuse a score paired with an actual, successful navigation round."""
    if not trace_path.exists():
        return False
    navigated = False
    verified = False
    tools_left = 0
    try:
        for line in trace_path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "judge_error":
                navigated = False
                tools_left = 0
            elif event["event"] == "judge_model":
                choice = event["response"]["choices"][0]
                if choice.get("finish_reason") == "length":
                    continue
                message = choice["message"]
                calls = message.get("tool_calls", [])
                if calls:
                    navigated = True
                    tools_left = len(calls)
                else:
                    try:
                        judgment = parse_judgment(message.get("content"))
                    except JudgmentFormatError:
                        continue
                    if judgment == validate_judgment(saved):
                        verified = navigated and tools_left == 0
            elif event["event"] == "judge_tool":
                navigated = navigated and navigation_succeeded(event["result"])
                tools_left -= 1
    except (ValueError, TypeError, KeyError, IndexError):
        return False
    return verified


def judge_leaf(
    model,
    budget,
    trace,
    leaf,
    tree,
    docs,
    prompt,
    *,
    max_steps=8,
    max_output_tokens=DEFAULT_OUTPUT_TOKENS,
):
    if max_output_tokens <= 0:
        raise ValueError("Judge output token limit must be positive")
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f'Evaluate this criteria against the documentation:\n\nCriteria: "{leaf["requirement"]}"\n\n'
                f"Documentation tree:\n```json\n{json.dumps(tree, indent=2)}\n```\n\n"
                f"Valid documentation content paths: {json.dumps(documentation_paths(docs))}\n\n"
                "First, you need to find the relevant documentation section that covers this criteria "
                "through `docs_navigator` tool.\nThen, you need to evaluate if the criteria is mentioned. "
                "Respond with only the JSON object specified, with no preamble or search summary. "
                "Keep reasoning and evidence concise and escape quotes inside JSON strings."
            ),
        },
    ]
    navigated = False
    repairs = 0

    def repair(failure, step):
        nonlocal repairs
        if repairs >= 2 or step + 1 >= max_steps:
            raise failure
        repairs += 1
        trace.emit("judge_response_retry", path=leaf["path"], error=str(failure), repair=repairs)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The previous response could not be scored: {failure}. "
                    "Use the documentation already retrieved and return one complete JSON object "
                    "with criteria, score (integer 0 or 1), reasoning and evidence. "
                    "Keep reasoning and evidence concise; no prose outside JSON. "
                    "Escape quotes inside strings correctly."
                ),
            }
        )

    for step in range(max_steps):
        response = budget.complete(model, messages, [NAVIGATOR], max_tokens=max_output_tokens)
        trace.emit("judge_model", path=leaf["path"], step=step, response=response)
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            # Keep retrieved evidence but do not execute partial tool calls or score
            # truncated JSON, even if a prefix looks like a valid judgment.
            repair(
                AgentFailure(
                    f"Truncated judge response (response limit {max_output_tokens:,} tokens); "
                    "raise --max-judge-output-tokens if this persists"
                ),
                step,
            )
            continue
        message = choice["message"]
        messages.append({k: message[k] for k in ("role", "content", "tool_calls") if k in message})
        calls = message.get("tool_calls", [])
        if not calls:
            if not navigated:
                messages.append(
                    {
                        "role": "user",
                        "content": "Read actual page content with docs_navigator before scoring. Correct any failed paths using the available content paths; a failed read is not evidence of missing documentation.",
                    }
                )
                continue
            try:
                return parse_judgment(message.get("content"))
            except JudgmentFormatError as exc:
                repair(exc, step)
                continue
        if len(calls) > 10:
            raise AgentFailure("Judge exceeded tool-call budget")
        navigated = True
        for call in calls:
            try:
                if call["function"]["name"] != "docs_navigator":
                    raise ValueError("Unknown evaluation tool")
                args = json.loads(call["function"]["arguments"])
                result = navigate(docs, args["paths"])
                navigated = navigated and navigation_succeeded(result)
            except (ValueError, TypeError, KeyError) as exc:
                result = {"error": str(exc), "available_paths": documentation_paths(docs)}
                navigated = False
            trace.emit("judge_tool", path=leaf["path"], result=result)
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
            )
    raise AgentFailure("Judge did not finish within step budget")


def evaluate(
    root: Path,
    config,
    official: dict,
    *,
    workers=4,
    max_tokens=100_000,
    context_bytes=180_000,
    max_output_tokens=DEFAULT_OUTPUT_TOKENS,
    model_factory=AzureModel,
    report=None,
):
    if max_tokens < 0 or context_bytes < 0:
        raise ValueError("Judge budget limits must be nonnegative")
    if max_output_tokens <= 0:
        raise ValueError("Judge output token limit must be positive")
    report = report or (lambda *args, **kwargs: None)
    docs = read_json(root / "wiki" / "structured_docs.json")
    tree = read_json(root / "wiki" / "docs_tree.json")
    generation = read_json(root / "wiki" / "generation.json")
    if generation.get("status") != "completed" or generation["docs_digest"] != digest(docs):
        raise ValueError("Generation incomplete or generated documentation changed")
    rubrics = read_json(root / "evaluation" / "rubrics.json")
    rubrics = rubrics.get("rubrics", rubrics) if isinstance(rubrics, dict) else rubrics
    validate_rubrics(rubrics)
    leaves = official["collect_leaf_requirements"](rubrics)
    identity = {
        "docs": digest(docs),
        "tree": digest(tree),
        "rubrics": digest(rubrics),
        "judge_model": config.deployment,
        "endpoint": config.endpoint,
        "reasoning_effort": config.reasoning_effort,
        "upstream_commit": EVALUATOR_REVISION,
        "upstream_source": official["source_sha256"],
        "adapter": ADAPTER,
    }
    output = root / "evaluation" / "judgments"
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        previous = read_json(identity_path)
        if previous != identity:
            # These adapters fix navigation, budgets and response handling. Keep valid
            # earlier scores for identical docs/model/rubrics, recording their provenance
            # before updating the identity so an interrupted migration is safe to resume.
            if previous.get("adapter") not in (
                LEGACY_ADAPTER,
                PREVIOUS_ADAPTER,
                RESPONSE_ADAPTER,
            ) or previous != {
                **identity,
                "adapter": previous["adapter"],
            }:
                raise ValueError("Evaluation inputs/model changed; use a new output directory")
            for leaf in leaves:
                saved_path = output / (leaf["path"] + ".json")
                if saved_path.exists():
                    saved = read_json(saved_path)
                    saved.setdefault("adapter", previous["adapter"])
                    atomic_json(saved_path, saved)
    atomic_json(identity_path, identity)

    def one(leaf):
        destination = output / (leaf["path"] + ".json")
        if destination.exists():
            saved = read_json(destination)
            if saved.get("status") == "completed":
                validate_judgment(saved)
                if cached_content_access(output / (leaf["path"] + ".jsonl"), saved):
                    return leaf["path"], saved, True
                # Retain the former score for review; never silently trust a cached
                # score whose trace cannot prove successful page access.
                atomic_json(
                    output / "invalidated" / (leaf["path"] + "-" + digest(saved)[:12] + ".json"),
                    saved,
                )
        model = model_factory(config)
        trace = Trace(output / (leaf["path"] + ".jsonl"), secrets=(config.api_key,))
        budget = UsageBudget(output / (leaf["path"] + "-usage.json"), max_tokens, context_bytes)
        result = None
        try:
            for attempt in range(2):
                try:
                    result = {
                        **judge_leaf(
                            model,
                            budget,
                            trace,
                            leaf,
                            tree,
                            docs,
                            official["EVALUATION_SYSTEM_PROMPT"],
                            max_output_tokens=max_output_tokens,
                        ),
                        "status": "completed",
                        "attempt": attempt + 1,
                        "requirement": leaf["requirement"],
                        "tokens": budget.data,
                    }
                    break
                except Exception as exc:  # noqa: BLE001 -- record provider errors as unscored
                    # Local AgentFailure messages are controlled by this runner. Provider
                    # exception text can include credentials or full request bodies.
                    error = str(exc) if isinstance(exc, AgentFailure) else type(exc).__name__
                    if config.api_key:
                        error = error.replace(config.api_key, "[REDACTED]")
                    if isinstance(exc, BudgetExceeded):
                        error += (
                            "; raise --max-judge-tokens / --max-judge-context-bytes "
                            "(0 disables the corresponding cap)"
                        )
                    trace.emit(
                        "judge_error",
                        error_type=type(exc).__name__,
                        error=error,
                        attempt=attempt + 1,
                    )
                    result = {
                        "status": "error",
                        "score": None,
                        "error_type": type(exc).__name__,
                        "error": error,
                        "requirement": leaf["requirement"],
                        "tokens": budget.data,
                    }
                    if isinstance(exc, BudgetExceeded):
                        break
            result.update(
                adapter=ADAPTER,
                limits={
                    "max_tokens": max_tokens,
                    "context_bytes": context_bytes,
                    "max_steps": 8,
                    "max_output_tokens": max_output_tokens,
                },
            )
            atomic_json(destination, result)
            return leaf["path"], result, False
        finally:
            model.close()

    evaluations = {}
    reused = errors = 0
    report("evaluation_started", total=len(leaves), workers=workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, leaf) for leaf in leaves]
        for future in as_completed(futures):
            path, result, cached = future.result()
            evaluations[path] = result
            reused += int(cached)
            errors += int(result["status"] != "completed")
            report(
                "criterion_judged",
                path=path,
                status=result["status"],
                completed=len(evaluations),
                total=len(leaves),
                judged=len(evaluations) - errors,
                reused=reused,
                errors=errors,
                error_type=result.get("error_type"),
                error=result.get("error"),
            )
    failed = [path for path, value in evaluations.items() if value["status"] != "completed"]
    result = {
        "identity": identity,
        "status": "incomplete" if failed else "completed",
        "criteria": len(leaves),
        "judged": len(leaves) - len(failed),
        "error_paths": failed,
        "overall_score": None,
        "score_percent": None,
        "judge_tokens": sum(v["tokens"]["total_tokens"] for v in evaluations.values()),
        "reused": reused,
        "judgment_adapters": {
            adapter: sum(v.get("adapter") == adapter for v in evaluations.values())
            for adapter in sorted({v["adapter"] for v in evaluations.values()})
        },
        "limitations": [
            "Single judge model; adapted Azure tool runner, not the upstream CLI.",
            "Rubric coverage is not a factuality or graph-edge accuracy score.",
        ],
    }
    if not failed:
        scored = official["calculate_scores_bottom_up"](rubrics, evaluations)
        overall = sum(n["score"] * n["weight"] for n in scored) / sum(n["weight"] for n in scored)
        result.update(overall_score=overall, score_percent=round(100 * overall, 3))
        atomic_json(root / "evaluation" / "scored-rubrics.json", scored)
    atomic_json(root / "evaluation" / "result.json", result)
    return result
