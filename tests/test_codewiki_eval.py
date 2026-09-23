import copy
import json
from dataclasses import replace

import pytest

from hydra_agent.agent import Trace
from hydra_agent.bench_data import atomic_json, digest, read_json
from hydra_agent.codewiki_agent import AgentFailure, BudgetExceeded, UsageBudget
from hydra_agent.codewiki_eval import (
    ADAPTER,
    LEGACY_ADAPTER,
    PREVIOUS_ADAPTER,
    RESPONSE_ADAPTER,
    JudgmentFormatError,
    evaluate,
    judge_leaf,
    navigate,
    parse_judgment,
)
from hydra_agent.config import AzureConfig


@pytest.fixture
def docs():
    return {
        "title": "svelte",
        "content": {},
        "subpages": [
            {
                "title": "Architecture and Public Component API",
                "content": {"markdown": "Components are compiled."},
                "subpages": [
                    {
                        "title": "Details",
                        "content": {"markdown": "Nested evidence."},
                        "subpages": [],
                    }
                ],
            },
            {"title": "Reactivity", "content": {"markdown": "Effects react."}, "subpages": []},
        ],
    }


@pytest.mark.parametrize(
    "path",
    [
        ["subpages", 0, "content", "markdown"],
        ["subpages", "0", "content", "markdown"],
        ["Architecture and Public Component API", "content", "markdown"],
        ["svelte", "Architecture and Public Component API", "content", "markdown"],
        [0, "content", "markdown"],
    ],
)
def test_navigator_resolves_saved_judge_path_styles(docs, path):
    assert navigate(docs, [path])[0]["content"] == "Components are compiled."


def test_navigator_nested_titles_and_ambiguous_titles(docs):
    path = ["svelte", "Architecture and Public Component API", "Details"]
    assert navigate(docs, [path])[0]["content"]["content"]["markdown"] == "Nested evidence."
    docs["subpages"].append(copy.deepcopy(docs["subpages"][0]))
    error = navigate(docs, [["Architecture and Public Component API"]])[0]
    assert "Ambiguous" in error["error"]
    assert error["available_paths"][0]["path"] == ["subpages", 0, "content"]
    assert "content" in navigate(docs, [["subpages", 0]])[0]


@pytest.mark.parametrize("path", [["missing"], ["subpages", -1], [False], [0.5]])
def test_navigation_errors_include_valid_paths(docs, path):
    result = navigate(docs, [path])[0]
    assert "error" in result and "content" not in result
    assert result["available_paths"][0]["title"] == "Architecture and Public Component API"


def test_saved_gpt_oss_extra_path_wrapper_reads_the_page(docs):
    result = navigate(docs, [[["subpages", 0, "content", "markdown"]]])[0]
    assert result["content"] == "Components are compiled."
    assert result["resolved_path"] == ["subpages", 0, "content", "markdown"]


def test_ambiguous_nested_paths_are_not_flattened(docs):
    result = navigate(docs, [[["subpages", 0], ["subpages", 1]]])[0]
    assert "error" in result and "content" not in result


@pytest.mark.parametrize("content", [None, "", "  ", {}, [], "<detail_content>"])
def test_empty_or_placeholder_content_is_an_explicit_read_error(docs, content):
    docs["subpages"][0]["content"]["markdown"] = content
    result = navigate(docs, [["subpages", 0, "content", "markdown"]])[0]
    assert "error" in result and "content" not in result
    assert result["available_paths"]


def test_metadata_request_includes_owning_page_content(docs):
    result = navigate(docs, [["subpages", 0, "title"]])[0]
    assert result["content"]["requested_metadata"] == "Architecture and Public Component API"
    assert result["content"]["content"]["markdown"] == "Components are compiled."
    assert result["resolved_path"] == ["subpages", 0, "content"]


def test_metadata_without_page_body_does_not_count_as_a_read(docs):
    docs["subpages"][0]["content"] = {}
    assert "error" in navigate(docs, [["subpages", 0, "title"]])[0]


def test_oversize_navigation_can_recover_by_selecting_one_page(docs):
    docs["subpages"][0]["content"]["markdown"] = "a" * 60_000
    docs["subpages"][1]["content"]["markdown"] = "b" * 60_000
    result = navigate(docs, [["svelte"]])
    assert "error" in result and result["available_paths"]
    assert "content" in navigate(docs, [["subpages", 0]])[0]


def response(*, paths=None):
    message = {"role": "assistant", "content": None}
    if paths is not None:
        message["tool_calls"] = [
            {
                "id": "read",
                "type": "function",
                "function": {"name": "docs_navigator", "arguments": json.dumps({"paths": paths})},
            }
        ]
    else:
        message["content"] = json.dumps(
            {"score": 1, "reasoning": "Explained", "evidence": "Components"}
        )
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if paths else "stop"}],
        "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
    }


class ReadingModel:
    def __init__(self, config=None):
        self.calls = 0
        self.closed = False

    def complete(self, messages, tools, **kwargs):
        self.calls += 1
        if self.calls == 1:
            assert '"subpages", 0, "content"' in messages[1]["content"]
            return response(paths=[["Architecture and Public Component API"]])
        assert "Components are compiled." in messages[-1]["content"]
        return response()

    def close(self):
        self.closed = True


def test_unlimited_judge_budget_retains_prior_usage_and_reads_titles(tmp_path, docs):
    usage = tmp_path / "usage.json"
    atomic_json(
        usage,
        {"prompt_tokens": 110_000, "completion_tokens": 1000, "total_tokens": 111_000, "calls": 9},
    )
    budget = UsageBudget(usage, 0, context_bytes=0)
    model = ReadingModel()
    result = judge_leaf(
        model,
        budget,
        Trace(tmp_path / "trace.jsonl"),
        {"path": "0", "requirement": "Components"},
        docs,
        docs,
        "judge",
    )
    assert result["score"] == 1 and model.calls == 2
    assert read_json(usage)["total_tokens"] == 111_200
    assert read_json(usage)["calls"] == 11


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"limit": 1}, "Model token budget"),
        ({"limit": 0, "context_bytes": 1}, "Context byte budget"),
    ],
)
def test_budget_guards_remain_independent_and_do_not_bill_unmade_calls(tmp_path, limits, reason):
    model = ReadingModel()
    budget = UsageBudget(tmp_path / "usage.json", **limits)
    with pytest.raises(BudgetExceeded, match=reason):
        budget.complete(model, [], [])
    assert model.calls == 0 and budget.data["total_tokens"] == 0
    assert not budget.path.exists()


def setup_evaluation(root, docs):
    atomic_json(root / "wiki/structured_docs.json", docs)
    atomic_json(root / "wiki/docs_tree.json", docs)
    atomic_json(root / "wiki/generation.json", {"status": "completed", "docs_digest": digest(docs)})
    atomic_json(root / "evaluation/rubrics.json", [{"requirements": "Components", "weight": 1}])
    return {
        "source_sha256": "test",
        "EVALUATION_SYSTEM_PROMPT": "judge",
        "collect_leaf_requirements": lambda r: [{"path": "0", "requirement": "Components"}],
        "calculate_scores_bottom_up": lambda r, e: [{"score": e["0"]["score"], "weight": 1}],
    }


def test_evaluation_resumes_exhausted_budget_and_preserves_successful_judgments(tmp_path, docs):
    official = setup_evaluation(tmp_path, docs)
    config = AzureConfig("https://example.com", "judge", "secret-key")
    events = []
    result = evaluate(
        tmp_path,
        config,
        official,
        max_tokens=1,
        model_factory=ReadingModel,
        report=lambda kind, **data: events.append({"event": kind, **data}),
    )
    assert result["status"] == "incomplete" and result["overall_score"] is None
    assert "conservatively reserved" in events[-1]["error"]
    output = tmp_path / "evaluation/judgments"
    assert (output / "0.jsonl").read_text().count('"event": "judge_error"') == 1
    atomic_json(
        output / "0-usage.json",
        {"prompt_tokens": 110_000, "completion_tokens": 1000, "total_tokens": 111_000, "calls": 9},
    )
    result = evaluate(
        tmp_path, config, official, max_tokens=0, context_bytes=0, model_factory=ReadingModel
    )
    assert result["status"] == "completed" and result["score_percent"] == 100
    assert result["judge_tokens"] == 111_200
    assert read_json(output / "0.json")["limits"]["max_tokens"] == 0
    saved = (output / "0.json").read_bytes()

    def no_model(config):
        pytest.fail("Completed judgments must not call a model again")

    result = evaluate(tmp_path, config, official, model_factory=no_model)
    assert result["reused"] == 1 and result["judge_tokens"] == 111_200
    assert (output / "0.json").read_bytes() == saved


@pytest.mark.parametrize("previous_adapter", [LEGACY_ADAPTER, PREVIOUS_ADAPTER, RESPONSE_ADAPTER])
def test_legacy_resume_records_provenance_and_rejects_changed_model(
    tmp_path, docs, previous_adapter
):
    official = setup_evaluation(tmp_path, docs)
    config = AzureConfig("https://example.com", "judge", "secret-key")
    evaluate(tmp_path, config, official, model_factory=ReadingModel)
    output = tmp_path / "evaluation/judgments"
    identity = read_json(output / "identity.json")
    identity["adapter"] = previous_adapter
    atomic_json(output / "identity.json", identity)
    saved = read_json(output / "0.json")
    saved.pop("adapter")
    atomic_json(output / "0.json", saved)

    def no_model(config):
        pytest.fail("Legacy valid scores must be reused")

    with pytest.raises(ValueError, match="inputs/model changed"):
        evaluate(
            tmp_path,
            replace(config, deployment="different-judge"),
            official,
            model_factory=no_model,
        )
    assert read_json(output / "identity.json") == identity
    result = evaluate(tmp_path, config, official, model_factory=no_model)
    assert result["reused"] == 1
    assert result["judgment_adapters"] == {previous_adapter: 1}
    assert read_json(output / "0.json") == {**saved, "adapter": previous_adapter}
    assert read_json(output / "identity.json")["adapter"] == ADAPTER
    assert evaluate(tmp_path, config, official, model_factory=no_model)["reused"] == 1


def test_provider_exception_payload_is_not_logged(tmp_path, docs):
    official = setup_evaluation(tmp_path, docs)

    class BrokenModel(ReadingModel):
        def complete(self, *args, **kwargs):
            raise RuntimeError("PRIVATE REQUEST and secret-key")

    result = evaluate(
        tmp_path,
        AzureConfig("https://example.com", "judge", "secret-key"),
        official,
        model_factory=BrokenModel,
    )
    assert result["status"] == "incomplete"
    output = tmp_path / "evaluation/judgments"
    assert read_json(output / "0.json")["error"] == "RuntimeError"
    assert "PRIVATE" not in (output / "0.jsonl").read_text()


def test_judge_cannot_score_without_successful_navigation(tmp_path, docs):
    class GuessingModel(ReadingModel):
        def complete(self, *args, **kwargs):
            return response()

    with pytest.raises(AgentFailure, match="step budget"):
        judge_leaf(
            GuessingModel(),
            UsageBudget(tmp_path / "usage.json", 0),
            Trace(tmp_path / "trace.jsonl"),
            {"path": "0", "requirement": "Components"},
            docs,
            docs,
            "judge",
            max_steps=2,
        )


@pytest.mark.parametrize(
    "wrapper", ["{}", "```json\n{}\n```", "My evaluation:\n```json\n{}\n```\nDone."]
)
def test_judge_accepts_complete_json_with_markdown_and_prose(wrapper):
    judgment = {
        "score": 0,
        "reasoning": 'No description of "events".',
        "evidence": "Only overview.",
    }
    assert parse_judgment(wrapper.format(json.dumps(judgment))) == judgment


def test_saved_judgment_with_fenced_source_evidence_remains_valid():
    judgment = {"score": 1, "reasoning": "Explained", "evidence": "```svelte\n<p>Example</p>\n```"}
    text = (
        "Example:\n```javascript\nlet {a} = value;\n```\n\n```json\n"
        + json.dumps(judgment)
        + "\n```"
    )
    assert parse_judgment(text) == judgment


@pytest.mark.parametrize(
    "content",
    [
        None,
        '```json\n{"score": 1, "reasoning": "unfinished\n```',
        '{"score": true, "reasoning": "reason", "evidence": "section"}',
        '{"score": 2, "reasoning": "reason", "evidence": "section"}',
        '{"score": 1, "reasoning": "reason"}',
        '{"score": 1, "reasoning": "reason", "evidence": ""}',
        '[{"score": 1}]',
        '```json\n{"score": 1}\n```\n```json\n{"score": 0}\n```',
    ],
)
def test_judge_does_not_silently_repair_or_choose_invalid_answers(content):
    with pytest.raises(JudgmentFormatError):
        parse_judgment(content)


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), **kwargs})
        return next(self.responses)


def test_judge_cannot_score_after_a_mixed_failed_read_until_it_recovers(tmp_path, docs):
    model = SequenceModel(
        [
            response(paths=[["subpages", 0], ["missing"]]),
            response(),
            response(paths=[[["subpages", 0, "content", "markdown"]]]),
            response(),
        ]
    )
    result = judge_leaf(
        model,
        UsageBudget(tmp_path / "usage.json", 0),
        Trace(tmp_path / "trace.jsonl"),
        {"path": "0", "requirement": "Components"},
        docs,
        docs,
        "judge",
    )
    assert result["score"] == 1 and len(model.requests) == 4
    assert "failed read is not evidence" in model.requests[2]["messages"][-1]["content"]


def test_cached_judgment_without_content_access_is_rejudged_and_archived(tmp_path, docs):
    official = setup_evaluation(tmp_path, docs)
    config = AzureConfig("https://example.com", "judge", "secret-key")
    evaluate(tmp_path, config, official, model_factory=ReadingModel)
    output = tmp_path / "evaluation/judgments"
    saved = read_json(output / "0.json")
    trace = output / "0.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    for event in events:
        if event["event"] == "judge_tool":
            event["result"] = [{"path": ["subpages", 0], "content": None}]
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    result = evaluate(tmp_path, config, official, model_factory=ReadingModel)
    assert result["status"] == "completed" and result["reused"] == 0
    assert [read_json(p) for p in (output / "invalidated").glob("*.json")] == [saved]
    assert evaluate(tmp_path, config, official, model_factory=ReadingModel)["reused"] == 1


@pytest.mark.parametrize("failure", ["malformed", "truncated", "truncated_tool"])
def test_judge_corrects_answers_without_discarding_evidence(tmp_path, docs, failure):
    broken = response()
    if failure == "malformed":
        broken["choices"][0]["message"]["content"] = '{"score": 1, "evidence": "bad "quotes""}'
    elif failure == "truncated":
        # Even a complete-looking object must not be accepted on a length finish.
        broken["choices"][0]["finish_reason"] = "length"
    else:
        broken = response(paths=[["Reactivity"]])
        broken["choices"][0]["message"]["tool_calls"][0]["id"] = "partial-tool"
        broken["choices"][0]["finish_reason"] = "length"
    model = SequenceModel([response(paths=[["subpages", 0]]), broken, response()])
    budget = UsageBudget(tmp_path / "usage.json", 0, context_bytes=0)
    result = judge_leaf(
        model,
        budget,
        Trace(tmp_path / "trace.jsonl"),
        {"path": "0", "requirement": "Components"},
        docs,
        docs,
        "judge",
    )
    assert result["score"] == 1 and len(model.requests) == 3
    assert all(request["max_tokens"] == 8192 for request in model.requests)
    final_messages = model.requests[-1]["messages"]
    assert final_messages[-1]["role"] == "user"
    assert "could not be scored" in final_messages[-1]["content"]
    tools = [message for message in final_messages if message["role"] == "tool"]
    assert len(tools) == 1 and "Components are compiled." in tools[0]["content"]
    assert "partial-tool" not in json.dumps(final_messages)
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "judge_response_retry" for e in events) == 1
    assert budget.data["total_tokens"] == 300


@pytest.mark.parametrize("steps,expected_calls", [(2, 2), (8, 4)])
def test_bad_json_corrections_are_bounded_and_never_scored(tmp_path, docs, steps, expected_calls):
    broken = response()
    broken["choices"][0]["message"]["content"] = "not JSON"
    model = SequenceModel([response(paths=[["subpages", 0]]), *([broken] * 3)])
    with pytest.raises(JudgmentFormatError, match="single complete JSON"):
        judge_leaf(
            model,
            UsageBudget(tmp_path / "usage.json", 0),
            Trace(tmp_path / "trace.jsonl"),
            {"path": "0", "requirement": "Components"},
            docs,
            docs,
            "judge",
            max_steps=steps,
            max_output_tokens=16384,
        )
    assert len(model.requests) == expected_calls
    assert all(r["max_tokens"] == 16384 for r in model.requests)
