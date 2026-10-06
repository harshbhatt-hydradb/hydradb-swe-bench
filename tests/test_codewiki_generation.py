import io
import json
import tarfile
from dataclasses import replace

import pytest
from test_codewiki import corpus as corpus_fixture

from hydra_agent.agent import Trace
from hydra_agent.bench_data import read_json
from hydra_agent.codewiki_agent import DocumentationAgent, UsageBudget, generate, validate_page
from hydra_agent.codewiki_data import code_corpus
from hydra_agent.indexing import Corpus, build_corpus

corpus = corpus_fixture


def article(corpus):
    return (
        "# Checkout\n"
        + "The checkout function returns 42. " * 15
        + f"[checkout]({corpus.scope.repository}/blob/{corpus.scope.base_commit}/src/shop.py#L1-L2)"
    )


def test_build_formats_are_available_only_to_documentation_policy(corpus, tmp_path):
    allowed = {
        "CMakeLists.txt": b"project(example)",
        "cmake/options.cmake": b'option(EXAMPLE_TESTS "Build tests" ON)',
        "cmake/config.cmake.in": b"@PACKAGE_INIT@",
        "app.sln": b"Microsoft Visual Studio Solution File",
        "src/app.csproj": b"<Project><PropertyGroup /></Project>",
        "src/Directory.Build.props": b"<Project />",
        "src/build.targets": b"<Project />",
        "src/messages.resx": b"<root />",
        "build.ps1": b"dotnet test",
    }
    excluded = {
        "tests/fixtures/fake.csproj": b"fixture",
        "build/generated.csproj": b"generated",
        ".env.ps1": b"SECRET",
        "../escape.csproj": b"escape",
        "binary.csproj": b"\x00binary",
        "nonutf8.sln": b"\xff",
    }
    archive = tmp_path / "build.tar"
    with tarfile.open(archive, "w") as tree:
        for path, content in {**allowed, **excluded}.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            tree.addfile(info, io.BytesIO(content))
    result = code_corpus(archive, corpus.scope)
    assert {s.path for s in result.sources} == set(allowed)
    assert {e["path"] for e in result.excluded} == set(excluded)
    # Expanding CodeWiki inputs must not silently change SWE-bench source IDs/policy.
    assert {s.path for s in build_corpus(archive, corpus.scope).sources} == {"CMakeLists.txt"}


def test_literal_search_paginates_matching_lines_and_stays_in_allowlist(corpus, tmp_path):
    source = replace(
        corpus.sources[0],
        path="src/catalog.cs",
        text="\n".join(f"public Entry{i} Literal.*Value" for i in range(83)),
    )
    agent = DocumentationAgent(None, None, Corpus(corpus.scope, [source], []), None, None)
    (tmp_path / "reference.json").write_text("Literal.*Value GOLD")
    args = {"query": "literal.*value", "path_contains": "src/", "offset": 0}
    first = agent.tool("search_source", args)
    second = agent.tool("search_source", {**args, "offset": first["next_offset"]})
    last = agent.tool("search_source", {**args, "offset": second["next_offset"]})
    assert first["total"] == second["total"] == last["total"] == 83
    assert [m["line"] for page in (first, second, last) for m in page["matches"]] == list(
        range(1, 84)
    )
    assert last["next_offset"] is None
    assert first["matches"][0]["url"].endswith("/src/catalog.cs#L1")
    assert not agent.session_reads  # Search snippets cannot stand in for verified source reads.
    assert agent.tool("search_source", {**args, "query": "literal.+value"})["total"] == 0
    assert agent.tool("search_source", {**args, "path_contains": "../"})["matches"] == []
    assert agent.tool("search_source", {**args, "query": "GOLD"})["matches"] == []
    assert agent.tool("search_source", {**args, "offset": 1000})["matches"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"offset": -1},
        {"offset": True},
        {"query": ""},
        {"query": " "},
        {"query": 12},
        {"query": "x" * 501},
        {"path_contains": []},
    ],
)
def test_search_rejects_invalid_arguments(corpus, overrides):
    agent = DocumentationAgent(None, None, corpus, None, None)
    with pytest.raises(ValueError):
        agent.tool(
            "search_source", {"query": "checkout", "path_contains": "", "offset": 0, **overrides}
        )


def test_search_shows_matches_deep_inside_long_lines(corpus):
    source = replace(corpus.sources[0], text="x" * 5000 + "needle" + "y" * 5000)
    agent = DocumentationAgent(None, None, Corpus(corpus.scope, [source], []), None, None)
    hit = agent.tool("search_source", {"query": "needle", "path_contains": "", "offset": 0})[
        "matches"
    ][0]
    assert "needle" in hit["text"] and len(hit["text"]) <= 700 and hit["truncated"]


class ReviewAgent:
    def __init__(self, corpus, root):
        self.corpus = corpus
        self.trace = Trace(root / "trace.jsonl")
        self.calls = []
        self.interrupt = True

    def run(self, task, *, label, **kwargs):
        self.calls.append(label)
        if label == "outline":
            return json.dumps(
                {"pages": [{"slug": "overview", "title": "Overview", "description": "Checkout"}]}
            )
        if label == "overview":
            return article(self.corpus)
        assert label == "review:overview"
        assert article(self.corpus) in task
        assert "checkout function returns 42" not in kwargs["retrieval_query"]
        if self.interrupt:
            self.interrupt = False
            raise TimeoutError("review provider unavailable")
        return article(self.corpus) + "\n\n## Reviewed behavior\nCheckout has no arguments."


def test_interrupted_review_reuses_draft_and_only_exports_reviewed_page(corpus, tmp_path):
    agent = ReviewAgent(corpus, tmp_path)
    root = tmp_path / "wiki"
    metadata = {"repo_name": "shop", "commit_id": corpus.scope.base_commit}
    with pytest.raises(TimeoutError):
        generate(agent, root, metadata, identity={"model": "fixture"})
    saved = read_json(root / "generation.json")
    assert saved["draft_pages"] and saved["completed_pages"] == {}
    assert not (root / "structured_docs.json").exists()
    assert not (root / "pages/overview.md").exists()
    state = generate(agent, root, metadata, identity={"model": "fixture"})
    assert agent.calls == ["outline", "overview", "review:overview", "review:overview"]
    assert state["status"] == "completed"
    docs = read_json(root / "structured_docs.json")
    assert "Reviewed behavior" in docs["subpages"][0]["content"]["markdown"]
    assert "Reviewed behavior" not in (root / "drafts/overview.md").read_text()
    generate(agent, root, metadata, identity={"model": "fixture"})
    assert len(agent.calls) == 4


@pytest.mark.parametrize("changed", ["draft", "outline", "review_setting", "page_limit"])
def test_review_resume_rejects_changed_artifacts_and_settings(corpus, tmp_path, changed):
    agent = ReviewAgent(corpus, tmp_path)
    root = tmp_path / "wiki"
    metadata = {"repo_name": "shop", "commit_id": corpus.scope.base_commit}
    with pytest.raises(TimeoutError):
        generate(agent, root, metadata, identity={})
    kwargs = {}
    if changed == "draft":
        (root / "drafts/overview.md").write_text("changed")
    elif changed == "outline":
        plan = read_json(root / "outline.json")
        plan["pages"][0]["description"] = "changed scope"
        (root / "outline.json").write_text(json.dumps(plan))
    elif changed == "review_setting":
        kwargs["review_pages"] = False
    else:
        kwargs["max_pages"] = 10
    with pytest.raises(ValueError, match="changed"):
        generate(agent, root, metadata, identity={}, **kwargs)
    assert len(agent.calls) == 3


def test_page_validation_repairs_bad_citations_inside_agent_session(corpus, tmp_path):
    queries = []
    valid = article(corpus)
    invalid = valid.replace("#L1-L2", "#L1-L999")

    class Memory:
        def search(self, query, **kwargs):
            queries.append(query)
            return []

    class Model:
        def complete(self, messages, tools, **kwargs):
            assert "search_source" in {t["function"]["name"] for t in tools}
            retry = messages[-1]["role"] == "user"
            if retry:
                assert "Correct invalid source links" in messages[-1]["content"]
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": valid if retry else invalid},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 20},
            }

    agent = DocumentationAgent(
        Model(),
        Memory(),
        corpus,
        UsageBudget(tmp_path / "usage.json", 1000000),
        Trace(tmp_path / "trace.jsonl"),
        max_steps=3,
    )
    assert (
        agent.run(
            "Review this draft: " + invalid,
            label="review:overview",
            retrieval_query="How does checkout work?",
            output_validator=lambda text: validate_page(text, corpus),
        )
        == valid
    )
    assert queries == ["How does checkout work?"]
    assert agent.budget.data["calls"] == 2


@pytest.mark.parametrize("replacement", ["#L1-L999", "/src/missing.py#L1", "/blob/main/"])
def test_page_validation_rejects_invalid_citation(corpus, replacement):
    text = article(corpus)
    if replacement.startswith("#"):
        text = text.replace("#L1-L2", replacement)
    elif replacement.startswith("/src"):
        text = text.replace("/src/shop.py#L1-L2", replacement)
    else:
        text = text.replace(f"/blob/{corpus.scope.base_commit}/", replacement)
    with pytest.raises(ValueError, match="source link"):
        validate_page(text, corpus)
