"""One source-grounded repository survey before documentation is written.

The survey names candidate modules. It does not walk them in later sessions, and
it is not a parser-proven call graph. Evidence locations must have actually been
delivered by read_file in the survey session.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from .bench_data import atomic_json, digest, read_json
from .codewiki_agent import json_object

RELATIONS = ("module", "imports", "calls", "uses", "tests", "related")
# The initial traversal implementation used identical evidence/checkpoint semantics,
# but reported validation errors singly and stopped after two corrections.
VALIDATION_V1_CODE = "b304c95448ade3f4d47f6300bbca15fab244d45fd3e4cc1d968bef7fc384f076"


@dataclass(frozen=True)
class ExplorationConfig:
    # Depth and module count stay in the checkpoint fingerprint. They do not
    # schedule sessions. max_branches caps modules named by the one survey.
    max_depth: int = 3
    max_modules: int = 24
    max_branches: int = 6

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"Exploration {key} must be a positive integer")


def _string(value, label, maximum=1800):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be nonempty text of at most {maximum} characters")
    return value.strip()


def _evidence(value, agent):
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("Provide 1–8 evidence ranges from files read in this session")
    results = []
    for ref in value:
        if not isinstance(ref, dict):
            raise TypeError("Evidence must contain path, start_line and end_line")
        path = ref.get("path")
        source = agent.sources.get(path) if isinstance(path, str) else None
        start, end = ref.get("start_line"), ref.get("end_line")
        if source is None or type(start) is not int or type(end) is not int:
            raise ValueError("Evidence must reference allowed source paths and integer lines")
        lines = source.text.splitlines()
        if not 1 <= start <= end <= len(lines) or end - start >= 20:
            raise ValueError("Evidence ranges must contain 1–20 existing source lines")
        numbered = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start - 1, end))
        reads = [r for r in agent.session_reads if r["path"] == source.path]
        if not any(numbered in r["text"] for r in reads):
            raise ValueError(
                f"Evidence {source.path}:{start}-{end} was not fully read in this session"
            )
        results.append(
            {
                "path": source.path,
                "start_line": start,
                "end_line": end,
                "source_sha256": source.sha256,
                "excerpt": "\n".join(lines[start - 1 : end])[:2000],
                "validation": "source_location_and_read_checked; relationship semantics remain inferred",
            }
        )
    return results


def validate_discovery(text, agent, node, config):
    value = json_object(text)
    errors = []

    def check(label, function, *args):
        try:
            return function(*args)
        except (ValueError, TypeError) as exc:
            errors.append(f"{label}: {exc}")
            return None

    summary = check("summary", _string, value.get("summary"), "summary")
    evidence = check("evidence", _evidence, value.get("evidence"), agent)
    if (
        evidence
        and node["path"] is not None
        and not any(e["path"] == node["path"] for e in evidence)
    ):
        errors.append(
            f"evidence: The summary must cite the module's own source file {node['path']}"
        )
    questions = value.get("open_questions")
    if not isinstance(questions, list) or len(questions) > 8:
        errors.append("open_questions must be a list of at most eight questions")
        questions = []
    questions = [
        check(f"open_questions[{i}]", _string, q, "open question", 500)
        for i, q in enumerate(questions)
    ]
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, list) or len(dependencies) > config.max_branches:
        errors.append(f"dependencies must contain at most {config.max_branches} entries")
        dependencies = []
    checked = []
    for i, dep in enumerate(dependencies):
        label = f"dependencies[{i}]"
        if not isinstance(dep, dict):
            errors.append(f"{label}: Each dependency must be an object")
            continue
        path = dep.get("path")
        if not isinstance(path, str) or path not in agent.sources:
            errors.append(f"{label}.path: Dependency path is outside the allowed source set")
        symbol = dep.get("symbol", "")
        if not isinstance(symbol, str) or len(symbol) > 160:
            errors.append(f"{label}.symbol: must be a string of at most 160 characters")
        else:
            symbol = symbol.strip()
            if (
                symbol
                and isinstance(path, str)
                and path in agent.sources
                and symbol not in agent.sources[path].text
            ):
                errors.append(f"{label}.symbol: Symbol {symbol!r} is absent from {path}")
        relation = dep.get("relation")
        if relation not in RELATIONS:
            errors.append(f"{label}.relation: Use relation {'/'.join(RELATIONS)}")
        refs = check(f"{label}.evidence", _evidence, dep.get("evidence"), agent)
        origin = node["path"] if node["path"] is not None else path
        if refs and not any(e["path"] == origin for e in refs):
            errors.append(
                f"{label}.evidence: A dependency must cite its originating module {origin}"
                if node["path"] is not None
                else f"{label}.evidence: A discovered root module must cite its own source {origin}"
            )
        question = check(
            f"{label}.question", _string, dep.get("question"), "dependency question", 500
        )
        checked.append(
            {
                "path": path,
                "symbol": symbol,
                "relation": relation,
                "question": question,
                "evidence": refs,
            }
        )
    if errors:
        raise ValueError("Fix these discovery fields together:\n- " + "\n- ".join(errors))
    return {
        "summary": summary,
        "evidence": evidence,
        "dependencies": checked,
        "open_questions": questions,
    }


def _node_id(path, symbol=""):
    return digest({"path": path, "symbol": symbol})[:24]


def _save(path, state):
    # A digest also detects accidental changes to already completed notes/frontiers.
    atomic_json(path, {"state": state, "digest": digest(state)})


def _prompt(agent, config):
    groups = Counter(str(PurePosixPath(p).parent) for p in agent.sources)
    inventory = sorted(groups.items(), key=lambda p: (-p[1], p[0]))[:50]
    return (
        "Discover distinct module entry points across this repository, including public APIs, "
        "runtime/compiler subsystems, build/configuration and testing where present. "
        f"Allowed-source directory inventory (first 50): {inventory}. "
        "Use list_files for the remaining paths. Read each selected module before citing it. "
        "Dependency labels are hypotheses backed by cited source; do not invent call edges. "
        "Name the modules a later documentation session should read. They will not be explored "
        "as separate sessions. Leave unresolved questions explicit. "
        "Return one JSON object with a concise summary (aim for 1200 characters; hard limit 1800), "
        f"1–8 evidence ranges, up to {config.max_branches} dependencies and up to 8 open_questions. "
        "All evidence must be 1–20 source lines actually read in this session. "
        f"relation must be exactly one of: {', '.join(RELATIONS)}. "
        "For this repository survey, each dependency must cite its own target module. "
        "If you cannot support a dependency from the required source, omit it and record an open question; "
        "do not attach unrelated citations merely to pass validation. "
        "Use exact allowed paths. symbol may be empty or an exact source symbol. Schema:\n"
        '{"summary":"...","evidence":[{"path":"src/example.py","start_line":1,"end_line":4}],'
        '"dependencies":[{"path":"src/helper.py","symbol":"helper","relation":"module",'
        '"question":"How does this module implement the behavior?",'
        '"evidence":[{"path":"src/helper.py","start_line":1,"end_line":2}]}],'
        '"open_questions":[]}'
    )


def _fingerprint(agent, identity, config):
    return digest(
        {
            "generation": identity,
            "corpus": agent.corpus.manifest(),
            "config": asdict(config),
            "version": 2,
        }
    )


def _upgrade_validation_checkpoint(agent, root, saved, identity, config):
    """Allow only the known validation fix, before any outline/pages have been written."""
    previous_identity = {**identity, "code": VALIDATION_V1_CODE}
    state = saved["state"]
    if (
        identity.get("code") in (None, VALIDATION_V1_CODE)
        or state["fingerprint"] != _fingerprint(agent, previous_identity, config)
        or (root.parent / "outline.json").exists()
        or any((root.parent / "pages").glob("*.md"))
    ):
        raise ValueError("Exploration settings or source changed; use a new output directory")
    backup = root / f"state-before-validation-fix-{state['fingerprint']}.json"
    if backup.exists():
        if read_json(backup) != saved:
            raise ValueError("Exploration migration backup changed")
    else:
        atomic_json(backup, saved)
    state.setdefault("code_upgrades", []).append(
        {
            "reason": "aggregate_validation_errors_and_use_remaining_steps",
            "previous_fingerprint": state["fingerprint"],
            "previous_code": VALIDATION_V1_CODE,
            "code": identity["code"],
            "retained_notes": len(state["completed_order"]),
        }
    )
    state["fingerprint"] = _fingerprint(agent, identity, config)
    _save(root / "state.json", state)
    agent.trace.emit("exploration_upgraded", completed=len(state["completed_order"]))


def explore(agent, root, identity, config):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "state.json"
    fingerprint = _fingerprint(agent, identity, config)
    if path.exists():
        saved = read_json(path)
        state = saved["state"]
        if saved["digest"] != digest(state):
            raise ValueError("Exploration checkpoint changed")
        if state["fingerprint"] != fingerprint:
            _upgrade_validation_checkpoint(agent, root, saved, identity, config)
        if state["status"] in ("completed", "bounded"):
            agent.exploration = state
            agent.trace.emit("exploration_reused", completed=len(state["completed_order"]))
            return state
    else:
        node_id = _node_id(None)
        state = {
            "fingerprint": fingerprint,
            "config": asdict(config),
            "status": "running",
            "nodes": {
                node_id: {
                    "id": node_id,
                    "path": None,
                    "symbol": "",
                    "question": "Discover repository modules",
                    "depth": 0,
                    "status": "pending",
                }
            },
            "frontier": [node_id],
            "completed_order": [],
            "edges": [],
            "deferred": [],
            "stop_reasons": [],
            "evidence": {},
        }
    agent.exploration = state
    _save(path, state)
    root_id = _node_id(None)
    survey = state["nodes"][root_id]
    if survey["status"] != "completed":
        agent.trace.emit(
            "exploration_started",
            path="repository",
            depth=0,
            completed=len(state["completed_order"]),
            pending=1,
        )
        try:
            raw = agent.run(
                _prompt(agent, config),
                label="explore:repository",
                max_tokens=5000,
                output_validator=lambda text: validate_discovery(text, agent, survey, config),
            )
            result = validate_discovery(raw, agent, survey, config)
        except BaseException:
            state["status"] = "interrupted"
            _save(path, state)
            raise
        survey.update(status="completed", result=result)
        state["completed_order"] = [root_id]
        state["frontier"] = []
        _save(path, state)
        agent.trace.emit(
            "exploration_completed",
            path="repository",
            depth=0,
            completed=1,
            pending=0,
            deferred=0,
        )
    named = []
    seen = set()
    state["edges"] = []
    state["evidence"] = {}
    result = survey["result"]
    evidence = result["evidence"] + [
        entry for dep in result["dependencies"] for entry in dep["evidence"]
    ]
    for entry in evidence:
        key = digest({k: entry[k] for k in ("path", "source_sha256", "start_line", "end_line")})
        state["evidence"][key] = entry
    for dep in result["dependencies"]:
        state["edges"].append(
            {
                "from": root_id,
                "to": _node_id(dep["path"], dep["symbol"]),
                **dep,
                "verification": "inferred_from_read_source",
            }
        )
        if dep["path"] not in seen:
            seen.add(dep["path"])
            named.append(dep)
    state["frontier"] = []
    state["deferred"] = []
    state["stop_reasons"] = ["unresolved_questions"] if result["open_questions"] else []
    state["status"] = "bounded" if state["stop_reasons"] else "completed"
    state["coverage"] = {
        "topics_explored": len(named),
        "cited_source_files": len({e["path"] for e in state["evidence"].values()}),
        "eligible_source_files": len(agent.sources),
        "unique_evidence_ranges": len(state["evidence"]),
        "unresolved_questions": len(result["open_questions"]),
    }
    lines = [
        "# Repository survey",
        "",
        f"Status: {state['status']}",
        f"Stop reasons: {', '.join(state['stop_reasons']) or 'survey finished'}",
        f"Source files cited: {state['coverage']['cited_source_files']}/{len(agent.sources)}",
        "",
        "Named modules are a reading list for documentation. They were not explored as separate",
        "sessions, and the dependency labels are inferred from read source.",
        "",
        "## Repository",
        "",
        result["summary"],
        "",
    ]
    lines.extend(f"- Open question: {q}" for q in result["open_questions"])
    lines.extend(["", "## Modules to read while writing", ""])
    lines.extend(
        f"- {dep['path']} {dep['symbol']}: {dep['relation']}. {dep['question']}" for dep in named
    )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    _save(path, state)
    agent.trace.emit(
        "exploration_finished",
        status=state["status"],
        modules=len(named),
        deferred=0,
        stop_reasons=state["stop_reasons"],
    )
    return state


def module_notes(state, contains, offset):
    if not isinstance(contains, str) or type(offset) is not int or offset < 0:
        raise ValueError("Use a text filter and nonnegative integer offset")
    nodes = [state["nodes"][key] for key in state["completed_order"]]
    nodes = [
        n
        for n in nodes
        if contains.casefold() in ((n["path"] or "repository") + " " + n["symbol"]).casefold()
    ]
    selected, size = [], 0
    for node in nodes[offset : offset + 6]:
        # Source coordinates are sufficient for follow-up reads; full excerpts stay
        # in state.json. Avoid repeating the same long excerpt on every edge.
        view = json.loads(json.dumps(node))
        for entry in view["result"]["evidence"] + [
            e for dep in view["result"]["dependencies"] for e in dep["evidence"]
        ]:
            entry.pop("excerpt", None)
        length = len(json.dumps(view).encode())
        if selected and size + length > 50_000:
            break
        selected.append(view)
        size += length
    next_offset = offset + len(selected)
    return {
        "notes": selected,
        "total": len(nodes),
        "next_offset": next_offset if next_offset < len(nodes) else None,
        "exploration_status": state["status"],
        "stop_reasons": state["stop_reasons"],
        "deferred_count": len(state["deferred"]),
    }


def exploration_context(state):
    overview = [
        {"path": n["path"], "symbol": n["symbol"], "summary": n["result"]["summary"][:220]}
        for key in state["completed_order"]
        for n in [state["nodes"][key]]
    ]
    return (
        "\nA source-backed repository survey is available through module_notes "
        "(contains='', offset=0; paginate). Use its summary and named modules as a reading list. "
        "Search HydraDB and read_file to verify the code you document. "
        "Inferred dependency labels are not proven calls. Preserve uncertainty and unresolved questions. "
        f"Exploration status: {state['status']}; stop reasons: {state['stop_reasons']}; "
        f"notes: {len(overview)}; named modules: {state['coverage']['topics_explored']}. "
        f"Survey summaries: {json.dumps(overview[:24])}"
    )
