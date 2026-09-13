"""Line-oriented terminal chat with one sandbox and shared model history per session."""

import json
from dataclasses import replace
from pathlib import Path

from .agent import run_agent
from .terminal import ChatTerminal, terminal_text  # noqa: F401 -- compatibility export

HELP = """Commands:
  /help    Show this help
  /paste   Enter a multiline message; finish with a line containing only .
  /diff    Show the cumulative patch
  /save    Export patch and conversation to the session directory
  /status  Show session usage and sandbox information
  /graph   Show the connected graph and last turn's retrieval usage
  /tools   Toggle compact/expanded tool output
  /last    Show the last shell output (within the harness output limit)
  /clear   Clear model conversation only; keep edits, memory exclusions and usage
  /exit    Save and close the sandbox (also /quit or Ctrl-D)
Ctrl-C cancels the current turn; the sandbox and earlier edits remain.
"""


def close_pending_tools(messages: list[dict]) -> None:
    """An interrupted batch must have a response for every call before another user turn."""
    last = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"),
        None,
    )
    if last is None:
        return
    answered = {m.get("tool_call_id") for m in messages[last + 1 :] if m.get("role") == "tool"}
    for call in messages[last].get("tool_calls", []):
        if call["id"] not in answered:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(
                        {
                            "error": "Turn stopped; tool result unavailable. Inspect current workspace before continuing."
                        }
                    ),
                }
            )


def chat_session(
    model,
    workspace,
    limits,
    trace,
    output: Path,
    *,
    memory=None,
    scope=None,
    initial_task=None,
    read_input=input,
    write=print,
) -> int:
    try:
        import readline  # noqa: F401 -- enables terminal line editing without saving history
    except ImportError:
        pass
    conversation, turns = [], []
    total_tokens = 0
    turn_tokens = 0
    pending_tokens = 0
    previous_observer = trace.observer
    ui = ChatTerminal(write=write, secrets=trace.secrets)
    graph_usage = {}
    active_tool = None

    def show(text):
        ui.notice(text)

    def observe(event):
        nonlocal turn_tokens, pending_tokens, active_tool
        if event["event"] == "model_request":
            pending_tokens = event["reserved_tokens"]
        elif event["event"] == "model":
            turn_tokens = event["accounted_tokens"]
            pending_tokens = 0
        if previous_observer:
            previous_observer(event)
        if event["event"] == "hydradb_request" and event.get("path") == "/query":
            graph_usage["requests"] += 1
        if event["event"] == "tool_start":
            active_tool = event["name"]
            if active_tool == "memory_search":
                graph_usage["searches"] += 1
            ui.activity(event["name"], next(iter(event["arguments"].values())))
        elif event["event"] == "tool":
            result = event["result"]
            if active_tool == "memory_search":
                graph_usage["evidence_chunks"] += len(result.get("hits", []))
                graph_usage["errors"] += int("error" in result)
            active_tool = None
            ui.tool_result(result)
        elif event["event"] == "model":
            choices = event["response"].get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                if message.get("tool_calls") and message.get("content"):
                    show("Agent: " + message["content"])

    def save():
        patch = workspace.patch()
        (output / "patch.diff").write_text(patch)
        transcript = json.dumps(conversation, indent=2, ensure_ascii=False)
        for secret in trace.secrets:
            if secret:
                transcript = transcript.replace(secret, "[REDACTED]")
        (output / "conversation.json").write_text(transcript + "\n")
        result = {
            "status": "chat_session",
            "turns": turns,
            "total_tokens": total_tokens,
            "patch_bytes": len(patch.encode()),
            "resumable": False,
        }
        (output / "session.json").write_text(json.dumps(result, indent=2) + "\n")
        return len(patch.encode())

    trace.observer = observe
    ui.header(workspace, model, memory, output)
    pending = initial_task
    try:
        while True:
            try:
                task = pending if pending is not None else read_input("\nYou › ")
                pending = None
                task = task.strip()
                if not task:
                    continue
                if task in {"/exit", "/quit"}:
                    break
                if task == "/help":
                    ui.panel("Session commands", HELP)
                    continue
                if task == "/graph":
                    ui.graph(memory, turns[-1]["graph_usage"] if turns else None)
                    continue
                if task == "/tools":
                    ui.expanded = not ui.expanded
                    show("Tool output: " + ("expanded" if ui.expanded else "compact"))
                    continue
                if task == "/last":
                    ui.panel("Last shell output", ui.last_output)
                    continue
                if task == "/status":
                    ui.panel(
                        "Session status",
                        f"Turns: {len(turns)} | tokens: {total_tokens}/{limits.max_total_tokens} | "
                        f"context: {len(json.dumps(conversation).encode())} bytes | "
                        f"memory searches: {getattr(memory, 'search_calls', 0)}/40",
                    )
                    continue
                if task == "/clear":
                    conversation.clear()
                    show(
                        "Conversation cleared. Workspace edits, memory exclusions and usage retained."
                    )
                    continue
                if task in {"/diff", "/save"}:
                    size = save()
                    if task == "/diff":
                        ui.diff((output / "patch.diff").read_text())
                    else:
                        show(f"Saved {size} patch bytes to {output / 'patch.diff'}")
                    continue
                if task == "/paste":
                    show("Enter message; end with a single . on its own line.")
                    lines = []
                    while (line := read_input("… ")) != ".":
                        lines.append(line)
                    task = "\n".join(lines).strip()
                    if not task:
                        continue
                elif task.startswith("/"):
                    show("Unknown command. Type /help.")
                    continue
            except EOFError:
                break
            except KeyboardInterrupt:
                show("Input cancelled. Use /exit to close the sandbox.")
                continue
            if total_tokens >= limits.max_total_tokens:
                show("Session token budget exhausted. Use /diff, /save or /exit.")
                continue
            ui.begin(len(turns) + 1)
            turn_tokens = pending_tokens = 0
            graph_usage = {
                "enabled": memory is not None,
                "searches": 0,
                "requests": 0,
                "evidence_chunks": 0,
                "errors": 0,
            }
            active_tool = None
            trace.emit("chat_turn", turn=len(turns) + 1)
            try:
                result = run_agent(
                    model,
                    workspace,
                    task,
                    replace(limits, max_total_tokens=limits.max_total_tokens - total_tokens),
                    trace,
                    memory=memory,
                    scope=scope,
                    conversation=conversation,
                )
            except KeyboardInterrupt:
                workspace.interrupt()
                result = {
                    "status": "interrupted",
                    "summary": "Turn interrupted. Inspect changes before continuing.",
                    "total_tokens": turn_tokens + pending_tokens,
                }
            finally:
                close_pending_tools(conversation)
            total_tokens += result["total_tokens"]
            result["graph_usage"] = dict(graph_usage)
            trace.emit("chat_turn_summary", turn=len(turns) + 1, **result)
            safe_result = json.dumps(result)
            for secret in trace.secrets:
                if secret:
                    safe_result = safe_result.replace(secret, "[REDACTED]")
            turns.append(json.loads(safe_result))
            patch_bytes = save()
            ui.answer(result["summary"] or f"Turn stopped: {result['status']}")
            ui.footer(result, total_tokens, limits.max_total_tokens, patch_bytes)
            if result["status"] == "context_limit":
                show(
                    "Context is full. Use /clear, then restate your next task; edits are retained."
                )
    finally:
        trace.observer = previous_observer
        save()
        show(f"Saved session and patch to {output}. Closing sandbox.")
    return 0
