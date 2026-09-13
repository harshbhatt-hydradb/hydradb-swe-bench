"""Line-oriented terminal chat with one sandbox and shared model history per session."""

import json
import re
from dataclasses import replace
from pathlib import Path

from .agent import run_agent

HELP = """Commands:
  /help    Show this help
  /paste   Enter a multiline message; finish with a line containing only .
  /diff    Show the cumulative patch
  /save    Export patch and conversation to the session directory
  /status  Show session usage and sandbox information
  /clear   Clear model conversation only; keep edits, memory exclusions and usage
  /exit    Save and close the sandbox (also /quit or Ctrl-D)
Ctrl-C cancels the current turn; the sandbox and earlier edits remain.
"""


def terminal_text(value: str) -> str:
    # Treat model/source output as data, not terminal escape sequences (including OSC).
    value = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", value)
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return "".join(c for c in value if c in "\n\t" or (ord(c) >= 32 and ord(c) != 127))


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

    def show(text):
        for secret in trace.secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        write(terminal_text(text))

    def observe(event):
        nonlocal turn_tokens, pending_tokens
        if event["event"] == "model_request":
            pending_tokens = event["reserved_tokens"]
        elif event["event"] == "model":
            turn_tokens = event["accounted_tokens"]
            pending_tokens = 0
        if previous_observer:
            previous_observer(event)
        if event["event"] == "tool_start":
            show(f"  {event['name']}: {next(iter(event['arguments'].values()))[:1000]}")
        elif event["event"] == "tool":
            result = event["result"]
            if "output" in result:
                show(f"  exit={result.get('exit_code', '?')}\n{result['output'][:1600]}")
            elif "hits" in result:
                show(f"  Retrieved {len(result['hits'])} evidence chunks.")
            elif "error" in result:
                show(f"  Tool error: {result['error']}")
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
    show("Hydra chat — one sandbox, continuous conversation. Type /help for commands.")
    show(f"Sandbox: {workspace.container or workspace.path}\nArtifacts: {output}")
    show(
        "Edits stay in the sandbox; patches are saved after each turn. Your checkout is unchanged."
    )
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
                    show(HELP)
                    continue
                if task == "/status":
                    show(
                        f"Turns: {len(turns)} | tokens: {total_tokens}/{limits.max_total_tokens} | "
                        f"context: {len(json.dumps(conversation).encode())} bytes | "
                        f"memory searches: {getattr(memory, 'search_calls', 0)}/40"
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
                    show(
                        (output / "patch.diff").read_text() or "No changes."
                        if task == "/diff"
                        else f"Saved {size} patch bytes to {output / 'patch.diff'}"
                    )
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
            show("Agent is working…")
            turn_tokens = pending_tokens = 0
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
            safe_result = json.dumps(result)
            for secret in trace.secrets:
                if secret:
                    safe_result = safe_result.replace(secret, "[REDACTED]")
            turns.append(json.loads(safe_result))
            save()
            show("\nAgent: " + (result["summary"] or f"Turn stopped: {result['status']}"))
            show(
                f"[{result['status']} · {result['total_tokens']} tokens · "
                f"session {total_tokens}/{limits.max_total_tokens}]"
            )
            if result["status"] == "context_limit":
                show(
                    "Context is full. Use /clear, then restate your next task; edits are retained."
                )
    finally:
        trace.observer = previous_observer
        save()
        show(f"Saved session and patch to {output}. Closing sandbox.")
    return 0
