import io

import pytest
from rich.cells import cell_len
from rich.console import Console

from hydra_agent.terminal import ChatTerminal, terminal_text


@pytest.mark.parametrize("width", [40, 80, 120])
def test_rendering_adapts_to_terminal_width(width):
    stream = io.StringIO()
    ui = ChatTerminal(console=Console(file=stream, width=width, color_system=None))
    ui.answer("**Canvas rendering**\n\n- Uses `drawPoint` for keypoints.\n- Uses lines for joints.")
    ui.panel("Session", "repository/" * 20)
    assert "Canvas rendering" in stream.getvalue()
    assert max(cell_len(line) for line in stream.getvalue().splitlines()) <= width


def test_tools_are_literal_sanitized_and_collapsed():
    display = []
    ui = ChatTerminal(write=display.append, secrets=("api-secret",))
    ui.activity("shell", "echo '[bold]api-secret[/bold]'\x1b[2J")
    ui.tool_result({"exit_code": 0, "output": "\n".join(f"line {i}" for i in range(30))})
    text = "\n".join(display)
    assert "[bold][REDACTED][/bold]" in text
    assert "api-secret" not in text and "\x1b" not in text
    assert "Preview collapsed" in text
    assert "line 29" not in text
    assert "line 29" in ui.last_output
    ui.expanded = True
    ui.tool_result({"exit_code": 0, "output": ui.last_output})
    assert "line 29" in "\n".join(display)


def test_graph_footer_never_equates_connection_with_retrieval():
    display = []
    ui = ChatTerminal(write=display.append)
    result = {
        "status": "model_stopped",
        "total_tokens": 120,
        "graph_usage": {
            "enabled": True,
            "searches": 0,
            "requests": 0,
            "evidence_chunks": 0,
            "errors": 0,
        },
    }
    ui.footer(result, 240, 1000, 0)
    assert "Graph NOT queried this turn" in "\n".join(display)
    result["graph_usage"]["searches"] = 1
    ui.footer(result, 240, 1000, 0)
    assert "no HTTP query sent" in "\n".join(display)
    result["graph_usage"].update(requests=1, evidence_chunks=3)
    ui.footer(result, 240, 1000, 0)
    assert "1 HTTP query attempts · 3 chunks" in "\n".join(display)


def test_control_characters_and_bidi_are_removed():
    assert terminal_text("hello\x9b\u202eworld") == "helloworld"
