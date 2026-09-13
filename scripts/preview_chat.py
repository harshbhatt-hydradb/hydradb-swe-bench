"""Preview the terminal design with synthetic data; no API calls or sandbox required."""

from pathlib import Path
from types import SimpleNamespace

from hydra_agent.terminal import ChatTerminal


def main():
    ui = ChatTerminal()
    ui.notice("DESIGN PREVIEW — synthetic session, not a live agent run")
    ui.header(
        SimpleNamespace(
            backend="docker", repo=Path("targets/YogaIntelliJ"), base_commit="2cc0eaa7ea5c"
        ),
        SimpleNamespace(config=SimpleNamespace(deployment="grok-4.3")),
        SimpleNamespace(collection="attempt_19fe9877a5504235920aaf66012525dc", retrieval_only=True),
        Path("runs/design-preview"),
    )
    ui.begin(1)
    ui.activity("memory_search", "Where are pose keypoints rendered?")
    ui.tool_result(
        {
            "hits": [
                {"path": "frontend/src/pages/Yoga/Yoga.js"},
                {"path": "frontend/src/utils/helper/index.js"},
            ]
        }
    )
    ui.activity("shell", "sed -n '1,20p' frontend/src/utils/helper/index.js")
    ui.tool_result(
        {
            "exit_code": 0,
            "output": "export function drawPoint(ctx, x, y, r, color) {\n"
            "    ctx.beginPath();\n    ctx.arc(x, y, r, 0, 2 * Math.PI);\n"
            "    ctx.fillStyle = color;\n    ctx.fill();\n}",
        }
    )
    ui.answer(
        "Keypoints are drawn on an **HTML Canvas**, over the webcam view.\n\n"
        "- `drawPoint()` paints the joints.\n- `drawSegment()` connects them into a skeleton.\n\n"
        "The active yoga page uses these custom Canvas 2D helpers. No files were changed."
    )
    ui.footer(
        {
            "status": "model_stopped",
            "total_tokens": 2400,
            "elapsed_seconds": 4.2,
            "graph_usage": {
                "enabled": True,
                "searches": 1,
                "requests": 1,
                "evidence_chunks": 2,
                "errors": 0,
            },
        },
        2400,
        200000,
        0,
    )
    ui.notice("\nYou › Ask a follow-up, or type /help")


if __name__ == "__main__":
    main()
