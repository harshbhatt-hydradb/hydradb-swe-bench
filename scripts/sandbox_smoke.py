"""Verify the real Docker backend using synthetic source; no model calls or uploads."""

import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from hydra_agent.environment import Workspace, git


def main():
    checks = []

    def require(condition, name):
        if not condition:
            raise RuntimeError(f"Sandbox check failed: {name}")
        checks.append(name)
        print(f"PASS: {name}", flush=True)

    with tempfile.TemporaryDirectory(prefix="hydra-sandbox-check-") as directory:
        repo = Path(directory)
        original = "def add(a, b):\n    return a - b\n"
        (repo / "calc.py").write_text(original)
        (repo / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\n"
            "class TestAdd(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
        )
        git(repo, "init", "-q")
        git(repo, "add", ".")
        git(
            repo,
            "-c",
            "user.name=Smoke",
            "-c",
            "user.email=smoke@localhost",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "fixture",
        )
        with Workspace(repo) as workspace:
            container = workspace.container
            details = json.loads(
                subprocess.check_output(["docker", "inspect", container], timeout=30)
            )[0]
            host = details["HostConfig"]
            require(host["NetworkMode"] == "none", "network namespace disabled")
            require(host["ReadonlyRootfs"], "read-only root filesystem")
            require(not host.get("Binds") and not details["Mounts"], "no host mounts")
            require(
                not host["Privileged"] and host["CapDrop"] == ["ALL"],
                "unprivileged with capabilities dropped",
            )
            require("no-new-privileges" in host["SecurityOpt"], "no privilege escalation")
            require(
                host["Memory"] == 2 * 1024**3
                and host["NanoCpus"] == 2 * 10**9
                and host["PidsLimit"] == 128,
                "memory, CPU and process limits",
            )
            require(workspace.run("id -u", 10).output.strip() == "65534", "non-root user")
            require(
                workspace.run(
                    "test ! -e /var/run/docker.sock && test ! -d /Users && "
                    'test -z "${AZURE_OPENAI_API_KEY:-}${HYDRA_DB_API_KEY:-}"',
                    10,
                ).exit_code
                == 0,
                "host paths, Docker socket and controller keys absent",
            )
            require(
                workspace.run("touch /root-write-probe", 10).exit_code != 0,
                "root filesystem writes denied",
            )
            require(
                workspace.run("touch /workspace/write-probe /tmp/write-probe", 10).exit_code == 0,
                "workspace and temporary storage writable",
            )
            network_probe = (
                "import socket; s=socket.socket(); s.settimeout(2); "
                "raise SystemExit(0 if s.connect_ex(('1.1.1.1',443)) != 0 else 1)"
            )
            require(
                workspace.run("python -c " + shlex.quote(network_probe), 10).exit_code == 0,
                "outbound connection denied",
            )
            require(workspace.run("sleep 10", 0.2).timed_out, "command timeout enforced")
            require(
                workspace.run("python -m unittest -q", 20).exit_code != 0,
                "fixture fails before repair",
            )
            require(
                workspace.run("sed -i 's/a - b/a + b/' calc.py", 10).exit_code == 0,
                "source editing works",
            )
            require(
                workspace.run("python -m unittest -q", 20).exit_code == 0,
                "fixture passes after repair",
            )
            patch = workspace.patch()
            require("+    return a + b" in patch, "repair patch exported")
        require(
            subprocess.run(
                ["docker", "inspect", container], capture_output=True, timeout=30, check=False
            ).returncode
            != 0,
            "container removed on exit",
        )
        require(
            (repo / "calc.py").read_text() == original and git(repo, "status", "--porcelain") == "",
            "source checkout unchanged",
        )
        # Independently apply the exported patch in a new container at the same base.
        with Workspace(repo) as fresh:
            require(
                fresh.run("printf %s " + shlex.quote(patch) + " | git apply", 20).exit_code == 0,
                "patch applies to fresh snapshot",
            )
            require(
                fresh.run("python -m unittest -q", 20).exit_code == 0,
                "independent repaired snapshot passes tests",
            )
    print(json.dumps({"passed": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
