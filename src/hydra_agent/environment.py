import os
import selectors
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CommandResult:
    output: str
    exit_code: int
    timed_out: bool = False
    truncated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def execute(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 60,
    limit: int = 16000,
    env: dict | None = None,
) -> CommandResult:
    """Drain output continuously, retain bounded bytes, kill the process group on timeout."""
    start = time.monotonic()
    output = bytearray()
    total = 0
    timed_out = False
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    ) as proc:
        assert proc.stdout is not None
        try:
            return _collect(proc, start, timeout, limit, output, total, timed_out)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def _collect(proc, start, timeout, limit, output, total, timed_out):
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while selector.get_map():
            if time.monotonic() - start >= timeout:
                timed_out = True
                break
            for key, _ in selector.select(timeout=min(0.1, timeout)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                else:
                    total += len(chunk)
                    output.extend(chunk[: max(0, limit - len(output))])
        if not timed_out:
            try:
                proc.wait(timeout=max(0.01, timeout - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                timed_out = True
        # Also stop background descendants when the main shell finishes.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    return CommandResult(
        output.decode("utf-8", errors="replace"),
        124 if timed_out else proc.returncode,
        timed_out,
        total > limit,
    )


def git(repo: Path, *args: str) -> str:
    return (
        subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE, timeout=60)
        .decode()
        .strip()
    )


class Workspace:
    """Start at a commit in a disposable copy; never expose source Git history to the agent."""

    def __init__(
        self,
        repo: Path,
        *,
        revision: str = "HEAD",
        backend: str = "docker",
        image: str = "hydra-agent-sandbox:local",
    ):
        self.repo = repo.resolve()
        if not self.repo.is_dir():
            raise ValueError(
                "--repo must point to an existing directory. Replace example paths with "
                "the path to the Git repository you want to repair."
            )
        try:
            git(self.repo, "rev-parse", "--git-dir")
        except subprocess.CalledProcessError:
            raise ValueError(
                "--repo is not an accessible Git repository. Use a local Git checkout."
            ) from None
        try:
            self.base_commit = git(
                self.repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                "--revision must resolve to a commit in --repo. Check the branch, tag, or "
                "commit ID; HEAD requires at least one commit."
            ) from None
        self.backend = backend
        self.image = image
        self.container: str | None = None
        self.temp: tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None
        self.initial_commit = ""
        self.archive: Path | None = None
        # Only the host controller receives Azure credentials.
        self.local_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        }

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hydra-agent-")
        root = Path(self.temp.name)
        archive = root / "repo.tar"
        self.archive = archive
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo),
                    "archive",
                    "--format=tar",
                    "--output",
                    str(archive),
                    self.base_commit,
                ],
                check=True,
                timeout=60,
            )
            if self.backend == "docker":
                self.container = "hydra-agent-" + uuid.uuid4().hex
                subprocess.run(
                    [
                        "docker",
                        "run",
                        "--detach",
                        "--rm",
                        "--name",
                        self.container,
                        "--network=none",
                        "--read-only",
                        "--cap-drop=ALL",
                        "--security-opt=no-new-privileges",
                        "--pids-limit=128",
                        "--memory=2g",
                        "--cpus=2",
                        "--user=65534:65534",
                        "--tmpfs=/workspace:rw,exec,uid=65534,gid=65534,mode=0700,size=1g",
                        "--tmpfs=/tmp:rw,exec,mode=1777,size=256m",
                        "--workdir=/workspace",
                        self.image,
                        "sleep",
                        "infinity",
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                with archive.open("rb") as source:
                    subprocess.run(
                        [
                            "docker",
                            "exec",
                            "-i",
                            self.container,
                            "tar",
                            "--no-same-owner",
                            "-xf",
                            "-",
                            "-C",
                            "/workspace",
                        ],
                        stdin=source,
                        check=True,
                        capture_output=True,
                        timeout=60,
                    )
            elif self.backend == "local":
                self.path = root / "workspace"
                self.path.mkdir()
                with tarfile.open(archive) as source:
                    source.extractall(self.path, filter="data")
                self.local_env["HOME"] = str(root)
            else:
                raise ValueError("Unknown workspace backend")
            result = self.run(
                "git init -q && git add -f . && "
                "git -c user.name=Hydra -c user.email=agent@localhost "
                "-c core.hooksPath=/dev/null commit -qm baseline --allow-empty",
                60,
            )
            if result.exit_code:
                raise RuntimeError("Cannot initialize disposable repository: " + result.output)
            self.initial_commit = self.run("git rev-parse HEAD", 10).output.strip()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def run(self, command: str, timeout: float, limit: int = 16000) -> CommandResult:
        if self.backend == "docker":
            # timeout runs inside the container; killing the docker client alone is insufficient.
            result = execute(
                [
                    "docker",
                    "exec",
                    self.container,
                    "timeout",
                    "-s",
                    "KILL",
                    str(max(0.1, timeout)),
                    "bash",
                    "-c",
                    command,
                ],
                timeout=timeout + 5,
                limit=limit,
            )
            result.timed_out |= result.exit_code in (124, 137)
            return result
        return execute(
            ["bash", "-c", command], cwd=self.path, timeout=timeout, limit=limit, env=self.local_env
        )

    def patch(self) -> str:
        # The baseline hash is held outside the agent's mutable Git HEAD.
        result = self.run(
            f"git add -A && git diff --cached --binary --no-ext-diff "
            f"--no-textconv {self.initial_commit} --",
            30,
            limit=10_000_000,
        )
        if result.exit_code or result.truncated:
            raise RuntimeError("Patch extraction failed or exceeded 10 MB")
        return result.output

    def interrupt(self) -> None:
        """Stop outstanding commands in this disposable container, retaining its PID 1."""
        if self.backend != "docker":
            return  # execute() already kills the local command process group.
        code = (
            "import os, signal\n"
            "for entry in os.listdir('/proc'):\n"
            " if entry.isdigit() and int(entry) not in (1, os.getpid()):\n"
            "  try: os.kill(int(entry), signal.SIGKILL)\n"
            "  except ProcessLookupError: pass\n"
        )
        result = execute(["docker", "exec", self.container, "python", "-c", code], timeout=10)
        if result.exit_code:
            raise RuntimeError("Cannot stop sandbox commands safely; session must close")

    def changed_paths(self) -> set[str]:
        """Files changed from the indexed snapshot, including staged changes and deletions."""
        result = self.run(
            f"git diff --name-only -z --no-ext-diff --no-textconv {self.initial_commit} --",
            10,
            limit=1_000_000,
        )
        if result.exit_code or result.truncated:
            raise RuntimeError("Cannot verify repository freshness")
        return set(filter(None, result.output.split("\0")))

    def __exit__(self, *args):
        try:
            if self.container:
                subprocess.run(
                    ["docker", "rm", "-f", self.container],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
        finally:
            if self.temp:
                self.temp.cleanup()
