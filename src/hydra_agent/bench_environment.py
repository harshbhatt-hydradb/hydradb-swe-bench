"""Build inference-only images from pinned official dependency recipes, never gold tests."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .bench_data import EVALUATOR_VERSION, atomic_json, read_json, safe_task
from .bench_process import tracked_execute
from .environment import execute, git


def check_evaluator():
    try:
        version = importlib.metadata.version("swebench")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Install benchmark dependencies: uv sync --extra benchmark") from None
    if version != EVALUATOR_VERSION:
        raise ValueError(f"Expected swebench=={EVALUATOR_VERSION}; found {version}")


def docker_env() -> dict:
    # The Docker SDK used by the evaluator does not resolve CLI contexts itself.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AZURE_", "HYDRA_", "HYDRADB_", "OPENAI_"))
    }
    if env.get("DOCKER_CONTEXT") or not env.get("DOCKER_HOST"):
        host = subprocess.check_output(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            text=True,
            timeout=20,
        ).strip()
        if not host.startswith("unix://"):
            raise ValueError("This pipeline currently requires a local Unix Docker socket")
        env["DOCKER_HOST"] = host
    env.pop("DOCKER_CONTEXT", None)
    if not env["DOCKER_HOST"].startswith("unix://"):
        raise ValueError("This pipeline currently requires a local Unix Docker socket")
    return env


def preflight(root: Path, arch: str, *, min_free_gb: float = 120) -> dict:
    free = shutil.disk_usage(root).free / 1024**3
    if free < min_free_gb:
        raise ValueError(
            f"Only {free:.1f} GiB host disk free; need {min_free_gb:g} GiB before image work"
        )
    info = json.loads(
        subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True, timeout=20)
    )
    normalize = {"aarch64": "arm64", "arm64": "arm64", "amd64": "x86_64", "x86_64": "x86_64"}
    if normalize.get(info["Architecture"]) != arch or normalize.get(platform.machine()) != arch:
        raise ValueError(
            "Use a native runner matching the frozen architecture; emulation is not validated"
        )
    if info["MemTotal"] < 4 * 1024**3 - 128 * 1024**2:
        raise ValueError("Docker requires at least 4 GiB RAM for this serial smoke pipeline")
    disk = execute(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "ubuntu:22.04",
            "df",
            "-Pk",
            "/",
        ],
        timeout=120,
    )
    if disk.exit_code:
        raise ValueError("Cannot check Docker VM disk capacity")
    available = int(disk.output.strip().splitlines()[-1].split()[3]) * 1024 / 1024**3
    if available < min_free_gb:
        raise ValueError(f"Only {available:.1f} GiB Docker filesystem free; need {min_free_gb:g}")
    return {
        "host_free_gib": round(free, 1),
        "docker_free_gib": round(available, 1),
        "docker_memory_gib": round(info["MemTotal"] / 1024**3, 1),
        "arch": arch,
    }


def command(argv: list[str], log: Path, *, root: Path, timeout=3600, env=None, cwd=None):
    result = tracked_execute(root, argv, timeout=timeout, limit=2_000_000, env=env, cwd=cwd)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.output)
    if result.exit_code or result.truncated:
        raise RuntimeError(f"Command failed or log exceeded limit; inspect {log}")


def checkout(root: Path, task: dict) -> Path:
    repo = root / "repositories" / task["instance_id"]
    if not (repo / ".git").is_dir():
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "-q")
    try:
        commit = git(repo, "rev-parse", "--verify", task["base_commit"] + "^{commit}")
    except subprocess.CalledProcessError:
        command(
            [
                "git",
                "-C",
                str(repo),
                "fetch",
                "--depth=1",
                "https://github.com/" + task["repo"] + ".git",
                task["base_commit"],
            ],
            root / "build" / task["instance_id"] / "fetch.log",
            root=root,
            timeout=600,
        )
        commit = git(repo, "rev-parse", "--verify", task["base_commit"] + "^{commit}")
    if commit != task["base_commit"]:
        raise ValueError("Repository commit mismatch")
    return repo  # No checkout needed: Workspace uses git archive of the exact commit.


def recipe(task: dict, arch: str):
    check_evaluator()
    from swebench.harness.constants import MAP_REPO_TO_INSTALL, MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.test_spec import make_test_spec

    # Official recipe generation receives only safe fields plus empty evaluator placeholders.
    row = {**safe_task(task), "test_patch": "", "patch": "", "FAIL_TO_PASS": [], "PASS_TO_PASS": []}
    spec = make_test_spec(row)
    if spec.language != "py":
        raise ValueError("Initial benchmark environment adapter supports Python tasks only")
    if spec.arch != arch:
        raise ValueError("Official task recipe requires a different architecture")
    settings = MAP_REPO_VERSION_TO_SPECS[task["repo"]][task["version"]]
    install = ["source /opt/miniconda3/bin/activate testbed", "cd /workspace"]
    if task["repo"] in MAP_REPO_TO_INSTALL:
        install.append(MAP_REPO_TO_INSTALL[task["repo"]])
    install.extend(settings.get("pre_install", []))
    if settings.get("install"):
        install.append(settings["install"])
    install = [line.replace("/testbed", "/workspace") for line in install]
    return spec, "set -euxo pipefail\n" + "\n".join(install) + "\n"


def build_task(root: Path, task: dict, arch: str) -> dict:
    target = root / "build" / task["instance_id"]
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = target / "environment.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt["instance_id"], receipt["base_commit"], receipt["arch"]) != (
            task["instance_id"],
            task["base_commit"],
            arch,
        ):
            raise ValueError("Environment receipt does not match task")
        subprocess.run(
            ["docker", "image", "inspect", receipt["image"]],
            stdout=subprocess.DEVNULL,
            check=True,
            timeout=20,
        )
        return receipt
    repo = checkout(root, task)
    spec, install = recipe(task, arch)
    base, envdir, instance = [target / name for name in ("base", "env", "inference")]
    for path in (base, envdir, instance):
        path.mkdir(exist_ok=True)
    (base / "Dockerfile").write_text(spec.base_dockerfile)
    (envdir / "Dockerfile").write_text(spec.env_dockerfile)
    (envdir / "setup_env.sh").write_text(spec.setup_env_script)
    (base / ".dockerignore").write_text("**\n!Dockerfile\n")
    (envdir / ".dockerignore").write_text("**\n!Dockerfile\n!setup_env.sh\n")
    for context, tag in ((base, spec.base_image_key), (envdir, spec.env_image_key)):
        command(["docker", "build", "-t", tag, str(context)], context / "build.log", root=root)
    git(
        repo,
        "archive",
        "--format=tar",
        "--output",
        str(instance / "snapshot.tar"),
        task["base_commit"],
    )
    (instance / "install.sh").write_text(install)
    # No official instance image, repository clone/history, test patch, or eval script is copied.
    dockerfile = f"""FROM {spec.env_image_key}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep patch coreutils && rm -rf /var/lib/apt/lists/*
COPY snapshot.tar /opt/hydra-base.tar
COPY install.sh /opt/hydra-install.sh
RUN mkdir -p /workspace && tar -xf /opt/hydra-base.tar -C /workspace && cd /workspace && git init -q && git add -f . && git -c user.name=Hydra -c user.email=agent@localhost commit -qm baseline && bash /opt/hydra-install.sh && tar -xf /opt/hydra-base.tar -C /workspace && rm -rf /workspace/.git && mv /workspace /opt/hydra-seed && chmod -R a+rX /opt/hydra-seed && chown -R 65534:65534 /opt/hydra-seed && rm -f /opt/hydra-base.tar /opt/hydra-install.sh
ENV PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH
ENV HOME=/tmp PYTHONPATH=/workspace PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
USER 65534:65534
CMD ["sleep", "infinity"]
"""
    (instance / "Dockerfile").write_text(dockerfile)
    # Explicit context allowlist prevents build logs or campaign metadata entering the image.
    (instance / ".dockerignore").write_text("**\n!Dockerfile\n!snapshot.tar\n!install.sh\n")
    tag = "hydra-swe-inference:" + task["instance_id"].lower()
    command(["docker", "build", "-t", tag, str(instance)], instance / "build.log", root=root)
    image = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag], text=True, timeout=20
    ).strip()
    from .environment import Workspace

    with Workspace(
        repo, revision=task["base_commit"], image=image, seed_from_image=True
    ) as workspace:
        result = workspace.run(
            'python --version && test ! -d /opt/hydra-seed/.git && test -z "$(git status --porcelain)"',
            30,
        )
        if result.exit_code:
            raise RuntimeError("Inference environment validation failed")
    receipt = {
        "instance_id": task["instance_id"],
        "base_commit": task["base_commit"],
        "image": image,
        "tag": tag,
        "arch": arch,
        "repository": str(repo),
        "seed_from_image": True,
        "evaluator_version": EVALUATOR_VERSION,
    }
    atomic_json(receipt_path, receipt)
    return receipt
