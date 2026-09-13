"""Run ownership and cleanup for the serial benchmark controller."""

import fcntl
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from .bench_data import atomic_json, read_json
from .environment import execute


def identity(pid: int) -> str:
    """Include start time and command so a recycled PID is never signalled."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=,command="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def process_record(pid: int) -> dict:
    return {"pid": pid, "identity": identity(pid)}


def signal_record(record: dict, sig: int, *, group=False) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 1 or not record.get("identity"):
        return False
    if identity(pid) != record["identity"]:
        return False
    try:
        if group:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def tracked_execute(root: Path, argv: list[str], **kwargs):
    path = root / ".child.json"
    try:
        return execute(argv, on_start=lambda pid: atomic_json(path, process_record(pid)), **kwargs)
    finally:
        path.unlink(missing_ok=True)


def recover_child(root: Path) -> None:
    path = root / ".child.json"
    if path.exists():
        signal_record(read_json(path), signal.SIGKILL, group=True)
        path.unlink()


@contextmanager
def run_lock(root: Path, *, restart=False, stop=False, recover=True):
    """Keep the lock outside the results directory so archiving cannot bypass it."""
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / ("." + root.name + ".lock")
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            try:
                owner = json.load(lock)
            except (ValueError, OSError):
                owner = {}
            if not (restart or stop):
                raise ValueError(
                    f"Benchmark is running (PID {owner.get('pid', 'unknown')}). "
                    "Use ./benchmark --restart to take over, or ./benchmark stop."
                ) from None
            if not signal_record(owner, signal.SIGTERM):
                raise ValueError("Run owner is still starting or stopping; retry in a moment")
            print(f"Stopping benchmark PID {owner['pid']}…", flush=True)
            deadline = time.monotonic() + 45
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ValueError(
                            "Previous run is still cleaning up; retry shortly"
                        ) from None
                    time.sleep(0.2)
        lock.seek(0)
        lock.truncate()
        json.dump(process_record(os.getpid()), lock)
        lock.flush()
        try:
            if recover:
                recover_child(root)
            yield
        finally:
            lock.seek(0)
            lock.truncate()
            fcntl.flock(lock, fcntl.LOCK_UN)
