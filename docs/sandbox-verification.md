# Local sandbox verification — 2026-09-13

Installed Colima 0.10.3 and Lima 2.2.0 using Homebrew. Started the dedicated
`hydra-swe` profile on Apple Silicon with the VZ backend, 2 CPUs, 4 GiB memory,
20 GiB data disk, no host-directory mounts, and no SSH-agent forwarding. The
default Docker context was not changed. No login-time service was installed.
The Docker engine reports version 29.5.2.

Built `hydra-agent-sandbox:local` (image ID prefix `8a41639e8fc7`) from
`Dockerfile.sandbox`. The build context was 3.072 kB and excludes project source
and credentials. The initial runtime check exposed Git's dubious-ownership
guard because `/workspace` was root-owned. The backend now mounts that tmpfs
with UID/GID 65534 and mode 0700, matching the non-root container user.

Command:

```sh
DOCKER_CONTEXT=colima-hydra-swe uv run python scripts/sandbox_smoke.py
```

Result: **20 checks passed**. Checked network mode `none`, an actual denied
outbound TCP connection, read-only root with write denial, no host mounts,
no privileged mode, dropped capabilities, no-new-privileges, CPU/memory/PID
limits, UID 65534, absent host paths/Docker socket/controller key variables,
writable workspace and temporary storage, command timeout, and cleanup.

A synthetic addition bug fails its unit test before a deterministic edit and
passes afterward. The exported patch applies to a fresh container at the same
base and independently passes the test. The original checkout stays unchanged.

This smoke makes no model calls or HydraDB uploads. It does not verify a real
LLM repair, official SWE-bench performance, resistance to kernel/runtime
exploits, or adversarial background-process escape from command timeouts.
Repository dependencies must be baked into images before offline execution.
Only this synthetic Python fixture has been runtime-tested here.

Start again with `colima start hydra-swe`, select
`export DOCKER_CONTEXT=colima-hydra-swe`, and stop with
`colima stop hydra-swe`. Stopping preserves the image; completed attempts
remove their containers. The VM is left running after verification.
