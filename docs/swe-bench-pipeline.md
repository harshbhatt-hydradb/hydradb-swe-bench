# SWE-bench pipeline: implementation and operations

This is a serial operational pipeline for Experiment I in [technical_report.md](../technical_report.md), not a completed evaluation. It compares A (`baseline`, no memory) and H (`hydradb`, mandatory full-issue retrieval). Azure deployment, issue, tool-loop limits, and per-task inference image are shared. H incurs additional cold-ingestion and retrieval work; this is part of the treatment, not free compute.

## Current validation boundary

Offline tests exercise task allowlisting, worker-to-loop integration, full-question-first retrieval, no-memory control, frozen configuration/predictions, assignment accounting, timeout handling, safe resumption, and official report parsing. The installed official evaluator's CLI and a SymPy dependency recipe have been checked. Docker environment construction, real agent repairs, and official positive/negative grader controls have **not** yet been validated end to end. They remain launch gates, not assumed successes.

On the development Mac, host disk capacity is below the image-work threshold. No large image builds or benchmark inference were launched while implementing this pipeline. Provisioning another machine or deleting existing data is not automatic.

Development validation on 2026-09-13: **90 tests passed, 2 opt-in Docker tests skipped**. Real dataset preparation succeeded for `sympy__sympy-20590` in local `runs/swe-smoke-001`, with both arms and native ARM architecture, pinned to dataset revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Command preview and preliminary reporting succeeded. `doctor` stopped before Docker activity because only about 1 GiB host disk was free. Neither arm has run; the preliminary zero-resolved counts are not measured benchmark results.

## Commands

From the project root:

```bash
./benchmark
```

That one command selects SSD storage and starts its Colima VM on macOS, installs
optional benchmark dependencies through `uv`, prepares a default development task,
builds its environment, runs both arms, grades them, and writes a report. There is
no separate preparation step or campaign directory to name.

```bash
./benchmark --restart                 # Stop the active controller, then resume
./benchmark stop                      # Stop without resuming
./benchmark --fresh                    # Archive results and start new inference
./benchmark --restart --fresh          # Stop an active run and start over
./benchmark evaluate                  # Resume grading without model calls
./benchmark evaluate --dry-run        # Preview grading commands
./benchmark report                    # Read results without Docker/model calls
./benchmark --instances sympy__sympy-20590 --arms baseline
```

The default task is `sympy__sympy-20590`, with both `baseline` and `hydradb`, seed 42,
native architecture, 40 agent steps, 200,000 total tokens, and a 1,200-second agent
wall budget. `.env` supplies service settings. Use `--env-file` for another file.
Task selection and limits are remembered; omitting flags resumes the same settings.
`--instances`, `--arms`, `--seed`, `--arch`, `--dataset-revision`, `--max-steps`,
`--max-total-tokens`, and `--wall-seconds` override them. The default never selects
the entire dataset automatically.

macOS defaults to `/Volumes/PortableSSD/hydra-swe/runs/benchmark`. Linux defaults
to `runs/benchmark` and uses the current Docker daemon. `--output` or
`HYDRA_BENCH_OUTPUT` chooses another location. `HYDRA_BENCH_STORAGE=local` disables
the Mac SSD wrapper. The underlying command is `uv run --extra benchmark hydra-bench`.
`--help` works without Docker or an attached SSD.

Repeat the same command after Ctrl-C: completed predictions are reused, unsuccessful
or interrupted attempts are moved into `history/` before retrying, and grading
resumes using completed official task reports. An active run reports its PID and
the restart command. `--restart` sends SIGTERM to the recorded controller, waits for
cleanup, and takes the lock. Controller SIGTERM follows the same cleanup path as
Ctrl-C. A recorded orphan worker/evaluator is stopped before resumption, with PID
start-time and command checks to avoid signalling a reused PID.

Code, model, task, or limit changes automatically start a new run, saving previous
results beside it with a timestamp. `--fresh` does the same even when settings are
unchanged. New settings are validated and prepared before moving previous results.
Changed code does not prevent grading or reporting saved predictions. The internal
`campaign.json` remains provenance metadata; it is not a required user workflow.
Legacy `--campaign` stage commands remain available for old scripts and retain
their conservative, immutable-attempt behavior.

The evaluator remains pinned to `swebench==3.0.15` and `uv.lock`. Dataset preparation
resolves the requested revision to a full SHA. Inference receives only allowlisted
task fields; hidden tests and reference patches stay in the evaluator records.
Starting or retrying inference uses the configured paid model/HydraDB services.

## Resource and compatibility gates

### External SSD on this Mac

`scripts/with_ssd.sh` runs commands with Colima state, runtime disks, download caches,
and temporary workspaces under `/Volumes/PortableSSD/hydra-swe`. It checks that the
volume is actually mounted before creating directories. The dedicated profile is
`hydra-swe-ssd`; the internal `hydra-swe` profile is retained separately. Settings
apply only to the wrapped command, not your global shell or Docker context.
This uses Colima's supported [storage environment variables](https://colima.run/docs/faq/).

From this project directory:

```bash
./benchmark                         # Includes SSD setup and VM startup
./benchmark --restart               # Resume after stopping the active controller
./benchmark stop                    # Stop before stopping the VM
bash scripts/with_ssd.sh colima stop # Stop the VM before ejecting the SSD
```

Existing named campaign directories are retained separately from the new default
`runs/benchmark` directory. To inspect older results, use
`./benchmark report --output /Volumes/PortableSSD/hydra-swe/runs/swe-smoke-002`.
To grade their saved predictions, replace `report` with `evaluate`; no model
configuration or code-version match is required for these read/grade stages.

The original prepared campaign is preserved internally; use only the SSD copy for
this run so the two copies do not turn into independent attempts. The project,
existing Python virtual environment, and `.env` remain on the Mac. The wrapper does
not move existing caches, erase files, or free internal disk space. macOS still
needs internal free space for its own operation and swap. Keep the SSD attached
throughout builds, inference, and grading; stop all wrapped commands and the VM
before ejecting it. The attached volume is not encrypted, so protect retained
source, benchmark records, and traces appropriately. No drive formatting is needed.
The VM uses 4.25 GiB because the 4 GiB allocation exposed only about 3.81 GiB to
Docker, below the usable-memory gate. This is only a development allocation.
At setup the Mac was already using about 30 GiB of swap; close memory-heavy apps
and free internal disk space before starting benchmark workloads. External Docker
storage does not relocate macOS swap or guarantee enough memory under load.
The external campaign's `doctor` check passed on 2026-09-13: 734.0 GiB free
on its host filesystem, 148.3 GiB in Docker, 4.1 GiB usable VM memory, native
ARM64. This included only the small Ubuntu diagnostic container, not benchmark
image construction, agent inference, or official grading. The original internal
disk-space blocker above still applies when using the internal campaign/VM.

### Runner requirements

Use a native Linux x86-64 runner for a larger campaign. For ARM development, choose `--arch arm64` during preparation; recipes explicitly requiring x86-64 are rejected. The runner and Docker daemon must match the frozen architecture. Docker SDK calls resolve the selected CLI context into its local Unix socket; remote Docker hosts are not supported in this initial adapter.

Preflight checks free host disk before touching Docker, then architecture, daemon memory, and free Docker-filesystem disk. Default minimum is 120 GiB in both filesystems. `--min-free-gb` is an explicit override for a measured small-task development run, not a way to make an undersized full-campaign runner adequate. At least 4 GiB daemon memory is required by this development gate; this is not a full-campaign sizing recommendation. For larger runs, follow the [official Docker setup guidance](https://www.swebench.com/SWE-bench/guides/docker_setup/) and measure the pilot.

Inference currently inherits the existing sandbox limits: 2 CPUs, 2 GiB memory, 128 PIDs, 1 GiB writable repository tmpfs, 256 MiB `/tmp`, non-root UID 65534, no network, read-only image, dropped capabilities, and no host mounts. Large source trees/builds or tests requiring writes outside those locations may fail. Such failures are explicit; representative repository environment tests must pass before making benchmark-quality claims. The grader has its own official execution configuration and a 1,800-second test timeout.

## Inference/evaluation boundary

The trusted preparation process may read full dataset rows. It writes two separate products:

- `inference/tasks.json`: only `instance_id`, `repo`, full `base_commit`, `problem_statement`, `version`, and optional `environment_setup_commit`.
- `evaluation/dataset.json`: full selected rows for official grading, including reference patches, hidden test patches, and labels. Never mount this file into an agent container or ingest it into HydraDB.

Workers receive only allowlisted task records in their job files and reject additional task fields. Source is fetched at the exact base commit, exported via `git archive`, and provided without upstream history. The environment builder uses official **base/dependency** Dockerfiles plus an installation script generated with empty grading placeholders. It does not use official instance images for inference, nor copy their evaluation scripts or hidden tests. Repository install paths are relocated from `/testbed` to `/workspace`; original tracked source is overlaid again after installation. The image retains generated build products in a seed tree, copied into each disposable workspace, with a fresh scratch Git baseline. Both arms use the resulting image ID.

Repository source tests are legitimate agent tools; they are not the withheld evaluator test patch. H indexes only the eligible raw base archive, not generated installation artifacts, grading data, or previous solutions. Each H attempt gets a fresh namespace, waits for readiness, and then the controller queries HydraDB with the **complete issue text** before the first model request. Shell remains available to verify current code, edit, and test. Additional memory requests must be natural-language questions. Existing changed-file exclusion applies. Automatic immutable-index reuse and collection cleanup are not implemented here; retain/report cold index cost and manage retention explicitly.

The model-controlled environment cannot access campaign host files or the Docker socket. The host controller and evaluator are trusted processes, not mutually isolated operating-system users. Treat local campaign directories as sensitive artifacts. This boundary does not certify HydraDB server-side tenant isolation or eliminate pretraining contamination from public benchmark tasks.

## Attempts, grading, and recovery

An exclusive campaign lock prevents concurrent controllers. The serial runner claims an attempt before starting its subprocess, applies an outer deadline of agent wall budget + 120 seconds (plus 900 seconds for H ingestion), and records its terminal state. It tries to remove only the uniquely named attempt container. On a hard timeout, unexported partial edits are not recovered; the assignment remains a timeout with no usable prediction.

Sealed predictions are hashed and validated before aggregation. Missing predictions become explicit empty patches; assigned tasks are never silently dropped. Every inference assignment must be sealed before actual grading begins. Official evaluation IDs depend on the frozen configuration, arm, and complete prediction rows, preventing stale grades from being reused for changed patches. Only the official per-instance `resolved` field determines success—not the agent's answer, a test-looking trace, or evaluator exit code.

The simple command retries failed or interrupted inference on the next invocation.
Previous attempts and invalidated evaluation artifacts are moved into `history/`;
successful generated predictions, including empty patches, are not resampled.
Incomplete/error grading is retried; valid official reports for the same prediction
are retained and skipped by the evaluator. Leftover evaluator containers are
removed using their exact task/run names before a retry and after grading exits.
Unresolved official grades are completed outcomes and are not retried to improve
the score. These are development recovery semantics; account for archived attempts
when analysing usage or publishing results.

The controller holds an exclusive output lock, records its process identity, and
tracks the current worker/build/evaluator process group. Timeout, Ctrl-C, and
SIGTERM stop that group. Following a controller crash, the next invocation checks
and stops its recorded orphan process before recovering attempt state. No manual
lock-file deletion is needed. Daemon-side Docker build work can outlive an abrupt
host failure; Docker layer caches remain available to the next build.

## Artifacts and interpretation

```text
campaign.json                     dataset/code hashes, schedule, model and limits
inference/tasks.json              allowlisted tasks
repositories/<id>/.git/           host-side base-commit source objects
build/<id>/                      Dockerfiles, setup/build logs, image receipt
attempts/<arm>/<id>/
  state.json                     attempt status and sealed prediction hash
  job.json                       safe worker configuration; no API keys
  trajectory.jsonl               model/tool trace, HydraDB requests and evidence
  index-manifest.json            H source coverage, collection, readiness/timing
  result.json                    agent termination, returned token usage, latency
  patch.diff / prediction.jsonl  submitted repair
  worker.log                     bounded worker diagnostics
predictions/<arm>.jsonl           validated official-format submission
evaluation/dataset.json          evaluator-only full selected records
evaluation/<arm>/                evaluator receipt, bounded log, official reports
report.json / report.md          all assignments, grades, failures, usage coverage
```

Reports use all assigned tasks as the rate denominator; unresolved, missing, and ungraded assignments are not counted as solved. Unstarted/ungraded generated attempts mark the report **preliminary** and suppress the paired significance test. Reported model tokens sum available provider usage; missing usage and unknown monetary cost are explicit, not treated as known zero. Agent elapsed time excludes cold setup/indexing; ingestion timing is separately available in H manifests. Full end-to-end cost accounting remains open.

The paired report includes H-only/A-only successes, the assigned-task rate difference, and an exact McNemar test once outcomes are accounted for. Bootstrap confidence intervals, repository-cluster sensitivity, power analysis, and held-out confirmatory reporting from the technical protocol are not yet implemented. Do not publish a small operational subset as proof that graphs help coding agents.

Before the 10-task pilot: validate ordinary tests in representative inference environments, run known-correct and deliberately wrong repairs through the isolated official evaluator, then complete one genuine paired agent task. These checks must not feed oracle repairs or grader feedback into the model. Freeze the held-out split and retry/analysis rules only after the development gates are satisfied.
