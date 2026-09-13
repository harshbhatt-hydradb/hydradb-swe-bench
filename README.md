# HydraDB coding-agent harness

An initial Azure OpenAI coding agent for controlled repository-memory experiments. Read [the design memo](design-memo.md) and [the detailed roadmap](todo.md).

The agent inspects, edits, and tests a disposable snapshot of a Git commit, then exports a patch. Optional HydraDB integration ingests that snapshot and gives the agent graph-aware retrieval. Official SWE-bench execution is still planned; no benchmark score is claimed.

## Setup

Requires Python 3.11+, `uv`, Git, and a running Docker daemon for the default backend.

```sh
uv sync --locked
cp .env.example .env
# Edit .env locally: endpoint, API key, and your Azure deployment name.
uv run hydra-agent doctor
uv run hydra-agent doctor --live
docker build -f Dockerfile.sandbox -t hydra-agent-sandbox:local .
```

The Azure deployment must support Chat Completions and function calling. The provider uses the [Azure v1 endpoint](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/how-to/switching-endpoints) with the official OpenAI Python SDK. An API key alone is insufficient: the resource endpoint and deployment name are also required. Environment variables override `.env`; use `--env-file PATH` **before** the subcommand to select another file. Never paste credentials into task prompts or commit `.env`.

## Interactive terminal chat

Use `chat` for back-and-forth conversation in one persistent sandbox. From this
project directory, using the existing YogaIntelliJ graph:

```sh
colima start hydra-swe
export DOCKER_CONTEXT=colima-hydra-swe
uv run hydra-agent chat \
  --repo targets/YogaIntelliJ \
  --revision 2cc0eaa7ea5c07b75911a9eefc8484b94826036d \
  --memory hydradb \
  --reuse-index runs/19fe9877a5504235920aaf66012525dc/index-manifest.json
```

At `You ›`, ask a question or request a change, then send follow-ups. The model
receives earlier messages and tool results, and the same container retains file
edits between turns. Graph readiness is checked once at startup; this reuse mode
never ingests. The source checkout remains unchanged. Omit the memory options to
chat without HydraDB, or use `--task` to send an initial message automatically.

Commands: `/help`, `/paste` (multiline input ending with `.`), `/diff`, `/save`,
`/status`, `/clear`, and `/exit` (also `/quit` or Ctrl-D). Ctrl-C during a model
turn interrupts it, stops outstanding container commands, and preserves edits.
Inspect `/diff` afterward because interrupted commands may have partially edited
files. `/clear` resets conversation only, not edits, graph exclusions or usage.

After every turn and on normal exit, the session directory contains `patch.diff`,
`conversation.json`, `session.json`, `trajectory.jsonl`, and the usual manifests.
Review/apply the exported patch yourself; nothing is applied to your checkout.
Chat does not produce a SWE-bench prediction. Traces and patches may contain
sensitive source; configured API keys are redacted from conversation/status
output, but this is not a general secret scanner.

`--max-total-tokens` covers the whole session; step and wall limits apply per turn.
HydraDB's 40-query cap also covers the whole session. Interrupted in-flight model
calls conservatively consume their reserved budget because billed usage may be
unknown. A context-limit stop requires `/clear` and a fresh task description;
automatic compaction is not implemented. Provider retries can extend wall time.

This first version is line-oriented terminal chat with tool progress, not a
full-screen UI or token-streaming renderer. Sessions cannot be resumed after
exit, and abrupt process termination may lose the current turn's unsaved edits.
The default Python image supports repository inspection but not this React
project's frontend tests; those need a Node/dependency-equipped `--image`.

## Run a repair

### Local sandbox on this Mac

A dedicated Colima profile named `hydra-swe` provides the Docker engine without
changing your default Docker context. Start it if stopped, then select it in the
terminal where you run the agent:

```sh
colima start hydra-swe
export DOCKER_CONTEXT=colima-hydra-swe
docker info
uv run python scripts/sandbox_smoke.py
```

The profile was provisioned with 2 CPUs, 4 GiB RAM, a 20 GiB data disk, no host
directory sharing, and no SSH-agent forwarding. For setup on another Mac, install
[Colima](https://github.com/abiosoft/colima#installation) and create the profile:

```sh
brew install colima docker
colima start hydra-swe --runtime docker --vm-type vz --cpus 2 --memory 4 \
  --disk 20 --mount none --ssh-agent=false --ssh-config=false --activate=false
export DOCKER_CONTEXT=colima-hydra-swe
docker build -f Dockerfile.sandbox -t hydra-agent-sandbox:local .
```

Each agent attempt gets a disposable, non-root container with networking disabled,
a read-only root filesystem, writable temporary workspace, no host mounts, no
controller credentials, and limits of 2 CPUs, 2 GiB RAM, and 128 processes. The
controller stays on the host and calls Azure/HydraDB; source editing and test
execution happen inside the container. The smoke test makes no API calls or
uploads and uses deterministic edits, not an LLM. It verifies isolation settings,
timeout, repair, independent patch application, and cleanup. These checks are not
a security audit or proof against kernel/container-runtime vulnerabilities.

Stop the VM when finished with `colima stop hydra-swe`; this preserves the built
image. Completed attempts remove their containers automatically. This setup does
not install a login-time background service.

### Agent command

Replace the example `--repo` path with an existing local Git checkout and `--task` with
the actual bug description. Check the inputs with `git -C /your/repo rev-parse --verify HEAD`;
the repository must have at least one commit. The harness directory itself only works as
the target if it is a Git checkout.

```sh
uv run hydra-agent run \
  --repo /absolute/path/to/target-repository \
  --revision HEAD \
  --task 'Fix the reported bug and run the relevant tests.'
```

Only committed files at the selected revision are copied; working-tree changes and untracked files are not inputs. Git archive attributes such as `export-ignore` also apply. The source checkout remains unchanged. The default container has no network, host mounts, or controller credentials; review tracked source for secrets before using it as input. Bake target dependencies into an appropriate `--image` before running. The generic image provides Python, Git, ripgrep, and patch; it does not contain arbitrary repository dependencies.

Each `runs/<attempt-id>/` contains `manifest.json`, `trajectory.jsonl`, `result.json`, `patch.diff`, and `prediction.jsonl`. Inspect the patch before applying it to the matching base commit:

```sh
git -C /absolute/path/to/target-repository apply --check /absolute/path/to/runs/ATTEMPT/patch.diff
git -C /absolute/path/to/target-repository apply /absolute/path/to/runs/ATTEMPT/patch.diff
```

Set `--instance-id` to the real benchmark ID when exporting a benchmark prediction. `prediction.jsonl` follows the [official prediction schema](https://www.swebench.com/SWE-bench/guides/evaluation/), but generation is not grading. `submitted` only means the agent called finish; independent tests determine success. Non-submission stops exit with status 2 and retain a partial patch when extraction succeeds.

Useful limits: `--max-steps`, `--max-total-tokens`, `--max-completion-tokens`, `--max-context-bytes`, `--command-timeout`, and `--wall-seconds`. The token preflight uses a conservative byte estimate and can stop early. The wall limit is checked between actions; SDK retries may extend elapsed time. No hard dollar cap is implemented.

For trusted local development without Docker, explicitly opt in:

```sh
uv run hydra-agent run --repo /path/to/trusted-repository \
  --task-file issue.txt --backend local --allow-local-shell
```

Local commands can access the host despite using a disposable workspace. Use Docker for untrusted code. Run traces contain source and tool output; treat them as sensitive. The current snapshot export does not expand submodules or Git LFS objects.

## Run with HydraDB repository context

Add `HYDRA_DB_API_KEY`, `HYDRA_DB_DATABASE`, and optionally `HYDRA_DB_BASE_URL` to `.env`. These target the hosted HydraDB v2 API. Azure credentials are also required for the coding model.

```sh
uv run hydra-agent run \
  --repo /absolute/path/to/target-repository \
  --revision HEAD \
  --task 'Fix the reported bug and run the relevant tests.' \
  --memory hydradb
```

The controller performs these steps:

1. Create a disposable workspace from the selected commit and derive the upload corpus from the same Git archive. Record source IDs, hashes, and exclusions in `index-manifest.json` before uploading.
2. Check the configured database, create it if absent, and wait for infrastructure readiness. Upload eligible files through multipart `POST /context/ingest` in batches of 20, in a fresh attempt collection. The service builds its automatic content graph; this version does not extract a static call graph.
3. Poll `/context/status` until **every source is `completed`**. `graph_creation` means chunks are searchable but the graph is unfinished, so the agent does not start at that stage. Setup failure records a failed index manifest and stops the run.
4. Start the Azure agent. `memory_search` calls HydraDB `/query` in `thinking` mode with graph context, collection scoping, and an explicit source-ID filter. The adapter returns bounded source evidence, paths, and hashes. It checks primary and related chunks against the local source allowlist; raw graph paths are not injected because they may reference unvalidated sources.
5. Keep current file reads, edits, and tests in the execution workspace through `shell`. Before retrieval, detect changes from the base snapshot and exclude those files from subsequent queries. Previously returned snippets can still be stale, so the prompt requires reading current source before editing. New files are available through shell; incremental re-ingestion is future work.
6. Export the patch and run artifacts. API status codes, request IDs, indexing duration, retrieval counts, and discarded-file counts appear in the trajectory. Controller credentials are not passed into the execution container.

Source coverage is explicit: the policy includes common source/text extensions and skips dependency/build directories, recognized credential filenames, symlinks, binary/non-UTF-8 content, empty files, and files over 500 KB. This is a filename/content-type policy, not a comprehensive secret scanner. `--index-max-file-bytes` changes the per-file limit. Exceeding the default 50 MB total (`--index-max-bytes`) aborts before upload rather than silently truncating the corpus. Git archive attributes also affect the snapshot.

The default indexing deadline is 900 seconds (`--index-timeout`), separate from agent execution time. Retrieval is capped at 40 searches and 8 evidence chunks per call. API requests have bounded retries for rate-limit/server errors; no hard dollar budget includes HydraDB's internal model usage. Client-side tests verify request scoping and output filtering, not the server's internal graph isolation. Use a dedicated benchmark database and verify isolation live before scoring.

Unless `--reuse-index` is supplied (see below), each run uses a fresh collection, re-ingests the snapshot, and incurs cold-indexing cost. Uploaded data, including partially indexed runs, remains in HydraDB until you remove it; the collection is recorded in the index manifest. Automatic snapshot caching, automatic retention, explicit parser-derived edges, and edit overlays remain on the roadmap. `--memory none` (the default) keeps the original baseline without uploads.

The adapter follows the [HydraDB v2 OpenAPI contract](https://docs.hydradb.com/api-reference/v2/openapi.json) and [integration guide](https://docs.hydradb.com/llms.txt), using `API-Version: 2`. Some cookbook examples use older payload shapes; the implementation uses the current multipart contract.

## Reuse an existing repository graph (no ingestion)

Pass `--reuse-index` with a completed `index-manifest.json` from an earlier run.
This is an explicit retrieval-only pipeline: it never creates a database or
uploads/re-ingests sources. Without this flag, `--memory hydradb` still creates
a fresh collection as described above.

For the existing YogaIntelliJ graph in `hydra_swe_bench`, collection
`attempt_19fe9877a5504235920aaf66012525dc`, run from this project directory:

```sh
export DOCKER_CONTEXT=colima-hydra-swe
uv run hydra-agent run \
  --repo targets/YogaIntelliJ \
  --revision 2cc0eaa7ea5c07b75911a9eefc8484b94826036d \
  --memory hydradb \
  --reuse-index runs/19fe9877a5504235920aaf66012525dc/index-manifest.json \
  --task 'Use memory_search to explain the frontend routes and pose detection flow. Verify against local source. Do not edit files.'
```

This inspection task works without Node; frontend repair/tests still require a
Node/dependency-equipped image. Azure and HydraDB query charges still apply.
`HYDRA_DB_DATABASE` must match the saved manifest (`hydra_swe_bench` here).
The manifest must match the repository path, selected commit, source hashes,
and original source IDs. Each source's remote status must still be completed.
Missing/incomplete data fails closed without starting a rebuild.

Agent attempts keep separate traces and patches, but query the original
collection with its source-ID allowlist. Edited files remain excluded from
retrieval. A reused run's index manifest preserves the original indexing scope
and records `reused_from`; its run manifest records the new agent attempt.
This does not implement automatic caching or incremental indexing.

To query that graph directly without a coding model or Docker:

```sh
uv run python scripts/query_existing_graph.py \
  --repo targets/YogaIntelliJ \
  --reuse-index runs/19fe9877a5504235920aaf66012525dc/index-manifest.json \
  --query 'What frontend routes and pose detection components does YogaIntelliJ contain?'
```

## Live service smoke test without Docker

```sh
uv run python scripts/live_services_smoke.py --index-timeout 300
```

This uses the configured real Azure and HydraDB credentials. It creates the configured database if absent, uploads one synthetic document into a fresh attempt collection, waits for graph completion, and checks that Azure can retrieve its verification phrase through the agent's memory tool. It writes results under `runs/services-smoke-*` and retains the uploaded document. The test disables all model-generated command execution, so it verifies service integration and tool calling, not code repair or Docker isolation.

## Development checks

```sh
uv run pytest
uv run ruff check .
```

Tests use simulated model responses, mocked HTTP transport, and temporary Git repositories, with no live API calls. They cover ingestion readiness, multipart requests, partial failures, scope filtering, stale-file exclusion, and the full CLI path through a repair and patch export. Live Azure/HydraDB verification, Docker runtime validation, and the official SWE-bench adapter are tracked in `todo.md`.
