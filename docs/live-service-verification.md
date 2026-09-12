# Live service verification — 2026-09-13

Azure endpoint: `https://model-apis.services.ai.azure.com/openai/v1/`.
Deployment: `grok-4.3`. Credentials are loaded from the ignored `.env`; the example file contains no keys.

Verified against real services:

- Azure Chat Completions returned a successful response.
- Azure accepted the harness's strict function schemas and returned a `finish` tool call.
- HydraDB authenticated requests, created database `hydra_swe_bench`, and provisioned its infrastructure.
- Multipart ingestion accepted one synthetic README in a unique test collection. Polling continued through indexing until completion.
- Direct retrieval returned the uploaded verification phrase.
- The real agent loop called `memory_search`, received HydraDB evidence, and submitted the expected phrase and addition contract.

The agent portion took 2 model steps, 1,951 service-reported tokens, and 8.573 seconds. This excludes database provisioning, graph indexing, the direct retrieval check, and the earlier connectivity calls. HydraDB's internal model usage and total monetary cost were not measured.

Artifacts: [result](../runs/services-smoke-356e40bcea50499389bbdfedad36ee7f/result.json), [manifest](../runs/services-smoke-356e40bcea50499389bbdfedad36ee7f/manifest.json), [trajectory](../runs/services-smoke-356e40bcea50499389bbdfedad36ee7f/trajectory.jsonl). Run artifacts are ignored by Git and remain local.

The test used `scripts/live_services_smoke.py`, which disables model-generated command execution. It does not establish code-repair accuracy, patch applicability for a live model, Docker isolation, or cross-tenant graph isolation. Only the Docker CLI was found on this machine; no working engine, Docker Desktop, OrbStack, or Colima was available at the checked locations.

The database and synthetic source remain stored in collection `attempt_356e40bcea50499389bbdfedad36ee7f`. No user repository was uploaded in this test. A full isolated repair test still needs a Docker-compatible runtime and the sandbox image.
