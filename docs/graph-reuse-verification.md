# Existing graph retrieval verification

Verified the explicit retrieval-only pipeline against database `hydra_swe_bench`,
collection `attempt_19fe9877a5504235920aaf66012525dc`. The saved manifest describes
38 sources at YogaIntelliJ commit `2cc0eaa7ea5c07b75911a9eefc8484b94826036d`.
The local snapshot regenerated matching source IDs and hashes under the original
indexing scope; the retrieval session used a new attempt scope.

The live direct-query script made exactly these successful requests:

| Request | Request ID |
| --- | --- |
| GET /context/status (first batch) | `4b354d70-a317-42d5-8e6e-a086ef218846` |
| GET /context/status (second batch) | `6151e14c-1e80-49bf-a3dc-4b6795990608` |
| POST /query | `ba058ccf-4a9c-4fb8-9ceb-c8bdd1700fc7` |

All 38 sources reported completed. Readiness took 1.295 seconds; the query took
4.979 seconds and returned 7 allowed evidence chunks. No source ingestion,
database creation, coding-model call, or source modification occurred.

The adapter enforces a request allowlist in retrieval-only mode. Offline tests
also cover invalid/mismatched manifests, incomplete/missing remote sources,
original source IDs, changed-file exclusion, and the full CLI reuse path with a
simulated coding model. This verification is not a real model-driven code repair
or a benchmark result. Remote source IDs are assumed immutable; status checks do
not independently rehash server-side contents.
