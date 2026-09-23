# CodeWikiBench — HydraDB documentation agent

Dataset: `anhnh2002/codewikibench` at `6d215eb7d50a164e370a9a5703b813f9da345965`.
Published judge source: `5e728fb40492effb54d59041f908dbf9079fe238`.

This is a selected-repository run using the published rubric prompt and hierarchical weights with an adapted Azure tool runner. It is not a full-suite result unless all 22 repositories finish. No baseline or isolated graph ablation was run.

| Repository | Indexed sources | Wiki pages | Judged criteria | Coverage / 100 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Chart.js | 8/229 | 0 | 0/0 | — | interrupted |

Completed repositories: 0/1.

## Interpretation

Scores measure whether generated documentation covers the published criteria. They do not prove claims or graph edges are correct. Source-link validation checks paths and line ranges only. Mermaid blocks are exported but not renderer-validated.

Generation sees only eligible code/build/test sources. Reference docs, benchmark rubrics, prose files, visual fixtures and lockfiles are excluded from agent inputs. The original snapshot and reference material stay in controller/evaluator artifacts; the agent has no filesystem or shell tool. Source comments remain available.

Generator and judge deployments are recorded per repository. Using the same model for both introduces possible self-evaluation bias. Model token usage is recorded; HydraDB internal token usage and dollar cost are not exposed by this harness.

Judge errors remain unscored and block an overall score. Re-running the same command resumes index status checks, completed pages and completed judgments.

- [Chart.js generated wiki](Chart.js/wiki/README.md)
- [Chart.js evaluation](Chart.js/evaluation/result.json)
- [Chart.js index manifest](Chart.js/index-manifest.json)
