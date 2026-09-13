# Interactive chat verification

Live session: `runs/d5802a01c7ed4b61a13600c36d888467/`.

The real Azure deployment ran two conversational turns in one Docker workspace
using the existing YogaIntelliJ graph, collection
`attempt_19fe9877a5504235920aaf66012525dc` in `hydra_swe_bench`.

1. Asked the agent to find the main frontend pose-detection file through memory
   and verify the path locally. It queried HydraDB and inspected
   `frontend/src/pages/Yoga/Yoga.js`.
2. Asked which function estimates poses in "that same file," without repeating
   the path. With the previous conversation retained, it read relevant source
   and identified `detectPose`, which calls `detector.estimatePoses`.
3. `/status` reported two turns, 34,926 model-reported tokens and one memory search.
   `/exit` saved the conversation/session artifacts and zero-byte cumulative patch,
   then closed the sandbox normally.

Both turns ended with ordinary assistant text (`model_stopped`), which returns
control to the user in chat just as a finish call does. No edits, tests of the
target application, ingestion, or database creation were performed in this live
inspection session. This is not a repair benchmark result.

Offline tests cover multi-turn state, cumulative edits, patch export, context
clearing without budget reset, protocol completion after interruption, usage
reservation, terminal/credential redaction, EOF handling, and CLI startup without
an initial task. Opt-in Docker tests exercise the persistent edited workspace and
stopping background commands without destroying the container.
