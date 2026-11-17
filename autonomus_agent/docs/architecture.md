# Architecture

```mermaid
flowchart LR
    UI[Browser UI] -- SSE --> API[FastAPI api.py]
    API --> CORE[agent/core.py loop]
    CORE --> LLM[llm.py planner: rules or Claude]
    CORE --> TL[tools.py dispatcher]
    TL --> CALC[calculator]
    TL --> SQL[(sqlite sales)]
    TL --> KB[(doc search)]
    CORE <--> MEM[(memory.json)]
```

```mermaid
stateDiagram-v2
    [*] --> Planner
    Planner --> Executor: plan
    Planner --> Handoff: no strategy left
    Executor --> Verifier: tool result
    Verifier --> [*]: pass (write memory)
    Verifier --> Planner: retry, steps < 4
    Verifier --> Handoff: budget exhausted
    Handoff --> [*]
```

## Design decisions
- **Verifier gates exit and memory writes.** The executor cannot declare success.
- **Errors are data.** The dispatcher returns `{ok, error}`; the planner reroutes around failed tools.
- **Retries only for transient errors.** Deterministic errors (bad args, no docs) fail fast.
- **Retrieved text is untrusted.** Injection-like sentences are stripped before they reach the answer.
- **Hard step budget (4).** Prevents runaway loops; exhaustion triggers human handoff.

## Known limitations
Single-tool plans only; keyword search stands in for vector/graph retrieval; no auth beyond an optional API key; memory is a local JSON file.
