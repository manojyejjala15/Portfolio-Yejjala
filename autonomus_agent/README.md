# Autonomous Agent Demo

A plan, execute, verify agent with typed tools, a step budget, injection sanitizing, write-gated memory, a 50-scenario eval harness, and a FastAPI + SSE streaming UI. It runs offline with a deterministic rule-based planner. Set `ANTHROPIC_API_KEY` to use Claude as the planner.

## Run it
Requires Python 3.10+.

```bash
cd agent-project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn api:app --reload           # then open http://localhost:8000
```

Click an example prompt, or try "Simulate tool outage" to watch the agent reroute and hand off.

## Other commands
```bash
python -m agent "Calculate 12*7+3"   # CLI: prints the event stream as JSON lines
python eval/run_eval.py              # 50-scenario eval; exits non-zero on any failure
curl -N -X POST localhost:8000/runs/stream -H 'Content-Type: application/json' -d '{"task":"Total sales by region"}'
```

## Optional config
| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Use Claude as the planner (falls back to rules on any error) |
| `AGENT_MODEL` | Planner model (default `claude-sonnet-4-6`) |
| `AGENT_API_KEY` | If set, clients must send `x-api-key` |
| `AGENT_MEMORY=off` | Disable long-term memory |

## Layout
```
api.py            FastAPI app (SSE streaming, serves the UI)
agent/core.py     plan/execute/verify loop + step budget
agent/tools.py    calculator, sql_query, search_docs + dispatcher
agent/llm.py      planner (rules or Claude)
agent/memory.py   write-gated long-term memory
static/index.html streaming timeline UI
eval/run_eval.py  50-scenario harness
docs/             architecture diagrams and design notes
```

## Extending
Add a tool in `agent/tools.py` (function + `TOOLS` entry), teach `llm.py` when to pick it, and add scenarios to `eval/run_eval.py`. To move to LangGraph, port `core.run` into nodes and keep `call_tool` as the single dispatcher.
