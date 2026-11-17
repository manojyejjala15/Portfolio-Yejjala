"""Plan -> execute -> verify loop with a step budget, streaming events as it goes."""
import time
from . import llm, tools, memory
MAX_STEPS = 4

def run(task, user_id="demo", fail=None):
    fail, failed, called, t0 = dict(fail or {}), set(), [], time.time()
    verdict, answer, steps = "fail", None, 0
    yield {"event": "start", "task": task}
    hits = memory.recall(user_id, task)
    if hits: yield {"event": "memory", "hits": hits}
    while steps < MAX_STEPS:
        steps += 1
        yield {"event": "node", "name": "planner", "step": steps}
        plan = llm.plan(task, failed)
        if not plan:
            yield {"event": "node", "name": "handoff", "reason": "no remaining strategy"}; break
        step = plan[0]
        yield {"event": "node", "name": "executor", "step": steps}
        res = tools.call_tool("executor", step["tool"], step["args"], fail)
        called.append(step["tool"])
        if not res["ok"]: failed.add(step["tool"])
        yield {"event": "tool", "name": step["tool"], "args": step["args"], "ok": res["ok"],
               "output": str(res["data"] if res["ok"] else res["error"])[:500], "latency_ms": res["latency_ms"]}
        yield {"event": "node", "name": "verifier", "step": steps}
        if res["ok"] and res["data"]:
            verdict, answer = "pass", tools.render(step["tool"], res["data"]); break
    else:
        yield {"event": "node", "name": "handoff", "reason": "step budget exhausted"}
    if verdict == "pass": memory.remember(user_id, task, answer)       # write-gated on verifier pass
    yield {"event": "done", "verdict": verdict, "steps": steps, "tools": called, "latency_s": round(time.time() - t0, 3),
           "answer": answer or "I couldn't complete this reliably, so I'm handing off to a human."}
