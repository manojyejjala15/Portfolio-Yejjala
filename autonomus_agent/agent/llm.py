"""Planner. Rule-based by default (offline, deterministic); uses Claude if ANTHROPIC_API_KEY is set."""
import json, os, re

def _rules(task, failed):
    t = task.lower()
    if re.search(r"\d\s*[-+*/x^%]\s*\d|calculate|compute", t):
        expr = max(re.findall(r"[\d\.\s\+\-\*/\(\)x]{3,}", task), key=len, default="").strip().replace("x", "*").replace("^", "**")
        step = {"tool": "calculator", "args": {"expression": expr}}
    elif any(w in t for w in ("sales", "revenue", "orders", "region")):
        step = {"tool": "sql_query", "args": {"query": "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC"}}
    else:
        step = {"tool": "search_docs", "args": {"query": task}}
    if step["tool"] in failed:                        # fall back once, then give up
        step = {"tool": "search_docs", "args": {"query": task}}
    return [] if step["tool"] in failed else [step]

def _claude(task, failed):
    import anthropic
    sys = ("You are a planner. Tools: calculator(expression), sql_query(query; table sales(region,product,amount,month); SELECT only), "
           f"search_docs(query). Unavailable: {sorted(failed)}. Reply ONLY JSON: {{\"tool\": name|null, \"args\": {{...}}}}")
    m = anthropic.Anthropic().messages.create(model=os.getenv("AGENT_MODEL", "claude-sonnet-4-6"), max_tokens=300,
                                              system=sys, messages=[{"role": "user", "content": task}])
    d = json.loads(re.search(r"\{.*\}", m.content[0].text, re.S).group())
    return [] if not d.get("tool") or d["tool"] in failed else [d]

def plan(task, failed):
    if os.getenv("ANTHROPIC_API_KEY"):
        try: return _claude(task, failed)
        except Exception: pass                         # degrade gracefully to rules
    return _rules(task, failed)
