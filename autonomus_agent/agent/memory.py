"""Tiny long-term memory: verified Q/A pairs per user, keyword recall, write-gated by the verifier."""
import json, os, re, time
ENABLED = os.getenv("AGENT_MEMORY", "on") != "off"
PATH = os.path.join(os.path.dirname(__file__), "..", "data", "memory.json")
_w = lambda s: {w for w in re.findall(r"[a-z]{4,}", s.lower())}

def _load():
    try: return json.load(open(PATH))
    except Exception: return {}

def recall(user, task, k=3):
    if not ENABLED: return []
    q = _w(task)
    hits = [e for e in _load().get(user, []) if len(q & _w(e["q"])) >= 2]
    return [e["q"] for e in hits[-k:]]

def remember(user, q, a):
    if not ENABLED: return
    db = _load(); items = db.setdefault(user, [])
    if all(e["q"] != q for e in items):          # dedupe
        items.append({"q": q, "a": a[:300], "ts": int(time.time())})
        json.dump(db, open(PATH, "w"))
