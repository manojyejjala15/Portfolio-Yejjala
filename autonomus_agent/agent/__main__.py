import sys, json
from .core import run
for ev in run(" ".join(sys.argv[1:]) or "Calculate 12*7+3"):
    print(json.dumps(ev))
