import json, os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from agent import run

app = FastAPI(title="Autonomous Agent Demo")
API_KEY = os.getenv("AGENT_API_KEY")        # optional; if set, clients must send x-api-key

class RunRequest(BaseModel):
    task: str = Field(min_length=3, max_length=2000)
    simulate_failure: bool = False

def sse(ev): return f"event: {ev['event']}\ndata: {json.dumps(ev)}\n\n"

@app.post("/runs/stream")
def stream(req: RunRequest, x_api_key: str | None = Header(None)):
    if API_KEY and x_api_key != API_KEY: raise HTTPException(401, "bad key")
    fail = {"calculator": 99, "sql_query": 99} if req.simulate_failure else None
    def gen():
        try:
            for ev in run(req.task, fail=fail): yield sse(ev)
        except Exception as e:
            yield sse({"event": "error", "message": f"Run failed: {type(e).__name__}"})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/")
def index(): return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
