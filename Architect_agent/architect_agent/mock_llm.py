"""Offline LLM stand-in for demos and tests. Returns a canned design for the sample project.

Round 1 of the review rejects the design (no caching); the second design adds a cache
and the second review approves, so the full review/revise loop is exercised without an API key.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

from .llm import LLMResponse

_REQS = {
    "summary": "A web platform that forecasts weekly product demand per store and recommends reorder quantities.",
    "functional": [
        {"id": "FR-1", "text": "Ingest daily sales from store POS systems", "priority": "must"},
        {"id": "FR-2", "text": "Generate 4-week demand forecasts per SKU and store", "priority": "must"},
        {"id": "FR-3", "text": "Show forecasts and reorder suggestions in a dashboard", "priority": "must"},
    ],
    "non_functional": [
        {"id": "NFR-1", "text": "Dashboard p95 response under 2 s", "priority": "must"},
        {"id": "NFR-2", "text": "Nightly pipeline finishes within 2 hours", "priority": "should"},
    ],
    "constraints": ["Python backend", "Streamlit frontend"],
    "assumptions": ["About 200 stores and 20,000 SKUs", "POS exports are available as daily files or an API"],
    "open_questions": ["Is real-time inventory needed, or is daily sufficient?"],
}


def _comp(name, layer, resp, tech, deps=()):
    return {"name": name, "layer": layer, "responsibility": resp, "tech": tech, "depends_on": list(deps)}


_DESIGN = {
    "style": "Modular monolith API plus scheduled batch pipeline",
    "overview": "A FastAPI service serves forecasts to a Streamlit dashboard. A nightly Airflow pipeline "
                "loads POS data into PostgreSQL, builds features, trains models and publishes forecasts.",
    "components": [
        _comp("Dashboard", "frontend", "Forecast and reorder views", "Streamlit", ["API Service"]),
        _comp("API Service", "backend", "AuthN/Z, forecast and inventory endpoints", "FastAPI", ["Forecast Store"]),
        _comp("Forecast Engine", "backend", "Trains models and writes forecasts", "Python, scikit-learn, XGBoost",
              ["Feature Tables", "Model Registry", "Forecast Store"]),
        _comp("POS Connector", "integration", "Pulls daily sales from POS exports", "Python, boto3"),
        _comp("Pipeline Scheduler", "integration", "Orchestrates nightly ingest and training", "Airflow",
              ["POS Connector", "Feature Tables", "Forecast Engine"]),
        _comp("Feature Tables", "data", "Cleaned sales and calendar features", "PostgreSQL"),
        _comp("Forecast Store", "data", "Published forecasts and reorder suggestions", "PostgreSQL"),
        _comp("Model Registry", "data", "Versioned model artifacts and metrics", "MLflow"),
        _comp("Monitoring", "infra", "Metrics, logs and alerts", "Prometheus, Grafana"),
    ],
    "key_flows": [
        {"name": "View forecast", "steps": [
            {"source": "User", "target": "Dashboard", "message": "Select store and SKU"},
            {"source": "Dashboard", "target": "API Service", "message": "GET /forecasts"},
            {"source": "API Service", "target": "Forecast Store", "message": "Query latest forecast"},
            {"source": "API Service", "target": "Dashboard", "message": "Forecast JSON"}]},
        {"name": "Nightly pipeline", "steps": [
            {"source": "Pipeline Scheduler", "target": "POS Connector", "message": "Fetch daily sales"},
            {"source": "POS Connector", "target": "Feature Tables", "message": "Upsert cleaned sales"},
            {"source": "Pipeline Scheduler", "target": "Forecast Engine", "message": "Train and predict"},
            {"source": "Forecast Engine", "target": "Forecast Store", "message": "Write forecasts"}]},
    ],
    "deployment": ["Docker images per service on a managed container platform",
                   "Managed PostgreSQL with daily backups", "CI pipeline runs tests and builds images"],
    "adrs": [{"title": "Batch forecasting instead of streaming",
              "decision": "Train and predict nightly",
              "rationale": "Reorders are planned daily; batch is simpler and cheaper",
              "alternatives": ["Streaming with Kafka"]}],
    "risks": [{"description": "POS data arrives late or malformed",
               "mitigation": "Schema validation, data-quality checks and alerts before training"}],
}

_REVIEWS = [
    {"score": 6, "approved": False, "issues": [
        {"severity": "medium", "description": "No caching, so dashboard reads may miss the 2 s target",
         "suggestion": "Add a read cache in front of the Forecast Store"}]},
    {"score": 8, "approved": True, "issues": [
        {"severity": "low", "description": "Define data retention for old forecasts",
         "suggestion": "Archive forecasts older than 13 months"}]},
]


class MockLLM:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def complete(self, stage: str, system: str, user: str, max_tokens: int = 4000) -> LLMResponse:
        self.calls[stage] += 1
        n = self.calls[stage]
        if stage == "requirements":
            data = _REQS
        elif stage == "design":
            data = copy.deepcopy(_DESIGN)
            if n >= 2:  # revision: add the cache the reviewer asked for
                data["components"].append(_comp("Forecast Cache", "data", "Read-through cache for hot forecasts", "Redis"))
                for c in data["components"]:
                    if c["name"] == "API Service":
                        c["depends_on"].append("Forecast Cache")
                data["adrs"].append({"title": "Add Redis read cache", "decision": "Cache forecast reads for 15 minutes",
                                     "rationale": "Meets the 2 s p95 target under dashboard load",
                                     "alternatives": ["Materialized views"]})
        elif stage == "review":
            data = _REVIEWS[min(n, len(_REVIEWS)) - 1]
        else:
            raise ValueError(f"unknown stage {stage!r}")
        text = json.dumps(data)
        return LLMResponse(text, input_tokens=len(user) // 4, output_tokens=len(text) // 4)
