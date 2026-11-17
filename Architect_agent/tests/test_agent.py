import unittest

from architect_agent import ArchitectAgent, AgentError, Settings
from architect_agent.diagrams import component_diagram
from architect_agent.guardrails import InputRejected, sanitize_input
from architect_agent.llm import LLMResponse, TransientLLMError
from architect_agent.mock_llm import MockLLM
from architect_agent.schemas import ArchitectureDesign, SchemaError

DESC = "Retail demand forecasting platform with a Streamlit dashboard and a Python backend."
FAST = Settings(retry_base_delay_s=0.0)


class Flaky:
    """Wraps MockLLM and injects failures: bad JSON and transient errors."""

    def __init__(self, bad_json=0, transient=0):
        self.inner, self.bad_json, self.transient = MockLLM(), bad_json, transient
        self.calls = 0

    def complete(self, stage, system, user, max_tokens=4000):
        self.calls += 1
        if self.transient > 0:
            self.transient -= 1
            raise TransientLLMError("503")
        if self.bad_json > 0 and stage == "requirements":
            self.bad_json -= 1
            return LLMResponse("Sure! here you go: not json")
        return self.inner.complete(stage, system, user, max_tokens)


class AgentTests(unittest.TestCase):
    def test_full_run_with_review_loop(self):
        result = ArchitectAgent(MockLLM(), FAST).run(DESC, "Demo")
        self.assertTrue(result.approved)
        self.assertEqual(result.review_rounds, 2)          # round 1 rejected, round 2 approved
        names = [c.name for c in result.design.components]
        self.assertIn("Forecast Cache", names)             # revision applied reviewer feedback
        self.assertIn("```mermaid", result.markdown)
        self.assertIn("flowchart LR", result.diagrams["Component architecture"])
        self.assertTrue(any(k.startswith("Flow:") for k in result.diagrams))
        self.assertEqual(result.usage.llm_calls, 5)        # req + 2x(design+review)

    def test_unapproved_after_max_rounds_is_flagged(self):
        result = ArchitectAgent(MockLLM(), Settings(retry_base_delay_s=0.0, max_review_rounds=1)).run(DESC, "Demo")
        self.assertFalse(result.approved)
        self.assertIn("NOT approved", result.markdown)

    def test_repairs_malformed_json(self):
        llm = Flaky(bad_json=1)
        result = ArchitectAgent(llm, FAST).run(DESC)
        self.assertTrue(result.approved)

    def test_gives_up_after_repair_attempts(self):
        with self.assertRaises(AgentError):
            ArchitectAgent(Flaky(bad_json=99), FAST).run(DESC)

    def test_retries_transient_errors(self):
        llm = Flaky(transient=2)
        result = ArchitectAgent(llm, FAST).run(DESC)
        self.assertTrue(result.approved)
        self.assertEqual(llm.calls, 5 + 2)

    def test_transient_errors_exhausted(self):
        with self.assertRaises(TransientLLMError):
            ArchitectAgent(Flaky(transient=99), FAST).run(DESC)

    def test_trace_records_every_stage(self):
        result = ArchitectAgent(MockLLM(), FAST).run(DESC)
        stages = [e.stage for e in result.trace if e.status == "ok"]
        self.assertEqual(stages[0], "validate_input")
        self.assertEqual(stages[-1], "diagrams_and_markdown")

    def test_runs_do_not_share_state(self):
        agent = ArchitectAgent(MockLLM(), FAST)
        a, b = agent.run(DESC), agent.run(DESC)
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertEqual(a.usage.llm_calls, 5)


class GuardrailTests(unittest.TestCase):
    def test_rejects_short_and_long(self):
        with self.assertRaises(InputRejected):
            sanitize_input("too short", 30, 100)
        with self.assertRaises(InputRejected):
            sanitize_input("x" * 101, 30, 100)

    def test_redacts_secrets(self):
        text = "Build an app. Contact bob@example.com, password: hunter2, key sk-abcdefghijklmnopqrstuvwxyz"
        clean, n = sanitize_input(text, 10, 500)
        self.assertEqual(n, 3)
        self.assertNotIn("hunter2", clean)
        self.assertNotIn("bob@example.com", clean)


class SchemaTests(unittest.TestCase):
    def test_unknown_dependency_rejected(self):
        bad = {"style": "x", "overview": "y", "components": [
            {"name": "A", "layer": "backend", "responsibility": "r", "tech": "t", "depends_on": ["B"]}]}
        with self.assertRaises(SchemaError):
            ArchitectureDesign.from_dict(bad)

    def test_diagram_survives_awkward_names(self):
        d = ArchitectureDesign.from_dict({"style": "x", "overview": "y", "components": [
            {"name": 'Auth "Gateway" (v2)', "layer": "backend", "responsibility": "r", "tech": "C#; <Go>"}]})
        out = component_diagram(d)
        self.assertNotIn(";", out.split("\n", 1)[1])
        self.assertNotIn("<Go>", out)


if __name__ == "__main__":
    unittest.main()
