"""Streamlit front end. Run from the project root: streamlit run ui/streamlit_app.py"""
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components

from architect_agent import AgentError, ArchitectAgent, Settings
from architect_agent.guardrails import InputRejected
from architect_agent.llm import ConfigError, TransientLLMError, build_client
from ui.mermaid_embed import mermaid_html
from ui.prompt_guide import SECTIONS, TEMPLATE, TIPS, coverage, has_unfilled_placeholders

SAMPLE = (ROOT / "sample_projects" / "retail_forecasting.md").read_text()
ICONS = {"started": "⏳", "ok": "✅", "error": "❌"}

st.set_page_config(page_title="Technical Architect Agent", page_icon="🏗️", layout="wide")


def render_mermaid(code: str, height: int = 480) -> None:
    components.html(mermaid_html(code), height=height, scrolling=True)


# ---- sidebar: settings -------------------------------------------------------
base = Settings.from_env()
with st.sidebar:
    st.header("Settings")
    has_key = bool(base.api_key)
    use_mock = st.checkbox("Use offline mock LLM (no API key needed)", value=base.use_mock or not has_key)
    if not has_key and not use_mock:
        st.warning("ANTHROPIC_API_KEY is not set.")
    model = st.text_input("Model", base.model, disabled=use_mock)
    rounds = st.slider("Max review rounds", 1, 4, base.max_review_rounds)
    st.caption("The mock always returns the sample retail design, so use a real model for your own projects.")

# ---- input -------------------------------------------------------------------
st.title("🏗️ Technical Architect Agent")
st.write("Describe a project. The agent extracts requirements, designs an architecture, "
         "reviews and revises it, then produces diagrams and a Markdown document.")

st.session_state.setdefault("desc", "")
st.session_state.setdefault("name", "My Project")


def load_sample() -> None:
    st.session_state.desc = SAMPLE
    st.session_state.name = "Retail Demand Forecasting Platform"


def load_template() -> None:
    st.session_state.desc = TEMPLATE


# ---- prompt guide ------------------------------------------------------------
with st.expander("📝 How to write a good project description",
                 expanded=not st.session_state.desc.strip()):
    st.markdown("The agent only knows what you tell it. The more concrete your description, the better "
                "the design, and anything you leave out is listed as an open question or an assumption.")
    rows = ["| Section | What to include | Example |", "|---|---|---|"]
    for sec in SECTIONS:
        label = sec.title + (" *(optional)*" if sec.optional else "")
        rows.append(f"| **{label}** | {sec.include} | {sec.example} |")
    st.markdown("\n".join(rows))
    st.markdown("**Tips**\n" + "\n".join(f"- {t}" for t in TIPS))
    st.markdown("**Template** (use the button below, or copy it with the icon in the corner)")
    st.code(TEMPLATE, language="markdown")

col_a, col_b, col_c = st.columns([3, 1, 1])
col_a.text_input("Project name", key="name")
col_b.button("Insert template", on_click=load_template, use_container_width=True)
col_c.button("Load sample project", on_click=load_sample, use_container_width=True)
st.text_area("Project description", key="desc", height=260,
             placeholder="Goal, core features, scale and performance targets, constraints, "
                         "integrations, security... (open the guide above for a template)")

# ---- live feedback on the description ----------------------------------------
_text = st.session_state.desc
_n = len(_text.strip())
_limits = f"{_n:,} / {base.max_input_chars:,} characters (minimum {base.min_input_chars})"
if _n:
    _marks = "  ".join(f"{'✅' if ok else '⬜'} {sec.title}" for sec, ok in coverage(_text))
    st.caption(f"{_limits}  \n{_marks}  \nChecklist is a hint based on keywords; it updates when you "
               "click outside the box.")
    if has_unfilled_placeholders(_text):
        st.warning("The description still contains `<placeholder>` lines from the template. "
                   "Replace or delete them before generating.")
else:
    st.caption(_limits)

if st.button("Generate architecture", type="primary"):
    settings = dataclasses.replace(base, use_mock=use_mock, model=model, max_review_rounds=rounds)
    try:
        with st.status("Running architect agent...", expanded=True) as status:
            def on_event(e):
                if e.status != "started":
                    st.write(f"{ICONS[e.status]} **{e.stage}** {e.duration_ms} ms {e.detail}")

            agent = ArchitectAgent(build_client(settings), settings, on_event)
            st.session_state.result = agent.run(st.session_state.desc, st.session_state.name)
            status.update(label="Done", state="complete", expanded=False)
    except (InputRejected, ConfigError) as exc:
        st.error(str(exc))
    except TransientLLMError:
        st.error("The model service is temporarily unavailable. Please try again in a minute.")
    except AgentError as exc:
        st.error(f"The agent could not produce a valid design: {exc}")
    except Exception:
        st.error("Unexpected error. Check the server logs for details.")
        raise

# ---- results -----------------------------------------------------------------
result = st.session_state.get("result")
if result:
    if result.redactions:
        st.info(f"{result.redactions} possible secret(s) or email(s) were redacted before analysis.")
    if not result.approved:
        st.warning("The reviewer did not fully approve this design. See the Review tab.")
    m = st.columns(4)
    m[0].metric("Review score", f"{result.review.score}/10")
    m[1].metric("Review rounds", result.review_rounds)
    m[2].metric("LLM calls", result.usage.llm_calls)
    m[3].metric("Tokens", result.usage.input_tokens + result.usage.output_tokens)

    t_sum, t_req, t_arch, t_dia, t_rev, t_trace = st.tabs(
        ["Summary", "Requirements", "Architecture", "Diagrams", "Review", "Trace"])
    with t_sum:
        st.subheader(result.design.style)
        st.write(result.design.overview)
        st.download_button("⬇ Download architecture.md", result.markdown,
                           file_name="architecture.md", mime="text/markdown")
    with t_req:
        st.write(result.requirements.summary)
        st.dataframe([vars(r) for r in result.requirements.functional], hide_index=True, use_container_width=True)
        st.dataframe([vars(r) for r in result.requirements.non_functional], hide_index=True, use_container_width=True)
        for label, items in (("Constraints", result.requirements.constraints),
                             ("Assumptions", result.requirements.assumptions),
                             ("Open questions", result.requirements.open_questions)):
            st.markdown(f"**{label}**\n" + "\n".join(f"- {i}" for i in items))
    with t_arch:
        st.dataframe([{**vars(c), "depends_on": ", ".join(c.depends_on)} for c in result.design.components],
                     hide_index=True, use_container_width=True)
        for a in result.design.adrs:
            with st.expander(f"ADR: {a.title}"):
                st.write(f"**Decision:** {a.decision}\n\n**Rationale:** {a.rationale}")
        st.dataframe([vars(r) for r in result.design.risks], hide_index=True, use_container_width=True)
    with t_dia:
        for title, code in result.diagrams.items():
            st.subheader(title)
            render_mermaid(code)
            with st.expander("Mermaid source"):
                st.code(code, language="text")
    with t_rev:
        st.dataframe([vars(i) for i in result.review.issues], hide_index=True, use_container_width=True)
    with t_trace:
        st.caption(f"Run id: {result.run_id}")
        st.dataframe([vars(e) for e in result.trace if e.status != "started"], hide_index=True,
                     use_container_width=True)
