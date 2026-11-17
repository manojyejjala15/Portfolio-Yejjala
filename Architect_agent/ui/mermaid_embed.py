"""HTML for rendering one Mermaid diagram inside a Streamlit component iframe."""
import json

# Pinned. Verified against the diagrams this project generates (flowchart + sequence).
MERMAID_VERSION = "11.14.0"
MERMAID_JS = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.esm.min.mjs"


def mermaid_html(code: str, js_url: str = MERMAID_JS, wait_until_visible: bool = True) -> str:
    """Return the iframe HTML.

    Streamlit keeps inactive tabs in the page with display:none, so a component iframe inside one
    has zero size when it loads, and Mermaid fails with "svg element not in render tree".
    With wait_until_visible the diagram is drawn only once the iframe has a real size.
    """
    # json.dumps gives a safe JS string literal; "</" is escaped so it can never close the script tag.
    source = json.dumps(code).replace("</", "<\\/")
    gate = "true" if wait_until_visible else "false"
    return f"""<div id="out" style="background:#fff;padding:12px;border-radius:8px;overflow:auto"></div>
<script type="module">
import mermaid from {json.dumps(js_url)};
const code = {source};
const out = document.getElementById("out");
mermaid.initialize({{ startOnLoad: false, theme: "neutral", securityLevel: "strict" }});

async function draw() {{
  try {{
    const {{ svg }} = await mermaid.render("diagram-" + Date.now(), code);
    out.innerHTML = svg;
  }} catch (e) {{
    const pre = document.createElement("pre");
    pre.style.cssText = "color:#b00020;white-space:pre-wrap;font:13px monospace";
    pre.textContent = "Diagram failed to render (mermaid {MERMAID_VERSION}): " + (e && e.message ? e.message : e);
    out.replaceChildren(pre);
  }}
}}

const visible = () => document.documentElement.clientWidth > 0 && window.innerHeight > 0;
if (!{gate} || visible()) {{
  draw();
}} else {{
  // Hidden (inactive tab): wait until the iframe is given a size, then draw once.
  const ro = new ResizeObserver(() => {{ if (visible()) {{ ro.disconnect(); draw(); }} }});
  ro.observe(document.documentElement);
}}
</script>"""
