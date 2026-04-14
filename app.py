import os
import re
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

# Load secrets into environment variables before any imports that need them
for key in ("OPENROUTER_KEY", "HF_DATASET_REPO"):
    if key not in os.environ:
        val = st.secrets.get(key, "")
        if val:
            os.environ[key] = val

st.set_page_config(page_title="AoPS Problem Finder", page_icon="📐", layout="centered")

st.title("📐 AoPS Problem Finder")
st.caption("Describe a math competition problem and I'll find it in the AoPS dataset (~43k problems).")

EXAMPLE_PROBLEM = (
    "Let S be a finite nonempty set of positive integers such that for all i, j ∈ S "
    "(not necessarily distinct), the number (i+j)/gcd(i,j) also belongs to S. "
    "Find all such sets S."
)

# Pre-recorded agent stream for the example problem (avoids API call).
# Delays captured from a real agent run.
EXAMPLE_STEPS = [
    {"type": "thinking", "text": "I'll search for this problem in the AoPS dataset.", "delay": 5.0},
    {"type": "search", "terms": ["\\frac{i+j}{\\gcd(i,j)}", "finite nonempty set", "positive integers"], "delay": 0.1},
    {"type": "search", "terms": ["(i+j)/gcd", "belongs to S", "finite set"], "delay": 0.05},
    {"type": "search", "terms": ["gcd(i,j)", "i+j", "closure property"], "delay": 0.05},
    {"type": "result", "text": "3 result(s)", "delay": 1.0},
    {"type": "result", "text": "1 result(s)", "delay": 0.05},
    {"type": "result", "text": "5 result(s)", "delay": 0.05},
    {"type": "thinking", "text": "Perfect! I found the problem. Let me get the full details:", "delay": 4.7},
    {"type": "details", "row_index": 2942, "delay": 0},
    {"type": "result", "text": "APMO", "delay": 5.0},
]
EXAMPLE_RESPONSE = (
    "## Problem Found\n\n"
    "This problem appeared in:\n\n"
    "**[2004 APMO](https://artofproblemsolving.com/community/c4121_2004_apmo)**\n"
    "- Problem link: https://artofproblemsolving.com/community/c6h82828p474988\n\n"
    "The problem asks to determine all finite nonempty sets $S$ of positive integers "
    "such that for all $i, j \\in S$, the quantity $\\frac{i+j}{\\gcd(i,j)}$ also belongs to $S$."
)


def replay_example(status_container):
    """Replay pre-recorded agent steps with realistic timing."""
    for step in EXAMPLE_STEPS:
        time.sleep(step.get("delay", 0.3))
        if step["type"] == "thinking":
            status_container.caption(step["text"])
        elif step["type"] == "search":
            status_container.markdown("🔍 " + ", ".join(f"`{t}`" for t in step["terms"]))
        elif step["type"] == "details":
            status_container.markdown(f"📋 Reading problem **#{step['row_index']}**")
        elif step["type"] == "result":
            status_container.caption(f"→ {step['text']}")
    return EXAMPLE_RESPONSE

@st.cache_resource(show_spinner="Loading dataset and agent (first time may take a minute)...")
def load_agent():
    from tools import get_df
    get_df()
    from agent import build_agent
    return build_agent()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Chat input (rendered early so we know if user typed something)
# ---------------------------------------------------------------------------

chat_input = st.chat_input("Describe a problem (any language)...")
prompt = st.session_state.pending_prompt or chat_input
st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Example problem (only when no conversation yet and nothing typed)
# ---------------------------------------------------------------------------

if not st.session_state.messages and not prompt:
    with st.container(border=True):
        st.markdown(f"🎲 **Combinatorics**")
        st.markdown(EXAMPLE_PROBLEM)
        if st.button("Search this problem", key="example", use_container_width=True):
            st.session_state.pending_prompt = EXAMPLE_PROBLEM
            st.rerun()

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = None
        is_example = prompt == EXAMPLE_PROBLEM and len(st.session_state.messages) == 1

        try:
            with st.status("Searching...", expanded=True) as status:
                if is_example:
                    response = replay_example(status)
                else:
                    agent = load_agent()
                    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]}):
                        if "model" in chunk:
                            msg = chunk["model"]["messages"][-1]
                            tool_calls = getattr(msg, "tool_calls", None)
                            if tool_calls:
                                if msg.content:
                                    st.caption(msg.content)
                                for tc in tool_calls:
                                    name = tc["name"]
                                    args = tc["args"]
                                    if name == "search_problems":
                                        terms = args.get("terms", [])
                                        st.markdown("🔍 " + ", ".join(f"`{t}`" for t in terms))
                                    elif name == "search_by_contest":
                                        st.markdown(f"🏆 Contest: `{args.get('contest_name', '')}`")
                                    elif name == "get_problem_details":
                                        st.markdown(f"📋 Reading problem **#{args.get('row_index', '')}**")
                            elif msg.content:
                                response = msg.content
                        elif "tools" in chunk:
                            msg = chunk["tools"]["messages"][-1]
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            if "No problems found" in content:
                                st.caption("→ No matches")
                            else:
                                matches = re.findall(r"^\[(\d+)\] (.+)$", content, re.MULTILINE)
                                if matches:
                                    st.caption(f"→ {len(matches)} result(s)")
                                elif content.startswith("Contest:"):
                                    m = re.match(r"Contest:\s*(.+)", content)
                                    if m:
                                        st.caption(f"→ {m.group(1).strip()}")
                status.update(label="Search complete", state="complete", expanded=False)
        except Exception as e:
            if "402" in str(e):
                response = "The search service is temporarily unavailable. Please try again later."
            else:
                response = "Something went wrong while searching. Please try again."

        if response:
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response or ""})
