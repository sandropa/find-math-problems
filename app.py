import os
import re
import sys
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
# Example problem (only when no conversation yet)
# ---------------------------------------------------------------------------

if not st.session_state.messages and not st.session_state.pending_prompt:
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

# ---------------------------------------------------------------------------
# Handle pending prompt (from example card click) or typed input
# ---------------------------------------------------------------------------

chat_input = st.chat_input("Describe a problem (any language)...")
prompt = st.session_state.pending_prompt or chat_input
st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = None
        try:
            agent = load_agent()
            with st.status("Searching...", expanded=True) as status:
                for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]}):
                    if "model" in chunk:
                        msg = chunk["model"]["messages"][-1]
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            # Agent is calling tools — show what it's thinking and doing
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
                            # Final response
                            response = msg.content
                    elif "tools" in chunk:
                        msg = chunk["tools"]["messages"][-1]
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if "No problems found" in content:
                            st.caption("→ No matches")
                        else:
                            # Count result entries like [12345]
                            matches = re.findall(r"^\[(\d+)\] (.+)$", content, re.MULTILINE)
                            if matches:
                                st.caption(f"→ {len(matches)} result(s)")
                            elif content.startswith("Contest:"):
                                # get_problem_details response — extract contest name
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
