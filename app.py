import os
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
        with st.spinner("Searching..."):
            try:
                agent = load_agent()
                result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                response = result["messages"][-1].content
            except Exception as e:
                if "402" in str(e):
                    response = "The search service is temporarily unavailable. Please try again later."
                else:
                    response = "Something went wrong while searching. Please try again."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
