import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

st.set_page_config(page_title="AoPS Problem Finder", page_icon="📐", layout="centered")

st.title("📐 AoPS Problem Finder")
st.caption("Describe a math competition problem and I'll find it in the AoPS dataset.")


@st.cache_resource(show_spinner="Loading dataset and agent...")
def load_agent():
    from agent import build_agent
    return build_agent()


def get_api_key() -> str:
    # Streamlit Cloud secrets take priority, then .env via environment
    return st.secrets.get("OPENROUTER_KEY", os.environ.get("OPENROUTER_KEY", ""))


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Describe a problem (any language)..."):
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
                response = f"Error: {e}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
