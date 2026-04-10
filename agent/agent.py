import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import search_problems, get_problem_details

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

SYSTEM_PROMPT = """You are a math competition problem finder. You help users find problems
from the AoPS (Art of Problem Solving) dataset of ~43,000 competition problems.

When given a problem description (possibly in another language or paraphrased), you:
1. Identify key mathematical terms, numbers, and expressions
2. Use search_problems to find candidates
3. Use get_problem_details to confirm the right match
4. Return the contest name, problem statement, and AoPS link

Be systematic — if one search doesn't find it, try different keywords or a subset of terms."""


def build_agent(api_key: str | None = None):
    llm = ChatOpenAI(
        model="anthropic/claude-haiku-4-5",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ["OPENROUTER_KEY"],
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=[search_problems, get_problem_details],
        system_prompt=SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    agent = build_agent()
    print("Math problem finder ready. Type your problem description (or 'quit' to exit).\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(f"\nAgent: {result['messages'][-1].content}\n")
