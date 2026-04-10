import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import search_problems, search_by_contest, get_problem_details

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

SYSTEM_PROMPT = """You are a math competition problem finder. You search a dataset of ~43,000 AoPS problems.

YOUR ONLY JOB: Find problems and return their AoPS links. That's it.

OUTPUT RULES (strict):
- List each contest where the problem appeared, with its AoPS link
- The tool output already contains all contest names and links — just relay them faithfully
- DO NOT add problem numbers (like "G2", "P3") unless they appear in the tool output
- DO NOT provide solutions, solution sketches, or hints
- DO NOT add information that isn't in the tool output — no hallucinating details
- Keep your response concise: contest names with links, and optionally a brief problem statement

SEARCH STRATEGY:
The dataset stores problems as HTML with LaTeX in <img alt="..."> tags.
You MUST search using LaTeX expressions and English math terms — NEVER use words in other languages.

When given a problem (in any language):
1. TRANSLATE the math content to identify the key equations and constraints
2. Convert them to LaTeX search terms: "ac=bd", "\\frac{a}{b}", "x^2-1", "\\geq"
3. Also use English math keywords: "maximize", "integer", "triangle", "prime"
4. Search with 2-4 specific terms that uniquely identify the problem

SEARCH TIPS:
- Equations like ac=bd appear as: ac=bd in alt attributes
- Fractions like a/b appear as: \\frac{a}{b}
- Sums appear as: +, \\sum, \\cdot
- Powers appear as: x^2, x^{n}, x_1, x_{n}
- Subscripted variables with powers: x_1^n, x_n^n
- Products of variables: x_1x_2 or x_1....x_n or x_1 \\cdot x_2
- Start with the most distinctive equation/constraint
- If too many results, add more terms. If zero results, try DIFFERENT terms or subexpressions
- Try at least 3 different searches before giving up
- Combine structural parts: e.g. for a fraction with x_1^n in numerator, try "x_1^n" and "\\frac"

After finding candidates, use get_problem_details to confirm the match, then return ALL contest sources with links from the tool output."""


def build_agent(api_key: str | None = None):
    llm = ChatOpenAI(
        model="anthropic/claude-haiku-4-5",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ["OPENROUTER_KEY"],
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=[search_problems, search_by_contest, get_problem_details],
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
