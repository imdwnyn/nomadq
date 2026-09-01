import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


def test_llm():
    """Send a simple test request to GPT-4o mini."""

    response = llm.invoke(
        "You are NomadQ, a travel planning assistant. "
        "Reply with one short sentence introducing yourself."
    )

    print("NomadQ response")
    print("----------------")
    print(response.content)


if __name__ == "__main__":
    test_llm()