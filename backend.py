import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


def ask_nomadq(query: str):
    """Send a user query to GPT-4o mini."""

    response = llm.invoke(
        f"""You are NomadQ, a helpful travel planning assistant.

User request:
{query}
"""
    )

    return response.content


if __name__ == "__main__":
    response = ask_nomadq(
        "Plan a 4 day trip to Dubai."
    )

    print("NomadQ response")
    print("----------------")
    print(response)