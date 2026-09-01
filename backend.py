import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")


def check_environment():
    """Check whether the required environment variables are available."""

    print("Environment check")
    print("-----------------")

    print(
        "OPENAI_API_KEY:",
        "Loaded" if OPENAI_API_KEY else "Missing"
    )

    print(
        "LANGSMITH_API_KEY:",
        "Loaded" if LANGSMITH_API_KEY else "Missing"
    )

    print(
        "LANGSMITH_PROJECT:",
        LANGSMITH_PROJECT or "Missing"
    )

    print(
        "LANGSMITH_TRACING:",
        LANGSMITH_TRACING or "Missing"
    )


if __name__ == "__main__":
    check_environment()