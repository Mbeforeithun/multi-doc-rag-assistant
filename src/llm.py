import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

def get_llm():
    return ChatAnthropic(
        model=os.getenv("MODEL_NAME", "claude-sonnet-5-5"),
        max_tokens=2048,
        max_retries=3
    )