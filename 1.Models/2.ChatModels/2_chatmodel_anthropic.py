from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()
print(f"Anthopic API Key: {os.getenv('ANTHROPIC_API_KEY')}")

model = ChatAnthropic(model="claude-opus-4-1")
model.invoke("What is the capital of India?")