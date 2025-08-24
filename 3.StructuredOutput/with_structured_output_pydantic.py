from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

class Review(BaseModel):
    key_themese: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary : str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(description="Write the name of the reviewer", default=None)


llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1")
model = ChatHuggingFace(llm = llm)
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The Hardware is great, but the software feels bloated.There are too many pre-installed apps that i can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result.name)

# Note: Pydantic schema is only supported by open AI 4.0, Google Gemini