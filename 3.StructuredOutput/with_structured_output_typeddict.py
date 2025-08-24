from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

class Review(TypedDict):
    key_themese: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary : Annotated[str, "A Brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neutral", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

load_dotenv()
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.3-70B-Instruct")
model = ChatHuggingFace(llm = llm)
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The Hardware is great, but the software feels bloated.There are too many pre-installed apps that i can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result["name"])