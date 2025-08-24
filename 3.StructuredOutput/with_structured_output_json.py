from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

#Schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties":{
        "key_themes":{
            "type":"array",
            "items":{
                "type": "string"
            },
            "description": "write down all the key themes discussed in the review in a list"
        },
        "summary" :{
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment":{
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros":{
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the pros inside a list"
        },
        "cons":{
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the cons inside a list"
        },
        "name":{
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary","sentiment"]
}


llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1")
model = ChatHuggingFace(llm = llm)
structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""The Hardware is great, but the software feels bloated.There are too many pre-installed apps that i can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this""")

print(result)

# Structured output is not supported by deep seek r1 too, we need to specify the structured output in prompt