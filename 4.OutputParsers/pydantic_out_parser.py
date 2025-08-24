from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()
access_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city of the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(template=("You are an assistant that extracts structured data.\n"
        "Get the name, age, and city of a politician from {state} that is publicly available.\n"
        "Return the result as a JSON object that follows this schema:\n"
        "{formatted_output}\n\n"
        "Do not return the schema itself. Only return a valid JSON instance.\n"),
                          input_variables=["state"],
                          partial_variables={"formatted_output":parser.get_format_instructions()})

prompt = template.invoke({"state":"Andhra Pradesh"})

llm = HuggingFacePipeline.from_model_id(model_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation"
                                        , model_kwargs={"token": access_token}
                                        )
model = ChatHuggingFace(llm =llm)

result = model.invoke(prompt)
print(result.content)
final_result = parser.parse(result.content)
print(final_result)


# chain = template | model | parser
# result = chain.invoke({"state":"andhrapradesh"})
# print(result)