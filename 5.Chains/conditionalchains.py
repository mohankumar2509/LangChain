from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache"

class Review(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description= "User sentiment review")

parser = PydanticOutputParser(pydantic_object=Review)

template1 = PromptTemplate(template="Analyze the sentiment of the review as positive or negative. Review: \n{review}. \n {format_instuctions}", input_variables=["review"]
                           , partial_variables={"format_instructions":parser.get_format_instructions()}
                                                )


model = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task = "text-generation")

classifier_chain = template1 | model | parser
result = classifier_chain.invoke({"review": "this is a wonderful phone"})
print(result)

template2 = PromptTemplate(template="Write an appropriate response to this positive feedback \n {feedback}", input_variables=["feedback"])

template3 = PromptTemplate(template="Write an appropriate response to this negative feedback \n {feedback}", input_variables=["feedback"])


branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", template2 | model | parser),
    (lambda x: x["sentiment"] == "negative", template3 | model | parser),
    RunnableLambda(lambda x : f"could not find sentiment {x}")
)


chain = classifier_chain | branch_chain

result = chain.invoke({"review": "this is a wonderful phone"})
print(result)

