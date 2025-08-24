from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_HOME'] = "W:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task="text-generation")
model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(template="Create a joke about topic {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Explain the following joke - {text}", input_variables=["text"])

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, template2, model, parser)

print(chain.invoke({"topic": "AI"}))