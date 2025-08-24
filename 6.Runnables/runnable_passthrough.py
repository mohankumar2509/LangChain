from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_HOME'] = "W:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task="text-generation")
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(template="generate a joke about topic {topic}", input_variables=["topic"])
prompt2 = PromptTemplate(template="Generate a linkedin post about joke {joke}", input_variables=["joke"])

parser = StrOutputParser()

paralleRunnable = RunnableParallel({"joke" : RunnablePassthrough(), "linkedin_post":RunnableSequence(prompt2, model, parser)})
sequenceRunnable = RunnableSequence(prompt1, model, parser)

mergerunnable = RunnableSequence(sequenceRunnable, paralleRunnable)

result = mergerunnable.invoke({"topic": "humans"})
print(result)


