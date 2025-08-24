from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_HOME'] = "W:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task="text-generation")
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(template="generate a tweet about topic {topic}", input_variables=["topic"])
parser = StrOutputParser()

sequenceChain = RunnableSequence(prompt1, model, parser)

parallelChain = RunnableParallel({"tweets": RunnablePassthrough(), 
                                  "Word count": RunnableLambda(lambda x: len(x.split()))})

merge_chain = RunnableSequence(sequenceChain, parallelChain)
result = merge_chain.invoke({"topic":"animal"})

print(result)