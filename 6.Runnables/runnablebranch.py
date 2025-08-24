from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_HOME'] = "W:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task="text-generation")
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(template="generate comprehensive details about topic {topic}", input_variables=["topic"])
prompt2 = PromptTemplate(template="Summaize the given details {text}", input_variables=["text"])

parser = StrOutputParser()

sequencechain = RunnableSequence(prompt1, model, parser)

branchchain = RunnableBranch(((lambda x : len(x.split())>10),RunnableSequence(prompt2, model, parser, RunnableLambda(lambda x: "Summarizing:\n"+x+"\n"+str(len(x.split()))))),
                             (RunnablePassthrough()))

mergechain = RunnableSequence(sequencechain, branchchain)
result = mergechain.invoke({"topic": "marraige"})

print(result)