from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache" # To redirect the model download to this folder

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                            task="text-generation",
                          pipeline_kwargs=dict(temperature=0.5,max_new_tokens=1500))
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()
template1 = PromptTemplate(template="Write a detailed description about the topic. {topic}", input_variables=['topic'])
template2 = PromptTemplate(template="Write 5 lines summary on the following text as pointers. {text}", input_variables=["text"])

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "Black Hole"})

print(result)
