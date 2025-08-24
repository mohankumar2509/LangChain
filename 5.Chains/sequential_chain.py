from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache"

template1 = PromptTemplate(template="Write a detailed short story about the topic {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Write a 5 point story from the text {text}", input_variables=["text"])


llm1 = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task = "text-generation")
model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task = "text-generation")
model2 = ChatHuggingFace(llm = llm2)

parser = StrOutputParser()

chain = template1 | model1 | parser | template2 | model2 | parser

result = chain.invoke({"topic":"Black Hole"})
print(result)

chain.get_graph().print_ascii()