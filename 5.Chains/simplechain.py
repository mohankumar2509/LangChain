from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache"
template = PromptTemplate(template="Who is the president of {country}", input_variables=["country"])

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task = "text-generation")
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = template | model | parser
result = chain.invoke({"country": "india"})
print(result)

chain.get_graph().print_ascii()
