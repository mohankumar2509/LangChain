from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache"

template1 = PromptTemplate(template="Write a detailed summary about topic {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Generate 5 simple question and answers on topic {topic}", input_variables=["topic"])
template3 = PromptTemplate(template="Merge the notes {notes} and the quiz {quiz}", input_variables=["notes", "quiz"])

llm1 = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task = "text-generation")
model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task = "text-generation")
model2 = ChatHuggingFace(llm = llm2)

parser = StrOutputParser()

parallel_chain = RunnableParallel({"notes": template1 | model1 | parser , "quiz" : template2 | model2 | parser})

merger_chain = template3 | model2 | parser

chain = parallel_chain | merger_chain

result = chain.invoke({"topic": "Support vector machines"})
print(result)

chain.get_graph().print_ascii()

