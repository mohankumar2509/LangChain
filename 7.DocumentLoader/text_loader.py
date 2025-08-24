from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", task="text-generation")
model = ChatHuggingFace(llm = llm)

loader = TextLoader("7.DocumentLoader\RAG_info.txt", encoding="utf-8")
docs = loader.load()

parser = StrOutputParser()

template = PromptTemplate(template="Summarize the content given below: {text}", input_variables=["text"])

chain = RunnableSequence(template, model, parser)
result = chain.invoke({"text": docs[0].page_content})
print(result)

# print(type(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)


