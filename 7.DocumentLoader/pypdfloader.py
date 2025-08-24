from langchain_community.document_loaders import PyPDFLoader
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

loader = PyPDFLoader("7.DocumentLoader\LabTest_21Jul2025.pdf")       
docs = loader.load()

parser = StrOutputParser()

input_val = "".join(doc.page_content for doc in docs)

template = PromptTemplate(template="Fetch the value of {medical_test} from the medical report: {report}", input_variables=["medical_test", "report"])

chain = RunnableSequence(template, model, parser)

while True:
    medical_test = input("YOU:")
    if medical_test == "exit":
        break
    result = chain.invoke({"medical_test":medical_test, "report": input_val})
    print(f"AI: {result}")