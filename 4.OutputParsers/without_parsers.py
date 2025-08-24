from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(template="Write a detailed description about the topic. {topic}")
prompt1 = template1.invoke({"topic": "Black Hole"})
result = model.invoke(prompt1)

template2 = PromptTemplate(template="Write 5 lines summary on the following text. {text}")
prompt2 = template2.invoke({"text": result.content})
result = model.invoke(prompt2)


print(result.content)