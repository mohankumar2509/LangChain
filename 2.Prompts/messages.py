from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
messages = [SystemMessage("You are a well trained Doctor"),
            HumanMessage("Tell me what is the most common reason for left hand pain")]

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation")
model = ChatHuggingFace(llm = llm)
result = model.invoke(messages).content
messages.append(AIMessage(content = result))

print(messages)