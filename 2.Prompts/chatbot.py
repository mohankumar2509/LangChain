from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation")
model = ChatHuggingFace(llm = llm)
chat_history = [SystemMessage(content="You are an Helpful AI Assistant")]
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content = user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI:", result.content)

print(chat_history)

"""
As the chat history keep growing, The chat_history list don't have information about which message belongs to whom.
So we need to implement chat_history as dictionary something like {"user": "-------", "AI": "********"}

Langchain itself handles this, Langchain has 3 types of messages
   - System Messages
   - Human Messages
   - AI Messages
"""