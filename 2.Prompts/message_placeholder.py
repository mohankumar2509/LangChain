from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


template = ChatPromptTemplate([
    ('system', 'you are an an helpful customer service agent'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

chat_history = []
with open("Prompts/chat_history.txt") as f:
    chat_history.extend(f.readlines())


user_input = "when will i get my refund"
prompt = template.invoke({"chat_history": chat_history, "query": user_input})

print(prompt)