from langchain_core.prompts import ChatPromptTemplate


chat_temlplate = ChatPromptTemplate([
('system', 'You a a helpful {domain} expert'),
('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_temlplate.invoke({"domain": "cricket", "topic": "yorker"})

print(prompt)