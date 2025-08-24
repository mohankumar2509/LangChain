from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Initiate the LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create a Prompt template
template = PromptTemplate(template="Suggest a catchy blog title about {topic}",
                        input_types=["topic"])

# Define the input
topic = input("Enter a topic")

# Format the prompt manually using PromptTemplate
prompt = template.format(topic= topic)

# call the LLM directly
blog_title = llm.predict(prompt)

# Print the output
print("Generated Blog Title:", blog_title)
