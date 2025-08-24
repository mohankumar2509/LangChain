from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
documents = ["Delhi is capital of india", "Kolkata is capital of West Bengal", "Paris is the capital of France"]

embeddings = OpenAIEmbeddings(model="", dimensions=42)
result = embeddings.embed_documents(documents)
print(str(result))