from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embeddings.embed_query("Delhi is capital of india")
print(str(result))