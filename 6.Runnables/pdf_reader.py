from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI


#  Load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings and store in FAISS
vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

# create a retriever (fetches relavent documents)
retriever = vectorstore.as_retriever()

# Manually retrieve relevant documents
query = "What are the key takeaways from the document"
retrieved_docs = retriever.get_relevant_documents(query)

# Combine Retrieved text into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in docs])

# Initialize the LLM
llm = OpenAI(model_name = "gpt-3.5-turbo", temperature=0.7)

# Manually pass retrieved text to LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = llm.predict(prompt)

#Print the answer
print("Answer:", answer)