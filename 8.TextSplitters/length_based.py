from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

text = """
Space exploration began as a geopolitical race but quickly evolved into one of humanity’s most ambitious scientific endeavors 

Just over a decade later, NASAss Apollo 11 mission landed humans on the Moon in 1969.
"""

splitter = CharacterTextSplitter(chunk_size = 300, chunk_overlap = 20, separator='')

result = splitter.split_text(text)
print(result)

# To read multiple docs
loader = PyPDFLoader(file_path="7.DocumentLoader\LabTest_22Jul2025.pdf")
docs = loader.load()

print('*' * 100)
result = splitter.split_documents(docs)
print(result[0].page_content)

