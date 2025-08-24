from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path= '7.DocumentLoader', glob='*.pdf', loader_cls=PyPDFLoader)
docs = loader.load()

print(docs[2].page_content)

#With Lazy loader
print("Lazy Load starts")
docs = loader.lazy_load()

for x in docs:
    print(x.metadata)