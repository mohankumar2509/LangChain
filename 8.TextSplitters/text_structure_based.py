from langchain.text_splitter import RecursiveCharacterTextSplitter


text = """
Space exploration began as a geopolitical race but quickly evolved into one of humanity’s most ambitious scientific endeavors 

Just over a decade later, NASAss Apollo 11 mission landed humans on the Moon in 1969.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 20)
chunks = splitter.split_text(text)

print(len(chunks))