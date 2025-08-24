from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = """
Space exploration began as a geopolitical race but quickly evolved into one of humanity’s most ambitious scientific endeavors 

Just over a decade later, NASAss Apollo 11 mission landed humans on the Moon in 1969.
"""

splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation", breakpoint_threshold_amount=1)
chunks = splitter.create_documents([text])
print(len(chunks))