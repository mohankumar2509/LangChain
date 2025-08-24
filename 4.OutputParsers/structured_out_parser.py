from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
from dotenv import load_dotenv
import os

load_dotenv()
# access_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") - use when downloading for the first time not consequent times

os.environ['HF_HOME'] = "W:/huggingface_cache" # To redirect the model download to this folder

llm = HuggingFacePipeline.from_model_id(model_id="google/gemma-2-2b-it", 
                            task="text-generation",
                            #model_kwargs={"token": access_token} - use when downloading for the first time not consequent times
                            )
model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name="fact_1", description="fact 1 on the topic"),
    ResponseSchema(name="fact_2", description="fact 2 on the topic"),
    ResponseSchema(name="fact_3", description="fact 3 on the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template="Give me 5 facts about {topic} \n {format_instructions}", input_variables=["topic"],                 partial_variables={"format_instructions": parser.get_format_instructions()})

prompt = template.invoke({"topic": "andhrapradesh politics"})
result = model.invoke(prompt)
print(result.content)
# final_result = parser.parse(result.content)


# Using Chains
# chain = template | model 
# result = chain.invoke({"topic":"andhrapradesh politics"})
# print(result.content) #this is failing to parse and display the output

#Using output fixing parser
# fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
# chain = template | model | fixing_parser
# result = chain.invoke({"topic":"andhrapradesh politics"})
# print(result)

#Failed to get an output
