from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['HF_HOME'] = "W:/huggingface_cache" # To redirect the model download to this folder

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                            task="text-generation",
                          pipeline_kwargs=dict(temperature=0.5,max_new_tokens=1500))
model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

#Example 1

template1 = PromptTemplate(template= "Give me the name, age and city of a fictional person \n {format_instruction}",
                           input_variables=[],
                           partial_variables={"format_instruction": parser.get_format_instructions()})


prompt = template1.format()
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)

# other way using chains

chain = template1 | model | parser
result = chain.invoke({})
print(result)

#Example 2 with schema enforcing - Flaw of jsonParser is it doesn't enforce schema, to enforce schema we have to use structured output parser

# topic is passed dynamically
template1 = PromptTemplate(template= "Give me top 5 highlights on {topic} \n {format_instruction}",
                           input_variables=["topic"],
                           partial_variables={"format_instruction": parser.get_format_instructions()})

chain = template1 | model | parser
result = chain.invoke({"topic":"Black Hole"})
print(result)