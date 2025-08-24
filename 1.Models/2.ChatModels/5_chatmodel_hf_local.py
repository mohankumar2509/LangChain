from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = "W:/huggingface_cache" # To redirect the model download to this folder

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                            task="text-generation",
                          pipeline_kwargs=dict(temperature=0.5,max_new_tokens=100)
                          )

model = ChatHuggingFace(llm = llm)
response = model.invoke("What is the Capital of India?")
print(response.content)


# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# import os

# # Redirect cache
# os.environ['HF_HOME'] = "W:/huggingface_cache"

# # Load model and tokenizer manually
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# # Create pipeline
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
#                 temperature=0.5, max_new_tokens=100)

# # Wrap with LangChain
# llm = HuggingFacePipeline(pipeline=pipe)
# chat_model = ChatHuggingFace(llm=llm)

# # Invoke
# response = chat_model.invoke("What is the capital of India?")
# print(response.content)