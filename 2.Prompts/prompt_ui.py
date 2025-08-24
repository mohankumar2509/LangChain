from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation")
model = ChatHuggingFace(llm = llm)
st.header("Research Tool")
# user_input = st.text_input("Enter your prompt")

paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attention Is Al You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Mod are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "M (3-5 paragraphs)", "Long (detailed explanation)"])


# Direct Usage
template = PromptTemplate(template= """Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
     - include relevant mathematical equations if present in the paper
     - Explain the mathematical concepts using simple, intutive code snippets where applicable
2. Analogies:
     - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.""", input_variables=['paper_input','style_input','length_input'], validate_template=True)

#Loading the template from template.json
template = load_prompt("Prompts/template.json")

#Fill the paceholders
prompt = template.invoke({
    "paper_input":paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Summarize"):
    st.text(model.invoke(prompt).content)


# instead of invoking template and model twice, we can create chain as below
st.header("Chain output", divider="red")

chain = template | model
result = chain.invoke({
    "paper_input":paper_input,
    "style_input": style_input,
    "length_input": length_input
})

st.write(result.content)
