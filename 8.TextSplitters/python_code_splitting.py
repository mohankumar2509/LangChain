from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text = """
import langchain
import requests

# print(langchain.__version__)

endpoint_url = "https://quasarmarket.coforge.com/aistudio-llmrouter-api/api/v2/chat/completions"
headers: dict = {
        "Content-Type": "application/json", 
        "X-API-KEY": "d7258a9b-fbf6-4952-87a7-758f5690389a"
    }
payload = {
            "model": "claude-sonnet-3-5",
            "messages": [{"role": "user", "content": "Write 5lines about cricket"}],
            "temperature": 1,
            "max_tokens": 25
        }

response = requests.post(endpoint_url, headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])
"""

splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,chunk_size = 300, chunk_overlap = 0)
chunks = splitter.split_text(text)

print(len(chunks))
