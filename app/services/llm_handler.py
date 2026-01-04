from langchain_ollama import ChatOllama
import os

ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

llm = ChatOllama(base_url=ollama_url, model="gemma3:1b")

response = llm.invoke("Hi, nice to meet you.")
print(response)