from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.services.ai_client import AIClient
from app.services.rag_service import RagService
from langchain_ollama import ChatOllama
import os

ai_client = AIClient()
rag_service = RagService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service.set_client(ai_client)
    rag_service.set_db()

    yield

    pass

app = FastAPI(lifespan=lifespan)

@app.get("/hello/{question}")
async def hello(question):
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    llm = ChatOllama(base_url=ollama_url, model="llama3-chatqa:8b")

    message = await rag_service.response(question=question, llm=llm)

    return {"messages": message}