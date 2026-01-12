from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.services.llm_handler import ModelManager
from app.services.rag_service import RagService
from langchain_ollama import ChatOllama
import os

model_manager = ModelManager()
rag_service = RagService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load_models()
    rag_service.set_manager(model_manager)
    rag_service.set_db()

    yield

    pass

app = FastAPI(lifespan=lifespan)

@app.get("/hello/{question}")
async def hello(question):
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    llm = ChatOllama(base_url=ollama_url, model="llama3-chatqa:8b")

    message = rag_service.response(question=question, llm=llm)

    return {"messages": message}