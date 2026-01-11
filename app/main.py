from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.services.llm_handler import ModelManager
from app.services.rag_service import RagService
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

model_manager = ModelManager()
rag_service = RagService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load_models()
    rag_service.set_manager(model_manager)
    rag_service.set_db()
    load_dotenv()

    yield

    pass

app = FastAPI(lifespan=lifespan)

@app.get("/hello/{question}")
async def hello(question):
    # ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    # llm = ChatOllama(base_url=ollama_url, model="gemma3:1b")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    message = rag_service.response(question=question, llm=llm)
    # message = rag_service.retrieve(question=question, k=4)
    # message = rag_service.rerank(question=question, docs=message)

    return {"messages": message}