from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.services.ai_client import AIClient
from app.services.rag_service import RagService
from app.services.vector_db import QdrantConncetor
from langchain_ollama import ChatOllama
import os

ai_client = AIClient()
rag_service = RagService()
qdrant_client = QdrantConncetor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await qdrant_client.connect()
    await qdrant_client.create_collection("arxiv_papers")
    
    rag_service.set_client(ai_client=ai_client, qdrant_client=qdrant_client)
    await rag_service.load_sparse_embeddings()
    await rag_service.load_dense_embeddings()
    await rag_service.set_db()

    yield

    await qdrant_client.close()

app = FastAPI(lifespan=lifespan)

@app.get("/answer/{question}")
async def answer(question):
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    llm = ChatOllama(base_url=ollama_url, model="llama3-chatqa:8b")

    message = await rag_service.response(question=question, llm=llm)

    return {"messages": message}

@app.post("/add/{paper_id}", status_code=200)
async def add(paper_id):
    try:
        await rag_service.add_document(paper_id)
        return {"message": "Add documents successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arxiv server error: {str(e)}")
    
@app.get("/checkdb")
async def checkdb():
    collection_info = qdrant_client.client.get_collection(collection_name="arxiv_papers")
    return {"vector": collection_info.indexed_vectors_count, "point": collection_info.points_count}