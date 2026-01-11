# from langchain_ollama import ChatOllama
# import os
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# llm = ChatOllama(base_url=ollama_url, model="gemma3:1b")

# response = llm.invoke("Hi, nice to meet you.")
# print(response)

class ModelManager:
    def __init__(self):
        self.embedding_model = None
        self.rerank_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-m3',
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.rerank_model = CrossEncoder(
            'BAAI/bge-reranker-v2-m3',
            device=self.device
        )
        print("Load Models Successfully!")

# model_manager = ModelManager()