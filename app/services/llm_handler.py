from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import torch

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