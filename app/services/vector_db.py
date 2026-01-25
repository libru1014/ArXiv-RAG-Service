import os
from qdrant_client import QdrantClient, models
# from langchain_qdrant import FastEmbedSparse
# from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

class QdrantConncetor:
    def __init__(self):
        self.url = os.getenv("VECTOR_DB_URL", "http://qdrant:6333")
        self.client = None
        
    async def connect(self):
        if self.client is None:
            self.client = QdrantClient(url=self.url)
            print("Connected to Qdrant.")

    async def close(self):
        if self.client:
            self.client.close()
            print("Close connection to Qdrant.")

    async def create_collection(self, collection_name: str):
        collections = self.client.get_collections()
        exists = any(c.name == collection_name for c in collections.collections)

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )},
                sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
            )
            print(f"Collection '{collection_name}' created.")

    # async def save_data(self, id, vector, metadata):
    #     await self.client.upsert(
    #         collection_name="arxiv",
    #         points=[
    #             models.PointStruct(
    #                 id=id,
    #                 vector=vector,
    #                 payload=metadata
    #             )
    #         ]
    #     )

    # async def search_data(self, query: str, k: int = 4):
    #     return await self.client.query(
    #         collection_name="arxiv",
    #         query_text=query,
    #         limit=k
    #     )