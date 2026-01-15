import httpx
import os
from typing import List

EMBED_URL = os.getenv("EMBED_URL", "http://embed-server:8080")
RERANK_URL = os.getenv("RERANK_URL", "http://rerank-server:8080")

class AIClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def get_embedding(self, text: str) -> List[float]:
        response = await self.client.post(
            f"{EMBED_URL}/embed",
            json={"inputs": text}
        )
        response.raise_for_status()
        
        return response.json()[0]

    async def get_rerank(self, query: str, documents: List[str]) -> List[dict]:
        response = await self.client.post(
            f"{RERANK_URL}/rerank",
            json={
                "query": query,
                "texts": documents
            }
        )
        response.raise_for_status()

        return response.json()