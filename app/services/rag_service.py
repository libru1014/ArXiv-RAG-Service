# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.ai_client import AIClient
import os
from typing import List

class RagService:
    def __init__(self):
        self.ai_client = None
        self.vectordb = None

    def set_client(self, ai_client: AIClient):
        self.ai_client = ai_client
    
    def set_db(self):
        embeddings = HuggingFaceEndpointEmbeddings(model=os.getenv("EMBED_URL", "http://embed-server:8080"))
        self.vectordb = Chroma(
            persist_directory="/app/db/chroma", embedding_function=embeddings, collection_name="arxiv_test"
        )

    # 문서 검색
    def retrieve(self, question: str, k: int):
        # 대규모 문서 중 검색이 필요, 따라서 fetch_k를 10으로 설정
        # 문서의 정확성이 중요하지만, 다양한 관점이 필요할 수도 있어 lambda_mult 값을 0.4로 설정 
        retriever = self.vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": 10, "lambda_mult": 0.4}
        )

        docs = retriever.invoke(question)

        results = [doc.page_content for doc in docs]

        return results
    
    # 문서 재정렬
    async def rerank(self, question: str, docs: List[str]):
        scores = await self.ai_client.get_rerank(question, docs)

        ranked_docs = []
        for item in scores:
            idx = item['index']
            ranked_docs.append(docs[idx])

        return ranked_docs
    
    # chain을 만들고 그 결과를 받아옴
    async def response(self, question: str, llm, k = 4, top_k = 2):
        docs = self.retrieve(question, k)
        ranked_docs = await self.rerank(question, docs)

        # 상위 top_k개 만큼의 문서만을 활용
        context = "\n\n".join(f"<document>{ranked_docs[i]}</document>" for i in range(top_k))

        prompt = PromptTemplate.from_template(
        """You are an AI assistant specializing in QA(Question-Answering) tasks within a Retrieval-Augmented Generation(RAG) system.
        Your mission is to answer questions based on provided context.
        Ensure your response is concise and directly addresses the question without any additional narration.
        If you can't find answer in provided context, just answer "I can't find answer. Sorry."

        #Question:
        {question}

        #Context:
        {context}

        #Answer:"""
        )

        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({"context": context, "question": question})

        return result