# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RagService:
    def __init__(self):
        self.model_manager = None
        self.vectordb = None

    def set_manager(self, model_manager):
        self.model_manager = model_manager
    
    def set_db(self):
        self.vectordb = Chroma(
            persist_directory="/app/db/chroma", embedding_function=self.model_manager.embedding_model, collection_name="arxiv_test"
        )

    # 문서 검색
    def retrieve(self, question, k):
        # 대규모 문서 중 검색이 필요, 따라서 fetch_k를 10으로 설정
        # 문서의 정확성이 중요하지만, 다양한 관점이 필요할 수도 있어 lambda_mult 값을 0.4로 설정 
        retriever = self.vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": 10, "lambda_mult": 0.4}
        )

        docs = retriever.invoke(question)

        results = [doc.page_content for doc in docs]

        return results
    
    # 문서 재정렬
    def rerank(self, question, docs):
        question_doc_pairs = [[question, doc] for doc in docs]

        scores = self.model_manager.rerank_model.predict(question_doc_pairs)
        scores = scores.tolist()

        # rerank 모델로부터 받은 점수를 기준으로 정렬
        ranked_docs = list(zip(docs, scores))
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return ranked_docs
    
    # chain을 만들고 그 결과를 받아옴
    def response(self, question, llm, k = 4, top_k = 2):
        docs = self.retrieve(question, k)
        ranked_docs = self.rerank(question, docs)

        # 상위 top_k개 만큼의 문서만을 활용
        context = "\n\n".join(f"<document>{ranked_docs[i][0]}</document>" for i in range(top_k))

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