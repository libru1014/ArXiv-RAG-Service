from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

    def retrieve(self, question, k):
        retriever = self.vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": 10, "lambda_mult": 0.4}
        )

        docs = retriever.invoke(question)

        results = [doc.page_content for doc in docs]

        return results
    
    def rerank(self, question, docs):
        question_doc_pairs = [[question, doc] for doc in docs]

        scores = self.model_manager.rerank_model.predict(question_doc_pairs)
        scores = scores.tolist()

        ranked_docs = list(zip(docs, scores))
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return ranked_docs
    
    def response(self, question, llm, k = 4, top_k = 2):
        docs = self.retrieve(question, k)
        ranked_docs = self.rerank(question, docs)

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


# def rag_service(model_manager, question, llm):
#     loader = PyPDFDirectoryLoader("/app/data/pdfs")

#     docs = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#     split_docs = text_splitter.split_documents(docs)

#     DB_PATH = "/app/db/chroma"

#     # chroma db 생성
#     # db = Chroma.from_documents(
#     #     documents=split_docs, embedding=model_manager.embedding_model, persist_directory=DB_PATH, collection_name="arxiv_test"
#     # )

#     db = Chroma(
#         persist_directory=DB_PATH, embedding_function=model_manager.embedding_model, collection_name="arxiv_test"
#     )

#     retriever = db.as_retriever()

#     results = retriever.invoke(question)

#     lists = []
#     for r in results:
#         lists.append(r.page_content)

#     ranks = model_manager.rerank_model.rank(
#         question,
#         lists
#     )

#     context = ''
#     for i in range(3):
#         context += lists[ranks[i]["corpus_id"]]
#         context += "\n"
#         context += "===============================\n"


#     prompt = PromptTemplate.from_template(
#         """You are an AI assistant specializing in QA(Question-Answering) tasks within a Retrieval-Augmented Generation(RAG) system.
#         Your mission is to answer questions based on provided context.
#         Ensure your response is concise and directly addresses the question without any additional narration.
#         If you can't find answer in provided context, just answer "I can't find answer. Sorry."

#         #Question:
#         {question}

#         #Context:
#         {context}

#         #Answer:"""
#     )

#     rag_chain = (
#         {"context": context, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain