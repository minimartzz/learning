"""
RAG Agent
============
A simple RAG based agent that takes in documents, performs the embedding transformation,
stores it in memory and retrieves and generates output from a query
"""

from deepeval.test_case import RetrievedContextData
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 2,
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="nomic-embed-text:latest"
        )
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        """Creates the in-memory vectorstore"""
        documents = []
        for document_path in self.document_paths:
            with open(document_path, "r", encoding="utf-8") as file:
                raw_text = file.read()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            documents.extend(splitter.create_documents([raw_text]))

        return self.vector_store_class.from_documents(documents, self.embedding_model)

    def retrieve(self, query: str) -> list[str | RetrievedContextData]:
        """Retrieve the most relevant docs and parts"""
        docs = self.vector_store.similarity_search(query, k=self.k)
        context: list[str | RetrievedContextData] = [doc.page_content for doc in docs]
        return context

    def generate(
        self,
        query: str,
        retrieved_docs: list,
        llm_model=None,
        prompt_template: str | None = None,
    ) -> str:
        """Generates a response based on the retrieved documents"""
        context = "\n".join(retrieved_docs)
        model = llm_model or ChatOllama(model="qwen2.5:14b-instruct", temperature=0)
        prompt = prompt_template or (
            "Answer the query using the context below.\n\nContext:\n{context}\n\nQuery:"
            "\n{query} Only use information from the context. If nothing relevant is"
            " found, respond with: 'No relevant information available.'"
        )
        prompt = prompt.format(context=context, query=query)
        return str(model.invoke([HumanMessage(content=prompt)]).content)

    def answer(self, query: str, llm_model=None, prompt_template: str | None = None):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(
            query, retrieved_docs, llm_model, prompt_template
        )
        return generated_answer, retrieved_docs


if __name__ == "__main__":
    doc_path = ["theranos_legacy.txt"]
    query = "What is the NanoDrop 3000, and what certifications does Theranos hold?"

    retriever = RAGAgent(doc_path)
    answer, retrieved_docs = retriever.answer(query)

    print()
    print(answer)
    print()
    print(retrieved_docs)
