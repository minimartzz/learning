"""
RAG Improvement
=================
Creates generator and retreiver tests, then implement iterations to improve RAG
performance
"""

from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, RetrievedContextData
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from rag_agent import RAGAgent

# Loading the raw JSON and agent
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    file_path="qa_agent_dataset.json",
)

document_path = ["theranos_legacy.txt"]
retriever = RAGAgent(document_path)

# Generate the test cases
# retriever_test_cases = []
# generator_test_cases = []
# for golden in dataset.goldens:
#     retrieved_docs = retriever.retrieve(golden.input)
#     generated_answers = retriever.generate(golden.input, retrieved_docs)
#     test_case = LLMTestCase(
#         input=golden.input,
#         actual_output=str(generated_answers),
#         expected_output=golden.expected_output,
#         retrieval_context=retrieved_docs,
#     )
#     generator_test_cases.append(test_case)
#     retriever_test_cases.append(test_case)

# print(f"Number of retriever test cases: {len(retriever_test_cases)}")
# print(f"Number of generator test cases: {len(generator_test_cases)}")


ollama_model = OllamaModel(model="qwen2.5:14b-instruct", temperature=0)
relevancy = ContextualRelevancyMetric(model=ollama_model, async_mode=False)
recall = ContextualRecallMetric(model=ollama_model, async_mode=False)
precision = ContextualPrecisionMetric(model=ollama_model, async_mode=False)

retriever_metrics: list[BaseMetric] = [relevancy, recall, precision]

# Test parameters
chunking_strategies = [500, 1024, 2048]

vector_store_classes = [("FAISS", FAISS), ("Chroma", Chroma)]

document_paths = ["theranos_legacy.txt"]
for chunk_size in chunking_strategies:
    for vector_store_class, vector_store_model in vector_store_classes:
        retriever = RAGAgent(
            document_paths,
            chunk_size=chunk_size,
            vector_store_class=vector_store_model,
        )
        retriever_test_cases = []
        for golden in dataset.goldens:
            if not isinstance(golden, Golden):
                continue
            retrieved_docs = retriever.retrieve(golden.input)
            context_list: list[str | RetrievedContextData] = [
                str(doc) for doc in retrieved_docs
            ]
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=golden.expected_output,
                expected_output=golden.expected_output,
                retrieval_context=context_list,
            )
            retriever_test_cases.append(test_case)
