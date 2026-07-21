"""
RAG Evaluator
================
RAG agent retrieves context from the knowledge base. Need to evaluate both the response
provided by the LLM and the context retrieved by the RAG
"""

import os

os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"

from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from rag_agent import RAGAgent

from deepeval import evaluate

# Loading the raw JSON
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    file_path="qa_agent_dataset.json",
)

# Defining the agent
doc_path = ["theranos_legacy.txt"]
agent = RAGAgent(doc_path)

# Loading the test cases
test_cases: list[LLMTestCase] = []
for golden in dataset.goldens:
    if not isinstance(golden, Golden):
        continue
    retrieved_docs = agent.retrieve(golden.input)
    response = agent.generate(golden.input, retrieved_docs)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=response,
        retrieval_context=retrieved_docs,
        expected_output=golden.expected_output,
    )
    test_cases.append(test_case)

# Evaluation framework
ollama_model = OllamaModel(model="qwen2.5:14b-instruct", temperature=0)
relevancy = ContextualRelevancyMetric(model=ollama_model, async_mode=False)
recall = ContextualRecallMetric(model=ollama_model, async_mode=False)
precision = ContextualPrecisionMetric(model=ollama_model, async_mode=False)


answer_correctness = GEval(
    name="Answer Correctness",
    model=ollama_model,
    criteria=(
        "Evaluate if the actual output's 'answer' property is correct and"
        " complete from the input and retrieved context. If the answer is not correct"
        " or complete, reduce score."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
)

citation_accuracy = GEval(
    name="Citation Accuracy",
    model=ollama_model,
    criteria=(
        "Check if the citations in the actual output are correct and relevant"
        " based on input and retrieved context. If they're not correct, reduce score."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
)

sync_config = AsyncConfig(run_async=False)

# Evaluate retriever
retriever_metrics: list[BaseMetric] = [relevancy, recall, precision]
evaluate(test_cases, retriever_metrics, async_config=sync_config)

# Evaluate generator
generator_metrics: list[BaseMetric] = [answer_correctness, citation_accuracy]
evaluate(test_cases, generator_metrics, async_config=sync_config)
