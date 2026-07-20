"""
Introduction
===============
A simple setup script to understand deepeval Python package
"""

from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from deepeval import evaluate

# Define local Ollama model
model = OllamaModel(model="qwen2.5:7b-instruct", temperature=0)

# GEval is the generic testing framework using LLM-as-a-judge
correctness_metric = GEval(
    name="Correctness",
    model=model,
    criteria=(
        "Determine if the 'actual output' is correct based on the 'expected output'."
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5,
)

# Define the test case
test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    actual_output=(
        "A persistent cough and fever could signal various illnesses, from minor"
        " infections to more serious conditions like pneumonia or COVID-19. It's"
        " advisable to seek medical attention if symptoms worsen, persist beyond a few"
        " days, or if you experience difficulty breathing, chest pain, or other"
        " concerning signs."
    ),
    expected_output=(
        "A persistent cough and fever could indicate a range of illnesses, from a mild"
        " viral infection to more serious conditions like pneumonia or COVID-19. You"
        " should seek medical attention if your symptoms worsen, persist for more than"
        " a few days, or are accompanied by difficulty breathing, chest pain, or other"
        " concerning signs."
    ),
)

evaluate([test_case], [correctness_metric])
