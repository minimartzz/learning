from deepeval.dataset import EvaluationDataset
from deepeval.models import OllamaEmbeddingModel, OllamaModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

# Define model and synthesizer
model = OllamaModel(model="qwen2.5:14b-instruct", temperature=0)
embedder = OllamaEmbeddingModel(model="nomic-embed-text:latest")
synthesizer = Synthesizer(model=model, async_mode=False)

context_config = ContextConstructionConfig(embedder=embedder, critic_model=model)
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["theranos_legacy.txt"],
    context_construction_config=context_config,
)

dataset = EvaluationDataset(goldens=goldens)
dataset.save_as(file_name="qa_agent_dataset.json", file_type="json", directory=".")
print("Successfully saved QA Agent dataset.")
