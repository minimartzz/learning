"""
basic_rag.py — Basic RAG Pipeline
===================================
This demonstrates the fundamental RAG pattern:
  1. Load documents
  2. Split into chunks
  3. Embed and store in ChromaDB
  4. Retrieve relevant chunks on query
  5. Generate answer with LLM

ARCHITECTURAL DECISIONS:
- RecursiveCharacterTextSplitter: Tries to split on natural boundaries
  (paragraphs → sentences → words) before resorting to character splits.
  This preserves semantic coherence within chunks.

- chunk_size=500, chunk_overlap=50: A starting heuristic. Too small = 
  missing context; too large = noisy retrieval + hitting context limits.
  Overlap ensures sentences split across boundaries aren't lost.

- HuggingFaceEmbeddings (all-MiniLM-L6-v2): A lightweight, fast model
  that runs locally without API costs. For production, consider 
  text-embedding-ada-002 (OpenAI) or text-embedding-3-small for better
  quality at scale.

- ChromaDB (in-memory): Zero-setup vector store, ideal for development.
  In production you'd persist to disk or use a hosted solution.

- k=3 retrieval: Fetching top-3 chunks balances context richness vs.
  noise. Too many chunks overwhelm the LLM; too few miss relevant info.
"""

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


# ─── 1. DOCUMENT LOADING ─────────────────────────────────────────────────────
def load_documents(data_dir: str = "./data"):
    """
    DirectoryLoader handles multiple file types with a single call.
    In production you'd extend this with loaders for PDFs, web pages,
    databases, APIs — whatever your knowledge sources are.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"[LOAD] Loaded {len(docs)} documents from '{data_dir}'")
    return docs


# ─── 2. CHUNKING ─────────────────────────────────────────────────────────────
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """
    WHY CHUNKING MATTERS:
    LLMs have context window limits. A 10,000-word document can't be
    shoved into every query. Chunking lets us surgically retrieve only
    the relevant pieces. This is the single most impactful tunable in a
    RAG pipeline — wrong chunk size = wrong answers.

    RecursiveCharacterTextSplitter priority order:
    ["\\n\\n", "\\n", " ", ""] — paragraph → line → word → char
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,   # Stores char offset in metadata — useful
                                # for highlighting source passages in UI
    )
    chunks = splitter.split_documents(docs)
    print(f"[CHUNK] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─── 3. EMBEDDING + VECTOR STORE ─────────────────────────────────────────────
def build_vectorstore(chunks, persist_dir=None):
    """
    WHY EMBEDDINGS:
    Embeddings convert text into dense vectors where semantic similarity 
    = geometric proximity. "gradient descent" and "backpropagation" end up
    near each other in embedding space, even without shared keywords.
    This beats keyword search for conceptual questions.

    PRODUCTION NOTE: Embedding is the expensive step. Pre-compute and
    persist your vectorstore — don't re-embed on every startup.
    """
    print("[EMBED] Loading embedding model (downloads on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # 80MB, fast, good quality
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine sim
    )

    print("[STORE] Building ChromaDB vectorstore...")
    kwargs = {"documents": chunks, "embedding": embeddings}
    if persist_dir:
        kwargs["persist_directory"] = persist_dir  # Persist to disk

    vectorstore = Chroma.from_documents(**kwargs)
    print(f"[STORE] Stored {vectorstore._collection.count()} vectors in ChromaDB")
    return vectorstore, embeddings


# ─── 4. RETRIEVAL ─────────────────────────────────────────────────────────────
def build_retriever(vectorstore, k=3):
    """
    The retriever converts a query to an embedding and finds the k most
    similar chunks via cosine similarity (or dot product if normalized).

    k=3 is a reasonable default. In production, tune this based on:
    - Average answer complexity (more context = higher k)
    - LLM context window size
    - Latency requirements (more chunks = slower LLM inference)
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",      # Options: similarity, mmr, similarity_score_threshold
        search_kwargs={"k": k},
    )
    return retriever


# ─── 5. PROMPT TEMPLATE ──────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Use ONLY the following context to answer "
        "the question. If the answer isn't in the context, say \"I don't know "
        "based on the provided documents.\"\n\n"
        "Context:\n{context}"
    )),
    ("human", "{question}"),
])
# WHY ChatPromptTemplate: langchain_core's native prompt type. Explicit role
# separation (system / human) is cleaner than a flat string template and maps
# directly to the chat message format every modern LLM API expects.
#
# {context} and {question} are populated by the LCEL chain in build_qa_chain —
# no magic wiring, just a plain dict flowing through the pipe.
#
# WHY A STRICT SYSTEM PROMPT: Grounding the LLM to retrieved context is the
# core RAG anti-hallucination mechanism. Without "use only the context", the
# LLM blends retrieved facts with parametric knowledge, making attribution
# impossible and errors harder to catch.


# ─── 6. QA CHAIN ─────────────────────────────────────────────────────────────
def _format_docs(docs) -> str:
    """Concatenate retrieved Document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(retriever, llm):
    """
    Pure langchain_core LCEL chain — no dependency on the langchain classic package.

    All imports come from langchain_core.runnables and langchain_core.output_parsers,
    which are the stable, low-level primitives that everything else builds on top of.

    HOW THE PIPE WORKS:
    The | operator connects Runnables into a directed pipeline. Each component
    receives the output of the previous one as its input.

        RunnableParallel → ChatPromptTemplate → LLM → StrOutputParser

    RunnableParallel runs its branches concurrently and merges results into a dict:
      - "context": invokes the retriever, then formats the Document list into a string
      - "question": RunnablePassthrough() forwards the original query unchanged

    The merged dict {"context": "...", "question": "..."} flows into RAG_PROMPT,
    which fills its {context} and {question} slots and returns a list of chat messages.
    The LLM receives those messages and produces an AIMessage.
    StrOutputParser extracts the plain text content from the AIMessage.

    WHY THIS OVER create_retrieval_chain:
    - Zero dependency on the langchain classic package (langchain_core only)
    - Every step is explicit and readable — no hidden Document formatting logic
    - The same primitives compose into any other chain shape without learning new APIs
    - Streaming works identically: chain.stream({"question": q})
    - Async works identically: await chain.ainvoke({"question": q})

    TRADE-OFF vs create_retrieval_chain:
    Source documents are not in the output dict by default (we formatted them
    to a string for the prompt). To retain them, use the pattern in the
    build_qa_chain_with_sources() variant below.
    """
    chain = (
        RunnableParallel({
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        })
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def build_qa_chain_with_sources(retriever, llm):
    """
    Variant that preserves retrieved Document objects in the output so callers
    can render source citations — equivalent to return_source_documents=True.

    RunnableParallel is used twice:
      1. First pass: retrieve docs and pass the question through in parallel.
      2. Second pass: generate the answer AND carry the raw docs forward.

    Output dict: {"question": str, "context": List[Document], "answer": str}
    """
    # Step 1: retrieve docs + pass question through simultaneously
    retrieve = RunnableParallel({
        "context":  retriever,
        "question": RunnablePassthrough(),
    })

    # Step 2: generate answer (formatting docs inline) while also forwarding them
    generate = RunnableParallel({
        "answer": (
            RunnableParallel({
                "context":  lambda x: _format_docs(x["context"]),
                "question": lambda x: x["question"],
            })
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        ),
        "context":  lambda x: x["context"],   # raw Document objects for citations
        "question": lambda x: x["question"],
    })

    return retrieve | generate


# ─── DEMO RUNNER ──────────────────────────────────────────────────────────────
def run_basic_rag_demo():
    print("\n" + "="*60)
    print("  BASIC RAG PIPELINE DEMO")
    print("="*60)

    # LLM: Qwen3 0.6B via Ollama (smallest Qwen open-source model)
    # WHY Qwen3-0.6B: Runs on CPU with <2GB RAM, Apache 2.0 licensed,
    # surprisingly capable for RAG tasks where the context does the heavy lifting.
    # The LLM's job in RAG is synthesis, not recall — small models handle this well.
    #
    # WHY Ollama: Zero-config local inference server. Handles model download,
    # quantization (Q4_K_M by default), and serves an OpenAI-compatible API.
    # No GPU required for 0.6B; runs comfortably on any modern laptop.
    #
    # SETUP (one-time):
    #   brew install ollama          # macOS
    #   ollama pull qwen3:0.6b       # ~400MB download
    #   ollama serve                 # starts API at localhost:11434
    #
    # For better quality (more RAM/GPU required), swap the model tag:
    #   qwen3:1.7b  (~1GB)   — noticeable quality improvement
    #   qwen3:4b    (~2.5GB) — rivals much larger older models
    #   qwen3:8b    (~5GB)   — strong quality, needs 8GB+ RAM
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(
        model="qwen3:0.6b",
        temperature=0,          # Deterministic — critical for RAG (no creative drift)
        # num_ctx=4096,         # Uncomment to extend context window if needed
    )
    # Fallback for CI/testing without Ollama running:
    # from langchain_community.llms.fake import FakeListLLM
    # llm = FakeListLLM(responses=["Answer based on context..."] * 10)

    # Build the pipeline
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    vectorstore, _ = build_vectorstore(chunks)
    retriever = build_retriever(vectorstore, k=3)

    # Use the sources variant so we can show citations
    qa_chain = build_qa_chain_with_sources(retriever, llm)

    # Run queries
    queries = [
        "What is supervised learning and what algorithms does it use?",
        "How do you prevent overfitting in machine learning models?",
        "How does Retrieval Augmented Generation work?",
    ]

    print("\n--- QUERY RESULTS ---\n")
    for query in queries:
        print(f"Q: {query}")
        result = qa_chain.invoke(query)
        print(f"A: {result['answer']}")

        # result["context"] holds raw Document objects — same as before
        sources = {doc.metadata.get("source", "unknown") for doc in result["context"]}
        print(f"   Sources: {', '.join(os.path.basename(s) for s in sources)}")
        print()

    return vectorstore  # Return for potential reuse


if __name__ == "__main__":
    run_basic_rag_demo()
