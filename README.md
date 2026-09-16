# DocMind: Multi-Document RAG Research Assistant

## The Problem
Researchers, students, and professionals often need to extract factual information, synthesize insights, and verify claims across multiple long-form PDF documents. Finding specific answers across a 200-page report is time-consuming, and traditional keyword search fails to understand the semantic intent behind complex questions.

## The Solution
DocMind is a Retrieval-Augmented Generation (RAG) system designed to solve this exact problem. By strictly separating deterministic information retrieval from the "intelligence" layer, it ensures that answers are faithful, factually grounded, and directly cite the provided sources. 

### Architecture

1. **Deterministic Processing Layer (Application Logic)**
   This layer handles the core application functionality reliably using standard Python logic, without relying on unpredictable AI:
   - **Ingestion**: PDF text extraction, cleaning, and overlapping text chunking (`chunker.py`).
   - **Embedding**: Fast, local vectorization of chunks using sentence-transformers (`all-MiniLM-L6-v2`) (`embedder.py`).
   - **Storage**: Persistent local vector storage using ChromaDB.
   - **Retrieval**: A robust 3-stage hybrid retrieval pipeline combining Dense Vector Search (Cosine Similarity), Sparse Keyword Search (BM25 Okapi), Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`) to deterministically filter and order the most relevant context (`retriever.py`, `reranker.py`).
   - **Context Formatting**: Strict numbered citation formatting to guide the LLM.

2. **Intelligence Layer (LLM Engine)**
   The LLM is strictly relegated to its core strength: Natural Language Processing. It receives the deterministic context and is instructed to reason over it, synthesize an answer, and generate human-readable text with explicit citations. The LLM does *not* control application flow, routing, or retrieval decisions.

### LLM-Agnostic Design
The system is built to be provider-independent. The core pipeline communicates with the LLM through a generic backend interface (`generation/llm_backends.py`), allowing you to easily swap providers via the `.env` file without modifying application logic.

Currently supported backends:
- **Groq** (Cloud, blazing fast inference)
- **Ollama** (Local, completely private and offline)

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the Environment:**
   Create a `.env` file in the root directory (you can copy `.env.example`).
   
   **To use Groq (Default):**
   ```env
   LLM_BACKEND=groq
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```
   
   **To use Ollama (Fully Local):**
   *Make sure you have Ollama installed and running locally (`ollama run llama3.2`).*
   ```env
   LLM_BACKEND=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```
   *Note: No `GROQ_API_KEY` is required when using Ollama.*

3. **Start the server:**
   ```bash
   streamlit run app.py
   ```
