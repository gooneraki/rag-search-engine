
# 🎬 Hoopla Search Engine – End-to-End RAG System

> **Built as part of [Boot.dev's](https://www.boot.dev/) "Learn Retrieval Augmented Generation" course**
> 
> Master advanced search techniques and build a production-ready RAG pipeline in Python.

An end-to-end Retrieval-Augmented Generation (RAG) system built from scratch.

This project implements a complete modern search pipeline including:

- Keyword search (BM25)
- Semantic search (embeddings + cosine similarity)
- Hybrid search (RRF + weighted fusion)
- Chunking & chunk-level retrieval
- LLM-based query enhancement (spell, rewrite, expand)
- LLM & cross-encoder re-ranking
- Evaluation metrics (Precision@K, Recall@K, F1)
- Retrieval-Augmented Generation (RAG)
- Multimodal search (image + text using CLIP)

The dataset is a movie collection powering a fictional streaming service called **Hoopla**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Example Commands](#-example-commands)
- [What This Project Demonstrates](#-what-this-project-demonstrates)
- [Future Improvements](#-future-improvements)

---

# 🚀 Features

## 🔎 Keyword Search
- Inverted Index
- TF, IDF, TF-IDF
- BM25 scoring
- Boolean-style retrieval
- Document length normalization

## 🧠 Semantic Search
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- Cosine similarity
- Chunked document embeddings
- Chunk-level search

## 🔀 Hybrid Search
- Weighted combination (α tuning)
- Reciprocal Rank Fusion (RRF)
- Score normalization (Min-Max)

## 🤖 LLM Enhancements (Gemini API)
- Spell correction
- Query rewriting
- Query expansion
- LLM-based reranking (individual + batch)
- LLM evaluation (0–3 scoring)

## ⚡ Cross-Encoder Reranking
- `cross-encoder/ms-marco-TinyBERT-L2-v2`
- Query-document joint scoring

## 📊 Evaluation
- Golden dataset testing
- Precision@K
- Recall@K
- F1 Score
- Error analysis logging

## 📚 Augmented Generation (RAG)
- Search + LLM answer generation
- Multi-document summarization
- Citation-aware responses
- Conversational Q&A

## 🖼 Multimodal Search
- Image embedding via CLIP
- Image-to-text semantic search
- Multimodal query rewriting (Gemini)

---

# 🏗 Project Structure

```
cli/
  keyword_search_cli.py      # BM25 keyword search CLI
  semantic_search_cli.py     # Semantic/embedding search CLI
  hybrid_search_cli.py       # Hybrid RRF/weighted search CLI
  evaluation_cli.py          # Evaluation metrics CLI
  augmented_generation_cli.py # RAG answer generation CLI
  multimodal_search_cli.py   # Image + text search CLI
  describe_image_cli.py      # Image description utilities
  test_gemini.py             # Gemini API testing utilities

cli/lib/
  keyword_search.py          # Inverted index & BM25 implementation
  semantic_search.py         # Embeddings & chunked search
  hybrid_search.py           # RRF & weighted fusion
  augmented_generation.py    # RAG prompts & generation
  multimodal_search.py       # CLIP-based multimodal search
  genai.py                   # Gemini API client & prompt templates
  search_utils.py            # Shared utilities & constants
  utils.py                   # Text processing utilities

data/
  movies.json                # Movie dataset (Hoopla catalog)
  golden_dataset.json        # Test cases for evaluation
  paddington.jpeg            # Sample image for multimodal search
  stopwords.txt              # Stopwords for text processing

cache/
  index.pkl                  # Inverted index cache
  docmap.pkl                 # Document mapping cache
  doc_lengths.pkl            # Document lengths for BM25
  term_frequencies.pkl       # Term frequency cache
  chunk_embeddings.npy       # Cached chunk embeddings
  chunk_metadata.json        # Chunk metadata mapping
```

---

# ⚙️ Setup

## 1️⃣ Create Virtual Environment

```bash
uv venv
source .venv/bin/activate
````

## 2️⃣ Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or manually add packages:

```bash
uv add nltk sentence-transformers numpy pillow python-dotenv google-genai
```

**Dependencies:**
- `nltk` - Text processing and tokenization
- `sentence-transformers` - Semantic embeddings (all-MiniLM-L6-v2, CLIP)
- `numpy` - Numerical operations
- `pillow` - Image processing
- `python-dotenv` - Environment variable management
- `google-genai` - Gemini API client

## 3️⃣ Add Gemini API Key

Create a `.env` file (or copy `.env.example`):

```bash
cp .env.example .env
```

Then add your API key:

```
GEMINI_API_KEY="your_actual_api_key_here"
```

⚠️ **Important:** `.env` is automatically excluded via `.gitignore` to protect your API key.

---

# 🧪 Example Commands

## Keyword Search (BM25)

```bash
uv run cli/keyword_search_cli.py bm25search "bear attack"
```

## Semantic Search

```bash
uv run cli/semantic_search_cli.py search_chunked "family bear movie"
```

## Hybrid Search (RRF)

```bash
uv run cli/hybrid_search_cli.py rrf-search "zombie apocalypse"
```

## Hybrid with Spell Enhancement

```bash
uv run cli/hybrid_search_cli.py rrf-search "Padington" --enhance spell
```

## Hybrid with Cross Encoder Reranking

```bash
uv run cli/hybrid_search_cli.py rrf-search "family bear movie" --rerank-method cross_encoder
```

## Evaluation

```bash
uv run cli/evaluation_cli.py --limit 5
```

## RAG Answer

```bash
uv run cli/augmented_generation_cli.py rag "best dinosaur movies"
```

## Multimodal Image Search

```bash
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

---

# 🧠 What This Project Demonstrates

This repository demonstrates practical mastery of modern RAG concepts taught in Boot.dev's course:

* **Information Retrieval fundamentals** - BM25, TF-IDF, inverted indices
* **Vector databases & embedding search** - Semantic similarity, cosine distance
* **Hybrid ranking strategies** - RRF, weighted fusion, score normalization
* **LLM query enhancement** - Spell correction, query rewriting, expansion
* **Reranking systems** - Cross-encoder and LLM-based reranking
* **Evaluation methodology** - Precision@K, Recall@K, F1 scores
* **Retrieval-Augmented Generation** - Context-aware LLM responses
* **Multimodal AI systems** - CLIP embeddings for image+text search
* **Production patterns** - Caching, chunking, error handling

This is not just a demo — it's a **full retrieval stack built step-by-step** following industry best practices.

---

# 📈 Future Improvements

Potential enhancements for production deployment:

* **Vector Database** - Replace file-based storage with PGVector, Weaviate, or Pinecone
* **HNSW Indexing** - Add approximate nearest neighbor search for faster retrieval
* **Caching Layer** - Implement Redis for query result caching
* **Async Operations** - Add async LLM calls for better throughput
* **Conversational RAG** - Add chat history and multi-turn conversations
* **Advanced Chunking** - Implement recursive character splitting and document hierarchy
* **Monitoring** - Add observability with LangSmith or similar tools
* **Web UI** - Build a frontend with Streamlit or FastAPI + React
* **Agentic Workflows** - Add recursive RAG and self-improving search

---

# 📜 License

MIT License

---

# 👨‍💻 About

Built as part of [Boot.dev's](https://www.boot.dev/) **Learn Retrieval Augmented Generation** course.

This project represents a complete implementation of a production-ready RAG system, demonstrating advanced search techniques, LLM integration, and modern AI engineering practices.

### 🔗 Learn More

- **Boot.dev RAG Course**: [https://www.boot.dev/](https://www.boot.dev/)
- **Topic**: Retrieval Augmented Generation (RAG)
- **Skills**: Python, NLP, Vector Search, LLM Integration, Evaluation Metrics

If you're interested in search systems, LLMs, or production RAG pipelines — Boot.dev offers hands-on courses that teach these skills through real projects like this one.