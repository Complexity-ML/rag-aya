# RAG-Aya: Multilingual RAG Pipeline

A Retrieval-Augmented Generation pipeline built around [Cohere's Aya](https://cohere.com/research/aya) model family, designed for multilingual question answering across diverse languages.

## Motivation

Most RAG systems are English-centric. This project leverages Aya 23 — a multilingual generative model — paired with Cohere's `embed-multilingual-v3.0` embeddings to build a RAG pipeline that works across languages. The goal is to make knowledge retrieval and generation accessible beyond high-resource languages.

## Architecture

```
Documents (Wikipedia, text files)
        |
    [ Chunker ]          character-based overlapping chunks
        |
    [ Embedder ]          Cohere embed-multilingual-v3.0
        |
    [ Retriever ]         cosine similarity over numpy vectors
        |
    [ Generator ]         Cohere Aya 23 (8B) with RAG context
        |
      Answer
```

| Module         | File            | Description                                  |
|----------------|-----------------|----------------------------------------------|
| Config         | `config.py`     | Pipeline settings (API keys, model, chunking)|
| Chunker        | `chunker.py`    | Split documents into overlapping text chunks |
| Embedder       | `embedder.py`   | Cohere multilingual embeddings (batched)     |
| Retriever      | `retriever.py`  | Vector index with cosine similarity search   |
| Generator      | `generator.py`  | Aya 23 generation with document context      |
| Data Loader    | `data_loader.py`| Load Wikipedia articles via HuggingFace      |
| Evaluation     | `evaluate.py`   | RAGAS metrics + simple fallback evaluation   |
| REST Server    | `server.py`     | aiohttp API for indexing, search, generation |
| CLI            | `main.py`       | Command-line interface for the full pipeline |

## Setup

### Prerequisites

- Python 3.9+
- A [Cohere API key](https://dashboard.cohere.com/api-keys)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy the example env file and add your API key:

```bash
cp .env.example .env
# Edit .env and set COHERE_API_KEY=your_key_here
```

## Usage

### CLI

```bash
# 1. Index Wikipedia documents (EN + FR)
python main.py index

# 2. Query the index
python main.py query "What is machine learning?"

# 3. Run evaluation
python main.py eval

# 4. Interactive demo
python main.py demo
```

CLI options:

```
--top-k         Number of chunks to retrieve (default: 5)
--max-docs      Max documents per language (default: 50)
--chunk-size    Chunk size in characters (default: 512)
--index-path    Path to save/load index (default: index/)
--model         Aya model name (default: c4ai-aya-23-8b)
```

### REST API

```bash
python server.py --port 8080
```

**Endpoints:**

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Health check + chunk count           |
| GET    | `/stats`    | Index statistics                     |
| POST   | `/index`    | Index documents (Wikipedia or text)  |
| POST   | `/search`   | Search chunks by query               |
| POST   | `/generate` | RAG: retrieve context + generate     |
| POST   | `/eval`     | Run evaluation on sample queries     |

**Example — generate an answer:**

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?", "k": 5}'
```

## Evaluation

The pipeline includes two evaluation modes:

- **Simple metrics**: answer rate, average answer length, context count
- **RAGAS** (optional): faithfulness, answer relevancy, context precision

```bash
python main.py eval
```

## Languages

Currently configured for **English** and **French**. Additional languages can be added by updating the `languages` list in `config.py` or passing them at runtime:

```bash
# CLI
python main.py index --max-docs 30

# API
curl -X POST http://localhost:8080/index \
  -H "Content-Type: application/json" \
  -d '{"source": "wikipedia", "languages": ["en", "fr", "ar", "sw"], "max_per_lang": 30}'
```

Both the embedding model (`embed-multilingual-v3.0`) and the generation model (Aya 23) support 100+ languages, so extending coverage requires only providing data in the target language.

## Project Structure

```
rag-aya/
  config.py          # Configuration dataclass
  chunker.py         # Document chunking
  embedder.py        # Cohere embeddings
  retriever.py       # Vector search + persistence
  generator.py       # Aya generation
  data_loader.py     # Wikipedia / file loaders
  evaluate.py        # RAGAS + simple evaluation
  main.py            # CLI entry point
  server.py          # REST API server
  requirements.txt   # Dependencies
  .env.example       # Environment template
```

## License

MIT
