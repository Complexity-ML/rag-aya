# RAG-Aya: Multilingual RAG Pipeline

A Retrieval-Augmented Generation pipeline built around [Cohere's Aya](https://cohere.com/research/aya) model family, designed for multilingual question answering across diverse languages.

Built for the [Expedition Tiny Aya](https://cohere.com/research/aya) program.

## Motivation

Most RAG systems are English-centric. This project leverages Aya 23 — a multilingual generative model — paired with multilingual embeddings to build a RAG pipeline that works across languages. The goal is to make knowledge retrieval and generation accessible beyond high-resource languages.

## Architecture

```
Documents (WMT25 parallel data, text files)
        |
    [ Chunker ]          character-based overlapping chunks
        |
    [ Embedder ]          Cohere API or local sentence-transformers
        |
    [ Retriever ]         cosine similarity + LRU cache
        |
    [ Generator ]         Cohere Aya 23 API or local GGUF engine
        |
      Answer (streamed via SSE)
```

### Two modes

| Mode | Embedder | Generator | Requires |
|------|----------|-----------|----------|
| **API** | Cohere `embed-multilingual-v3.0` | Cohere Aya 23 (8B) | `COHERE_API_KEY` |
| **Local** | `paraphrase-multilingual-MiniLM-L12-v2` | Custom C engine + GGUF model | GGUF file |

Local mode is auto-detected: drop a `.gguf` file in the project root and everything works without flags.

### Modules

| Module         | File            | Description                                  |
|----------------|-----------------|----------------------------------------------|
| Config         | `config.py`     | Pipeline settings (API keys, model, chunking)|
| Pipeline       | `pipeline.py`   | Factory: backend selection (API vs local)    |
| Chunker        | `chunker.py`    | Split documents into overlapping text chunks |
| Embedder       | `embedder.py`   | Cohere API or local sentence-transformers    |
| Retriever      | `retriever.py`  | Vector index with cosine similarity + LRU cache |
| Generator      | `generator.py`  | Aya API or local C engine (SSE streaming)    |
| Data Loader    | `data_loader.py`| WMT25 parallel data via mtdata               |
| Evaluation     | `evaluate.py`   | RAGAS metrics + simple fallback evaluation   |
| REST Server    | `server.py`     | aiohttp API for indexing, search, generation |
| CLI            | `main.py`       | Command-line interface for the full pipeline |
| C Engine       | `engine/`       | Custom GGUF inference server (Q4_K/Q6_K, AVX2/NEON, OpenMP) |

## Setup

### Prerequisites

- Python 3.9+
- For API mode: a [Cohere API key](https://dashboard.cohere.com/api-keys)
- For local mode: a GGUF model file (e.g. `tiny-aya-global-q4_k_m.gguf`)

### Installation

```bash
pip install -r requirements.txt

# For WMT data loading
pip install mtdata==0.4.3
```

### Building the C engine (local mode)

```bash
cd engine && make
```

Requires `gcc` with AVX2 support (Windows: MSYS2/ucrt64, macOS: Xcode CLI tools, Linux: gcc).
Apple Silicon builds automatically enable NEON SIMD.

### Configuration

For API mode, copy the example env file and add your API key:

```bash
cp .env.example .env
# Edit .env and set COHERE_API_KEY=your_key_here
```

For local mode, just place a `.gguf` file in the project root — no config needed.

## Usage

### CLI

```bash
# 1. Index WMT25 parallel data
python main.py index --lang-pair eng-fra --wmt-dataset news_commentary

# 2. Query the index
python main.py query "What is the role of gold in the global economy?"

# 3. Interactive demo
python main.py demo

# 4. Run evaluation
python main.py eval
```

GGUF auto-detection: if a `.gguf` file exists in the current directory, local mode activates automatically. No `--local` or `--gguf` flags needed.

CLI options:

```
--top-k           Number of chunks to retrieve (default: 5)
--chunk-size      Chunk size in characters (default: 512)
--index-path      Path to save/load index (default: index/)
--model           Aya model name (default: c4ai-aya-23-8b)
--local           Force local embedder + GGUF engine
--gguf PATH       Path to GGUF model file
--engine-port     C engine server port (default: 8089)
--lang-pair       WMT language pair (default: eng-ara)
--wmt-dataset     Small dataset: news_commentary, ted_talks, wikimatrix
--wmt-max-lines   Max parallel lines to load (default: 500)
```

### REST API

```bash
python server.py
```

**Endpoints:**

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Health check + chunk count           |
| GET    | `/stats`    | Index statistics                     |
| POST   | `/index`    | Index documents (WMT or text)        |
| POST   | `/search`   | Search chunks by query               |
| POST   | `/generate` | RAG: retrieve context + generate     |
| POST   | `/eval`     | Run evaluation on sample queries     |

**Example — generate an answer:**

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?", "k": 5}'
```

## C Inference Engine

The `engine/` directory contains a custom C inference server for running Aya GGUF models locally:

- **Quantization**: Q4_K and Q6_K support with SIMD acceleration
- **SIMD**: AVX2+FMA (x86), NEON (Apple Silicon), scalar fallback
- **Parallelism**: OpenMP for multi-core matvec operations
- **Streaming**: SSE (Server-Sent Events) for token-by-token output
- **BPE tokenizer**: with Cohere chat template support
- **Repetition penalty**: configurable (default 1.2)

```bash
# Build
cd engine && make

# Run standalone
./aya-server ../tiny-aya-global-q4_k_m.gguf 8089

# Test
curl -X POST http://localhost:8089/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50, "temperature": 0.7}'
```

## Evaluation

The pipeline includes two evaluation modes:

- **Simple metrics**: answer rate, average answer length, context count
- **RAGAS** (optional): faithfulness, answer relevancy, context precision

```bash
python main.py eval
```

## Languages

Supports 100+ languages via:
- **Embeddings**: Cohere `embed-multilingual-v3.0` or `paraphrase-multilingual-MiniLM-L12-v2`
- **Generation**: Aya 23 (API) or Tiny Aya (local GGUF)
- **Data**: WMT25 parallel corpora (eng-ara, eng-fra, eng-zho, eng-bho, eng-ukr, eng-kor, eng-jpn, eng-ces, eng-rus)

## Project Structure

```
rag-aya/
  config.py          # Configuration dataclass
  pipeline.py        # Factory: backend selection
  chunker.py         # Document chunking
  embedder.py        # Cohere API / local embeddings
  retriever.py       # Vector search + LRU cache + persistence
  generator.py       # Aya API / C engine (SSE streaming)
  data_loader.py     # WMT25 data loader via mtdata
  evaluate.py        # RAGAS + simple evaluation
  main.py            # CLI entry point
  server.py          # REST API server
  logger.py          # vLLM-style structured logging
  engine/            # Custom C inference server
    main.c           # HTTP server + tokenizer + generation loop
    model.c          # Transformer forward pass
    gguf.c           # GGUF file parser
    quant.h          # Q4_K/Q6_K SIMD kernels
    model.h          # Model structures
    gguf.h           # GGUF structures
    Makefile         # Build system (Windows/macOS/Linux)
```

## License

MIT
