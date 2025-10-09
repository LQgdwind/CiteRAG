# Multi-level Embedding Retrieval System

This project implements a multi-level document retrieval system for academic papers using various embedding models and traditional IR methods.

## Overview

The system evaluates document retrieval performance using:
- **Deep Learning Models**: Qwen3-Embedding-8B (via vLLM) and multilingual-e5-large (via SentenceTransformer)
- **Traditional Methods**: TF-IDF and BM25
- **Multi-level Strategy**: Combines title+abstract, title+abstract+introduction, and title+abstract+introduction+conclusion representations

## Requirements

Install required dependencies:
```bash
pip install torch vllm sentence-transformers faiss-gpu lmdb numpy gensim nltk
pip install ms-swift  # for training
pip install fastapi uvicorn httpx  # for API service
```

## Project Structure

```
.
├── train.sh                    # Training script
├── train_example.jsonl         # Example training data (200 samples)
├── api.py                      # FastAPI service for paper retrieval
└── eval/
    ├── test_vllm.py           # Evaluation using vLLM (Qwen model)
    ├── test_st.py             # Evaluation using SentenceTransformer
    └── test_traditional.py    # Evaluation using TF-IDF and BM25
```

## Data Preparation

### 1. Corpus Data
Prepare your corpus in LMDB format with the following structure:
- Key: paper_id (string)
- Value: JSON object containing:
  - `title`: paper title
  - `abstract`: paper abstract
  - `sections`: dictionary of section names to content (must include introduction and conclusion)

### 2. Test Dataset
Prepare test queries as JSON files in a directory:
```json
{
  "paper_id": "unique_id",
  "title": "paper title",
  "abstract": "paper abstract",
  "reference_labels": [
    {"paper_id": "referenced_paper_id"},
    ...
  ]
}
```

### 3. Training Data (for fine-tuning)
Prepare training data in JSONL format where each line contains:
```json
{
  "query": "query text",
  "response": "positive document text",
  "negative_response": ["negative doc 1", "negative doc 2", ...]
}
```

**Example File Included**: `train_example.jsonl` contains 200 sample training instances demonstrating the expected data format. This file can be used for:
- Understanding the training data structure
- Testing the training pipeline
- Quick experiments with small-scale training

For full-scale training, prepare a larger dataset following the same format.

### 4. Metadata File (for API service)
Prepare a JSON file containing metadata for all corpus documents:
```json
{
  "paper_id_1": {
    "title": "paper title",
    "abstract": "paper abstract"
  },
  "paper_id_2": {
    ...
  }
}
```

## Configuration

Before running, update the configuration constants at the top of each script:

### Training (train.sh)
- `nproc_per_node`: Number of GPUs
- `--model`: Path to Qwen3-Embedding-8B model
- `--dataset`: Path to training data JSONL file
- `--output_dir`: Directory for checkpoints and logs
- `--per_device_train_batch_size`: Batch size (adjust to avoid OOM)

### Evaluation Scripts
Update these paths in each evaluation script:
- `LMDB_PATH`: Path to corpus LMDB database
- `TEST_DATASET_PATH`: Path to test queries directory
- `QWEN_MODEL_PATH` or `BGE_M3_MODEL_PATH`: Path to embedding model
- `FAISS_INDEX_LEVEL*_PATH`: Paths for FAISS indexes (will be created)
- `N_GPUS`: Number of GPUs for inference

### API Service (api.py)
Update these configurations:
- `FAISS_INDEX_LEVEL*_PATH`: Paths to pre-built FAISS indexes
- `METADATA_JSON_PATH`: Path to corpus metadata JSON file
- `VLLM_API_URL`: URL of the vLLM embedding API endpoint
- `VLLM_MODEL_NAME`: Model name or path used by vLLM
- `VLLM_API_KEY`: API key if required (optional)
- `TOP_K`: Default number of results to return

## Usage

### Training
Train the embedding model:
```bash
bash train.sh
```

### Evaluation

Run evaluation with different methods:

**vLLM (Qwen model):**
```bash
python eval/test_vllm.py
```

**SentenceTransformer:**
```bash
python eval/test_st.py
```

**Traditional IR methods:**
```bash
python eval/test_traditional.py
```

Each evaluation script will:
1. Load the corpus and test queries
2. Build multi-level document representations
3. Create embeddings/indexes for all three levels
4. Perform retrieval and fusion
5. Calculate metrics (Recall@K, NDCG@K, MRR@K) for K=[1, 5, 10, 20, 50, 100]
6. Print results to console

## Multi-level Fusion

The system uses Reciprocal Rank Fusion (RRF) by default to combine results from three levels. You can change the fusion strategy by modifying the `FUSION_STRATEGY` constant:
- `rrf`: Reciprocal Rank Fusion (default)
- `max_score`: Maximum score across levels
- `sum_score`: Sum of scores across levels
- `borda_count`: Borda count voting

## Evaluation Metrics

- **Recall@K**: Proportion of relevant documents retrieved in top K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **MRR@K**: Mean Reciprocal Rank at K

## API Service

The project includes a FastAPI-based REST API for real-time paper retrieval.

### Architecture

The API service operates separately from the embedding model:
- **Embedding Service**: Run vLLM separately to serve the embedding model
- **Retrieval Service**: The FastAPI service (`api.py`) handles retrieval using pre-built FAISS indexes

### Setup

1. **Start the vLLM embedding service** (in a separate terminal):
```bash
vllm serve /path/to/embedding_model \
    --task embed \
    --port 8000 \
    --tensor-parallel-size <NUM_GPUS>
```

2. **Start the retrieval API service**:
```bash
python api.py
# Or with uvicorn:
uvicorn api:app --host 0.0.0.0 --port 8001
```

### API Endpoints

**GET /**
- Root endpoint, returns API information

**GET /health**
- Health check endpoint
- Returns status of FAISS indexes, metadata, and vLLM API connection

**GET /stats**
- Returns system statistics (index size, metadata count, configuration)

**POST /search**
- Search for papers based on title and abstract
- Request body:
```json
{
  "title": "paper title",
  "abstract": "paper abstract",
  "top_k": 20
}
```
- Response:
```json
{
  "results": [
    {
      "paper_id": "id",
      "title": "title",
      "abstract": "abstract",
      "score": 0.95
    },
    ...
  ],
  "query": "processed query text"
}
```

**POST /batch_search**
- Batch search for multiple queries
- Request body: Array of query objects
- Response: Array of search results

### Usage Example

```python
import requests

# Single search
response = requests.post(
    "http://localhost:8001/search",
    json={
        "title": "Deep Learning for NLP",
        "abstract": "This paper presents...",
        "top_k": 10
    }
)
results = response.json()

# Batch search
batch_response = requests.post(
    "http://localhost:8001/batch_search",
    json=[
        {"title": "Paper 1", "abstract": "Abstract 1"},
        {"title": "Paper 2", "abstract": "Abstract 2"}
    ]
)
batch_results = batch_response.json()
```

## Notes

- The vLLM script requires GPU(s) for inference
- FAISS indexes will be saved to disk and can be reused
- Adjust batch sizes and GPU settings based on your hardware
- The traditional methods script downloads NLTK data automatically
- For API service, pre-build FAISS indexes using evaluation scripts before starting the service

