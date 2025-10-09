import json
import faiss
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
# Multi-level retrieval constants
MULTILEVEL_RETRIEVAL_K = 200
FUSION_STRATEGY = "rrf"
RRF_CONSTANT = 60
FAISS_INDEX_LEVEL1_PATH = "/path/to/faiss_index_level1.index"
FAISS_INDEX_LEVEL2_PATH = "/path/to/faiss_index_level2.index"
FAISS_INDEX_LEVEL3_PATH = "/path/to/faiss_index_level3.index"
METADATA_JSON_PATH = "/path/to/corpus_metadata.json"
TOP_K = 20

VLLM_API_URL = "http://127.0.0.1:8000/v1/embeddings"
VLLM_MODEL_NAME = "/path/to/embedding_model"
VLLM_API_KEY = ""
VLLM_TIMEOUT = 30.0
TRUNCATE_PROMPT_TOKENS = 2048

# Global variables for storing initialized components
faiss_index_level1 = None
faiss_index_level2 = None
faiss_index_level3 = None
metadata = None
corpus_ids = None

# FastAPI app
app = FastAPI(title="Paper Retrieval API")

# HTTP client configuration
http_client = None

def initialize_components():
    """Initialize FAISS indexes and metadata"""
    global faiss_index_level1, faiss_index_level2, faiss_index_level3, metadata, corpus_ids, http_client

    # Initialize HTTP client
    http_client = httpx.AsyncClient(timeout=30.0)

    # Initialize three-level FAISS indexes
    print("Loading FAISS indexes...")
    faiss_index_level1 = faiss.read_index(FAISS_INDEX_LEVEL1_PATH)
    faiss_index_level2 = faiss.read_index(FAISS_INDEX_LEVEL2_PATH)
    faiss_index_level3 = faiss.read_index(FAISS_INDEX_LEVEL3_PATH)
    print("FAISS indexes loaded.")

    # Load metadata
    print("Loading metadata JSON...")
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    corpus_ids = list(metadata.keys())
    print(f"Loaded {len(corpus_ids)} metadata entries.")
    print("Initialization completed.")

class QueryInput(BaseModel):
    title: str
    abstract: str
    top_k: int = TOP_K

class EmbeddingRequest(BaseModel):
    """Request format for vLLM API"""
    input: str
    model: str
    encoding_format: str = "float"

class EmbeddingResponse(BaseModel):
    """Response format from vLLM API"""
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

def clean_text(text: str) -> str:
    """Clean text"""
    if not text:
        return ""
    text = " ".join(text.split())
    text = text.replace("\n", " ").replace("\t", " ")
    return text.strip()

def fuse_rrf(results1, results2, results3, corpus_ids):
    """Fuse results using Reciprocal Rank Fusion"""
    doc_scores = {}

    # Get rankings from all three results
    for results, weight in [(results1, 1.0), (results2, 1.0), (results3, 1.0)]:
        scores, indices = results
        for rank, idx in enumerate(indices):
            doc_id = corpus_ids[idx]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += weight / (RRF_CONSTANT + rank + 1)

    # Sort by fused scores
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    fused_ids = [doc_id for doc_id, score in sorted_docs]
    fused_scores = [score for doc_id, score in sorted_docs]

    return fused_ids, fused_scores

def multilevel_retrieve_single(query_embedding, top_k):
    """Perform multilevel retrieval for single query"""
    # Normalize query embedding for cosine similarity
    query_embedding_norm = query_embedding.copy().astype(np.float32)
    faiss.normalize_L2(query_embedding_norm.reshape(1, -1))

    # Retrieve from all three indexes
    scores1, indices1 = faiss_index_level1.search(query_embedding_norm.reshape(1, -1), MULTILEVEL_RETRIEVAL_K)
    scores2, indices2 = faiss_index_level2.search(query_embedding_norm.reshape(1, -1), MULTILEVEL_RETRIEVAL_K)
    scores3, indices3 = faiss_index_level3.search(query_embedding_norm.reshape(1, -1), MULTILEVEL_RETRIEVAL_K)

    # Prepare results for fusion
    results1 = (scores1[0], indices1[0])
    results2 = (scores2[0], indices2[0])
    results3 = (scores3[0], indices3[0])

    # Fuse results
    fused_ids, fused_scores = fuse_rrf(results1, results2, results3, corpus_ids)

    # Return top_k results
    return fused_ids[:top_k], fused_scores[:top_k]

async def get_embedding_from_vllm(text: str) -> np.ndarray:
    """Get embedding vector from external vLLM API"""
    try:
        logger.info(f"Requesting embedding for text length: {len(text)}")

        # Use standard OpenAI-compatible format
        request_data = {
            "input": text,
            "model": VLLM_MODEL_NAME,
            "encoding_format": "float"
        }

        # Build request headers
        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if needed
        if VLLM_API_KEY and VLLM_API_KEY not in [None, "your-api-key"]:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

        logger.info(f"Sending request to vLLM API: {VLLM_API_URL}")
        logger.info(f"Request data: {request_data}")

        # Send request to standard endpoint
        response = await http_client.post(
            VLLM_API_URL,
            json=request_data,
            headers=headers
        )

        logger.info(f"vLLM API response status: {response.status_code}")

        if response.status_code != 200:
            error_text = response.text
            logger.error(f"vLLM API request failed: {response.status_code} - {error_text}")

            # Try to parse error message
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_msg = error_json["error"].get("message", "Unknown error")
                    raise HTTPException(
                        status_code=500,
                        detail=f"vLLM API error: {error_msg}"
                    )
            except:
                pass

            raise HTTPException(
                status_code=500,
                detail=f"vLLM API request failed: {response.status_code} - {error_text}"
            )

        # Parse response
        response_data = response.json()
        logger.info(f"vLLM API response keys: {list(response_data.keys())}")

        # Check for errors
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown error")
            logger.error(f"vLLM API returned error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"vLLM API error: {error_msg}")

        # Extract embedding vector - standard OpenAI format
        if "data" not in response_data or len(response_data["data"]) == 0:
            logger.error(f"Invalid embedding response structure: {response_data}")
            raise HTTPException(status_code=500, detail="Invalid embedding response from vLLM API")

        embedding = response_data["data"][0]["embedding"]
        logger.info(f"Received embedding with length: {len(embedding)}")

        emb_np = np.array(embedding, dtype=np.float32)

        # Normalize
        faiss.normalize_L2(emb_np.reshape(1, -1))

        logger.info(f"Successfully processed embedding, shape: {emb_np.shape}")
        return emb_np

    except httpx.TimeoutException as e:
        logger.error(f"vLLM API timeout: {str(e)}")
        raise HTTPException(status_code=500, detail="vLLM API request timeout")
    except httpx.RequestError as e:
        logger.error(f"vLLM API request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"vLLM API request error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_embedding_from_vllm: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup"""
    initialize_components()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown"""
    global http_client
    if http_client:
        await http_client.aclose()

@app.get("/")
async def root():
    """Root path, returns API information"""
    return {
        "message": "Paper Retrieval API",
        "version": "1.0.0",
        "vllm_api_url": VLLM_API_URL,
        "model_name": VLLM_MODEL_NAME
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if FAISS index is available
        if faiss_index_level1 is None:
            return {"status": "unhealthy", "reason": "FAISS index not loaded"}

        # Check if metadata is available
        if metadata is None:
            return {"status": "unhealthy", "reason": "Metadata not loaded"}

        # Test vLLM API connection (optional)
        test_response = await http_client.get(VLLM_API_URL.replace("/v1/embeddings", "/health"))

        return {
            "status": "healthy",
            "faiss_index_size": faiss_index_level1.ntotal,
            "metadata_count": len(corpus_ids),
            "vllm_api_status": "connected" if test_response.status_code == 200 else "disconnected"
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.post("/search")
async def search_paper(query: QueryInput):
    """Search for papers"""
    try:
        logger.info(f"Starting search for query: title='{query.title[:50]}...', abstract='{query.abstract[:50]}...'")

        # Check if components are initialized
        if faiss_index_level1 is None or faiss_index_level2 is None or faiss_index_level3 is None or metadata is None:
            logger.error("Components not initialized")
            raise HTTPException(status_code=500, detail="Components not initialized")

        # Concatenate query text
        query_text = clean_text(query.title) + " " + clean_text(query.abstract)

        if not query_text.strip():
            logger.error("Empty query text after cleaning")
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        logger.info(f"Query text (length {len(query_text)}): {query_text}")

        # Get embedding vector from vLLM API
        emb = await get_embedding_from_vllm(query_text)
        logger.info(f"Embedding shape: {emb.shape}")

        # Use multilevel retrieval fusion
        fused_ids, fused_scores = multilevel_retrieve_single(emb, query.top_k)
        logger.info(f"Multilevel search completed. Retrieved {len(fused_ids)} papers")

        # Build results
        retrieved = []
        for paper_id, score in zip(fused_ids, fused_scores):
            if paper_id in metadata:
                meta = metadata[paper_id]
                retrieved.append({
                    "paper_id": paper_id,
                    "title": meta["title"],
                    "abstract": meta["abstract"],
                    "score": float(score)
                })

        logger.info(f"Retrieved {len(retrieved)} papers")
        return {"results": retrieved, "query": query_text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/batch_search")
async def batch_search_papers(queries: List[QueryInput]):
    """Batch search for papers"""
    try:
        if not queries:
            raise HTTPException(status_code=400, detail="No queries provided")

        # Process multiple queries concurrently
        tasks = [search_paper(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, convert exceptions to error messages
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query_index": i,
                    "error": str(result),
                    "results": []
                })
            else:
                processed_results.append({
                    "query_index": i,
                    "error": None,
                    **result
                })

        return {"batch_results": processed_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "faiss_index_size": faiss_index_level1.ntotal if faiss_index_level1 else 0,
        "metadata_count": len(corpus_ids) if corpus_ids else 0,
        "default_top_k": TOP_K,
        "vllm_config": {
            "api_url": VLLM_API_URL,
            "model_name": VLLM_MODEL_NAME,
            "has_api_key": bool(VLLM_API_KEY and VLLM_API_KEY != "your-api-key")
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)