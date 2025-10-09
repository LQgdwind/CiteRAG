"""
BGE-M3 embedding model evaluation using SentenceTransformer
"""

import json
import lmdb
import numpy as np
import os
import re
import torch
from sentence_transformers import SentenceTransformer
import faiss

# Configuration constants
LMDB_PATH = "/path/to/corpus_index.lmdb"
TEST_DATASET_PATH = "/path/to/test_dataset"
LMDB_MAX_READERS = 1000
BGE_M3_MODEL_PATH = "intfloat/multilingual-e5-large"
ST_BATCH_SIZE = 32
EVAL_TOP_K = [1, 5, 10, 20, 50, 100]
TEXT_SEPARATOR = " "
PROGRESS_PRINT_FREQ = 100
N_GPUS = 4

# Multi-level retrieval constants
MULTILEVEL_RETRIEVAL_K = 200
FUSION_STRATEGY = "rrf"
RRF_CONSTANT = 60
FAISS_INDEX_LEVEL1_PATH = "/path/to/faiss_index_level1.index"
FAISS_INDEX_LEVEL2_PATH = "/path/to/faiss_index_level2.index"
FAISS_INDEX_LEVEL3_PATH = "/path/to/faiss_index_level3.index"

# Section extraction patterns
INTRO_PATTERNS = [
    r"introduction",
    r"1\.\s*introduction", 
    r"1\s+introduction",
    r"i\.\s*introduction"
]

CONCLUSION_PATTERNS = [
    r"conclusion",
    r"conclusions", 
    r"summary",
    r"discussion",
    r"final remarks"
]

def load_st_model():
    """Initialize SentenceTransformer model for embedding"""
    print("Loading SentenceTransformer BGE-M3 model")
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(BGE_M3_MODEL_PATH, device=device)
    print("SentenceTransformer model loaded successfully")
    return model

def get_st_embeddings(model, texts):
    """Generate embeddings using SentenceTransformer model with batching"""
    print(f"Encoding {len(texts)} texts with SentenceTransformer")
    
    all_embeddings = []
    
    for i in range(0, len(texts), ST_BATCH_SIZE):
        batch_texts = texts[i:i + ST_BATCH_SIZE]
        
        batch_embeddings = model.encode(batch_texts, batch_size=ST_BATCH_SIZE, normalize_embeddings=True)
        all_embeddings.append(batch_embeddings)
        
        if (i // ST_BATCH_SIZE + 1) % PROGRESS_PRINT_FREQ == 0:
            print(f"Encoded {i + len(batch_texts)} texts")
    
    return np.vstack(all_embeddings)

def clean_text(text):
    """Basic text cleaning"""
    if not text:
        return ""
    text = " ".join(text.split())
    text = text.replace("\n", " ").replace("\t", " ")
    return text.strip()

def extract_section_from_dict(sections_dict, patterns):
    """Extract section content by matching keys in sections dictionary"""
    if not sections_dict or not isinstance(sections_dict, dict):
        return ""
    
    for key, content in sections_dict.items():
        key_lower = key.lower().strip()
        
        for pattern in patterns:
            if re.search(pattern, key_lower):
                return clean_text(str(content)) if content else ""
    
    return ""

def extract_introduction(sections):
    """Extract introduction section from sections dictionary"""
    return extract_section_from_dict(sections, INTRO_PATTERNS)

def extract_conclusion(sections):
    """Extract conclusion section from sections dictionary"""  
    return extract_section_from_dict(sections, CONCLUSION_PATTERNS)

def build_multilevel_corpus(corpus_ids):
    """Build three-level corpus texts from document data"""
    print("Building multilevel corpus")
    
    level1_texts = []
    level2_texts = []
    level3_texts = []
    
    # Re-open LMDB to get full document data including sections
    env = lmdb.open(LMDB_PATH, readonly=True, max_readers=LMDB_MAX_READERS)
    
    with env.begin() as txn:
        for doc_id in corpus_ids:
            doc_key = doc_id.encode()
            doc_value = txn.get(doc_key)
            
            if doc_value is None:
                # Skip documents not found in LMDB
                continue
            
            doc_data = json.loads(doc_value.decode())
            
            # Extract basic fields
            title = clean_text(doc_data.get("title", ""))
            abstract = clean_text(doc_data.get("abstract", ""))
            sections = doc_data.get("sections", {})
            
            # Parse sections if string
            if isinstance(sections, str):
                try:
                    sections = json.loads(sections) if sections else {}
                except:
                    sections = {}
            
            # Extract sections
            introduction = extract_introduction(sections)
            conclusion = extract_conclusion(sections)
            
            # Build level texts
            level1_text = TEXT_SEPARATOR.join(filter(None, [title, abstract]))
            level2_text = TEXT_SEPARATOR.join(filter(None, [title, abstract, introduction]))
            level3_text = TEXT_SEPARATOR.join(filter(None, [title, abstract, introduction, conclusion]))
            
            level1_texts.append(level1_text)
            level2_texts.append(level2_text)
            level3_texts.append(level3_text)
    
    env.close()
    
    print(f"Built multilevel corpus: Level1={len(level1_texts)}, Level2={len(level2_texts)}, Level3={len(level3_texts)}")
    return level1_texts, level2_texts, level3_texts

def load_corpus_ids():
    """Load corpus IDs from LMDB index"""
    print("Loading corpus IDs from LMDB")
    env = lmdb.open(LMDB_PATH, readonly=True, max_readers=LMDB_MAX_READERS)
    
    corpus_ids = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            paper_id = key.decode()
            corpus_ids.append(paper_id)
    
    env.close()
    print(f"Loaded {len(corpus_ids)} document IDs from corpus")
    return corpus_ids

def load_test_queries():
    """Load test queries with all positive references"""
    print("Loading test queries")
    test_queries = []
    
    total_positive_refs = 0
    total_positive_ids = 0
    total_queries = 0
    
    for filename in os.listdir(TEST_DATASET_PATH):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(TEST_DATASET_PATH, filename)
        with open(file_path, "r") as f:
            query_data = json.load(f)
        
        # Check if query has required fields - skip if missing title or abstract
        if "reference_labels" not in query_data:
            continue

        if not "title" in query_data or not "abstract" in query_data:
            print(f"Skipping {filename}: missing title or abstract")
            continue
            
        # Extract positive paper IDs from reference_labels
        positive_ids = []
        for ref in query_data["reference_labels"]:
            if "paper_id" in ref:
                positive_ids.append(ref["paper_id"])
        
        # Count all references (including those without paper_id) for denominator
        all_positive_refs = len(query_data["reference_labels"])
        positive_ids_count = len(positive_ids)
        
        # Create query text from title + abstract
        title = clean_text(query_data.get("title", ""))
        abstract = clean_text(query_data.get("abstract", ""))
        query_text = title + TEXT_SEPARATOR + abstract
        
        test_queries.append({
            "query_id": query_data["paper_id"],
            "query_text": query_text,
            "positive_ids": positive_ids,
            "total_positive_refs": all_positive_refs
        })
        
        total_positive_refs += all_positive_refs
        total_positive_ids += positive_ids_count
        total_queries += 1
    
    # Print statistics
    avg_positive_refs = total_positive_refs / total_queries if total_queries > 0 else 0
    avg_positive_ids = total_positive_ids / total_queries if total_queries > 0 else 0
    
    print(f"Loaded {total_queries} test queries")
    print(f"Average positive references per query: {avg_positive_refs:.2f}")
    print(f"Average positive references with paper_id per query: {avg_positive_ids:.2f}")
    print(f"Total test samples: {total_queries}")
    
    return test_queries

def build_faiss_index(embeddings, index_path):
    """Build FAISS index from corpus embeddings"""
    print(f"Building FAISS index at {index_path}")
    dimension = embeddings.shape[1]
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

def create_multilevel_indexes(model, level1_texts, level2_texts, level3_texts):
    """Create three FAISS indexes for multilevel corpus"""
    print("Creating multilevel FAISS indexes")
    
    # Generate embeddings for all levels
    print("Generating Level 1 embeddings")
    level1_embeddings = get_st_embeddings(model, level1_texts)
    
    print("Generating Level 2 embeddings") 
    level2_embeddings = get_st_embeddings(model, level2_texts)
    
    print("Generating Level 3 embeddings")
    level3_embeddings = get_st_embeddings(model, level3_texts)
    
    # Build indexes
    index1 = build_faiss_index(level1_embeddings, FAISS_INDEX_LEVEL1_PATH)
    index2 = build_faiss_index(level2_embeddings, FAISS_INDEX_LEVEL2_PATH)
    index3 = build_faiss_index(level3_embeddings, FAISS_INDEX_LEVEL3_PATH)
    
    print("Multilevel FAISS indexes created successfully")
    return index1, index2, index3

def calculate_recall_at_k(retrieved_ids, positive_ids, total_positive_refs, k):
    """Calculate recall@k for single query"""
    top_k_ids = retrieved_ids[:k]
    relevant_retrieved = len(set(top_k_ids) & set(positive_ids))
    return relevant_retrieved / total_positive_refs

def calculate_ndcg_at_k(scores, positive_ids, corpus_ids, k):
    """Calculate NDCG@k for single query"""
    if k == 1:
        # For k=1, NDCG equals 1 if top result is relevant, 0 otherwise
        top_doc_id = corpus_ids[0]
        return 1.0 if top_doc_id in positive_ids else 0.0
    
    # Manual NDCG calculation for k > 1
    dcg = 0.0
    for i in range(min(k, len(corpus_ids))):
        doc_id = corpus_ids[i]
        if doc_id in positive_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
    
    # Ideal DCG (assuming all positives are at top)
    num_positives = min(len(positive_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_positives))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_mrr_at_k(retrieved_ids, positive_ids, k):
    """Calculate MRR@k for single query"""
    top_k_ids = retrieved_ids[:k]
    for i, doc_id in enumerate(top_k_ids):
        if doc_id in positive_ids:
            return 1.0 / (i + 1)  # Reciprocal rank of first relevant item
    return 0.0

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

def fuse_max_score(results1, results2, results3, corpus_ids):
    """Fuse results using maximum score"""
    doc_scores = {}
    
    # Get max scores from all three results
    for results in [results1, results2, results3]:
        scores, indices = results
        for score, idx in zip(scores, indices):
            doc_id = corpus_ids[idx]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = score
            else:
                doc_scores[doc_id] = max(doc_scores[doc_id], score)
    
    # Sort by max scores
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    fused_ids = [doc_id for doc_id, score in sorted_docs]
    fused_scores = [score for doc_id, score in sorted_docs]
    
    return fused_ids, fused_scores

def fuse_sum_score(results1, results2, results3, corpus_ids):
    """Fuse results using sum of scores"""
    doc_scores = {}
    
    # Sum scores from all three results
    for results in [results1, results2, results3]:
        scores, indices = results
        for score, idx in zip(scores, indices):
            doc_id = corpus_ids[idx]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += score
    
    # Sort by sum scores
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    fused_ids = [doc_id for doc_id, score in sorted_docs]
    fused_scores = [score for doc_id, score in sorted_docs]
    
    return fused_ids, fused_scores

def fuse_borda_count(results1, results2, results3, corpus_ids):
    """Fuse results using Borda count"""
    doc_scores = {}
    k = MULTILEVEL_RETRIEVAL_K
    
    # Calculate Borda scores from all three results
    for results in [results1, results2, results3]:
        scores, indices = results
        for rank, idx in enumerate(indices):
            doc_id = corpus_ids[idx]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += k - rank
    
    # Sort by Borda scores
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    fused_ids = [doc_id for doc_id, score in sorted_docs]
    fused_scores = [score for doc_id, score in sorted_docs]
    
    return fused_ids, fused_scores

def fuse_retrieval_results(results1, results2, results3, corpus_ids, strategy):
    """Fuse results from three levels using specified strategy"""
    print(f"Fusing results using {strategy} strategy")
    
    if strategy == "rrf":
        return fuse_rrf(results1, results2, results3, corpus_ids)
    elif strategy == "max_score":
        return fuse_max_score(results1, results2, results3, corpus_ids)
    elif strategy == "sum_score":
        return fuse_sum_score(results1, results2, results3, corpus_ids)
    elif strategy == "borda_count":
        return fuse_borda_count(results1, results2, results3, corpus_ids)
    else:
        print(f"Unknown fusion strategy: {strategy}, using RRF as default")
        return fuse_rrf(results1, results2, results3, corpus_ids)

def multilevel_retrieve(query_embeddings, index1, index2, index3, corpus_ids):
    """Perform retrieval on all three levels and fuse results"""
    print("Starting multilevel retrieval")
    
    # Normalize query embeddings for cosine similarity
    query_embeddings_norm = query_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(query_embeddings_norm)
    
    # Retrieve from all three indexes
    print("Retrieving from Level 1 index")
    scores1, indices1 = index1.search(query_embeddings_norm, MULTILEVEL_RETRIEVAL_K)
    
    print("Retrieving from Level 2 index") 
    scores2, indices2 = index2.search(query_embeddings_norm, MULTILEVEL_RETRIEVAL_K)
    
    print("Retrieving from Level 3 index")
    scores3, indices3 = index3.search(query_embeddings_norm, MULTILEVEL_RETRIEVAL_K)
    
    # Fuse results for each query
    print("Fusing multilevel results")
    fused_results = []
    
    for i in range(len(query_embeddings)):
        results1 = (scores1[i], indices1[i])
        results2 = (scores2[i], indices2[i])
        results3 = (scores3[i], indices3[i])
        
        fused_ids, fused_scores = fuse_retrieval_results(results1, results2, results3, corpus_ids, FUSION_STRATEGY)
        fused_results.append((fused_ids, fused_scores))
    
    # Prepare individual level results for evaluation
    level1_results = []
    level2_results = []
    level3_results = []
    
    for i in range(len(query_embeddings)):
        level1_results.append([corpus_ids[idx] for idx in indices1[i]])
        level2_results.append([corpus_ids[idx] for idx in indices2[i]])
        level3_results.append([corpus_ids[idx] for idx in indices3[i]])
    
    return level1_results, level2_results, level3_results, fused_results


def evaluate_single_level_results(retrieved_results, test_queries):
    """Evaluate single level retrieval results"""
    results = {k: {"recall": [], "ndcg": [], "mrr": []} for k in EVAL_TOP_K}
    per_query_results = []
    
    for i, query in enumerate(test_queries):
        retrieved_ids = retrieved_results[i]
        positive_ids = query["positive_ids"]
        total_positive_refs = query["total_positive_refs"]
        
        query_metrics = {}
        
        # Calculate metrics for each k
        for k in EVAL_TOP_K:
            recall_k = calculate_recall_at_k(retrieved_ids, positive_ids, total_positive_refs, k)
            ndcg_k = calculate_ndcg_at_k(None, positive_ids, retrieved_ids, k)
            mrr_k = calculate_mrr_at_k(retrieved_ids, positive_ids, k)
            
            results[k]["recall"].append(recall_k)
            results[k]["ndcg"].append(ndcg_k)
            results[k]["mrr"].append(mrr_k)
            
            query_metrics[str(k)] = {
                "recall": recall_k,
                "ndcg": ndcg_k,
                "mrr": mrr_k
            }
        
        per_query_results.append(query_metrics)
    
    # Average results
    final_results = {}
    for k in EVAL_TOP_K:
        final_results[k] = {
            "recall": np.mean(results[k]["recall"]),
            "ndcg": np.mean(results[k]["ndcg"]),
            "mrr": np.mean(results[k]["mrr"])
        }
    
    return final_results, per_query_results

def evaluate_fused_results(fused_results, test_queries):
    """Evaluate fused retrieval results"""
    results = {k: {"recall": [], "ndcg": [], "mrr": []} for k in EVAL_TOP_K}
    per_query_results = []
    
    for i, query in enumerate(test_queries):
        fused_ids, fused_scores = fused_results[i]
        positive_ids = query["positive_ids"]
        total_positive_refs = query["total_positive_refs"]
        
        query_metrics = {}
        
        # Calculate metrics for each k
        for k in EVAL_TOP_K:
            recall_k = calculate_recall_at_k(fused_ids, positive_ids, total_positive_refs, k)
            ndcg_k = calculate_ndcg_at_k(None, positive_ids, fused_ids, k)
            mrr_k = calculate_mrr_at_k(fused_ids, positive_ids, k)
            
            results[k]["recall"].append(recall_k)
            results[k]["ndcg"].append(ndcg_k)
            results[k]["mrr"].append(mrr_k)
            
            query_metrics[str(k)] = {
                "recall": recall_k,
                "ndcg": ndcg_k,
                "mrr": mrr_k
            }
        
        per_query_results.append(query_metrics)
    
    # Average results
    final_results = {}
    for k in EVAL_TOP_K:
        final_results[k] = {
            "recall": np.mean(results[k]["recall"]),
            "ndcg": np.mean(results[k]["ndcg"]),
            "mrr": np.mean(results[k]["mrr"])
        }
    
    return final_results, per_query_results

def evaluate_multilevel_retrieval(query_embeddings, index1, index2, index3, test_queries, corpus_ids):
    """Evaluate multilevel retrieval performance"""
    print("Starting multilevel retrieval evaluation")
    
    # Perform multilevel retrieval
    level1_results, level2_results, level3_results, fused_results = multilevel_retrieve(
        query_embeddings, index1, index2, index3, corpus_ids
    )
    
    # Evaluate each level separately
    print("Evaluating Level 1 results")
    level1_metrics, level1_per_query = evaluate_single_level_results(level1_results, test_queries)
    
    print("Evaluating Level 2 results")
    level2_metrics, level2_per_query = evaluate_single_level_results(level2_results, test_queries)
    
    print("Evaluating Level 3 results")
    level3_metrics, level3_per_query = evaluate_single_level_results(level3_results, test_queries)
    
    print("Evaluating fused results")
    fused_metrics, fused_per_query = evaluate_fused_results(fused_results, test_queries)
    
    # Collect all per-query results
    all_per_query_results = {
        "bge_level1": level1_per_query,
        "bge_level2": level2_per_query,
        "bge_level3": level3_per_query,
        "bge_fused": fused_per_query
    }
    
    return ((level1_metrics, level2_metrics, level3_metrics, fused_metrics),
            all_per_query_results)

def run_bge_evaluation():
    """Run BGE-M3 embedding evaluation using SentenceTransformer"""
    print("Starting BGE-M3 SentenceTransformer evaluation")
    
    # Load model
    model = load_st_model()
    
    # Load corpus and test data
    corpus_ids = load_corpus_ids()
    test_queries = load_test_queries()
    query_texts = [query["query_text"] for query in test_queries]
    
    # Build multilevel corpus
    level1_texts, level2_texts, level3_texts = build_multilevel_corpus(corpus_ids)
    
    # Generate query embeddings
    print("Generating query embeddings")  
    query_embeddings = get_st_embeddings(model, query_texts)
    
    # Create multilevel indexes
    index1, index2, index3 = create_multilevel_indexes(model, level1_texts, level2_texts, level3_texts)
    
    # Evaluate multilevel retrieval
    aggregated_results, all_per_query_results = evaluate_multilevel_retrieval(
        query_embeddings, index1, index2, index3, test_queries, corpus_ids
    )
    
    return aggregated_results

def print_results(model_name, results):
    """Print evaluation results"""
    print(f"\nResults for {model_name}:")
    print("=" * 70)
    for k in EVAL_TOP_K:
        recall = results[k]["recall"]
        ndcg = results[k]["ndcg"]
        mrr = results[k]["mrr"]
        print(f"Recall@{k:3d}: {recall:.4f} | NDCG@{k:3d}: {ndcg:.4f} | MRR@{k:3d}: {mrr:.4f}")

def main():
    """Main evaluation function"""
    print("Starting BGE-M3 SentenceTransformer multilevel embedding evaluation")
    
    level1_metrics, level2_metrics, level3_metrics, fused_metrics = run_bge_evaluation()
    
    # Print results for all levels
    print_results("Level 1 (title+abstract)", level1_metrics)
    print_results("Level 2 (title+abstract+intro)", level2_metrics)  
    print_results("Level 3 (title+abstract+intro+conclusion)", level3_metrics)
    print_results(f"Multi-level Fusion ({FUSION_STRATEGY})", fused_metrics)
    
    print("BGE-M3 multilevel evaluation completed")

if __name__ == "__main__":
    main()
