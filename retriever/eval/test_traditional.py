"""
Traditional IR methods evaluation using TF-IDF and BM25
"""

import json
import lmdb
import numpy as np
import os
import re
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Configuration constants
LMDB_PATH = "/path/to/corpus_index.lmdb"
TEST_DATASET_PATH = "/path/to/test_dataset"
LMDB_MAX_READERS = 1000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
BM25_K1 = 1.2
BM25_B = 0.75
RETRIEVAL_BATCH_SIZE = 1000
EVAL_TOP_K = [1, 5, 10, 20, 50, 100]
TEXT_SEPARATOR = " "
PROGRESS_PRINT_FREQ = 100

# Multi-level retrieval constants
MULTILEVEL_RETRIEVAL_K = 200
FUSION_STRATEGY = "rrf"
RRF_CONSTANT = 60

# Text preprocessing constants
REMOVE_STOPWORDS = True
APPLY_STEMMING = True
MIN_WORD_LENGTH = 2

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

def preprocess_text(text):
    """Preprocess text for traditional IR methods"""
    if not text:
        return []
    
    # Convert to lowercase and tokenize
    text = text.lower()
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic tokens and short words
    tokens = [token for token in tokens if token.isalpha() and len(token) >= MIN_WORD_LENGTH]
    
    # Remove stopwords if enabled
    if REMOVE_STOPWORDS:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming if enabled
    if APPLY_STEMMING:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

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

def load_traditional_models():
    """Initialize traditional IR models"""
    print("Loading traditional IR models")
    print("Traditional IR models loaded successfully")
    return {"tfidf": None, "bm25": None}

def build_traditional_indexes(corpus_texts_list, corpus_ids):
    """Build traditional IR indexes for all levels"""
    print("Building traditional IR indexes")
    
    indexes = {}
    
    for level, corpus_texts in enumerate(corpus_texts_list, 1):
        print(f"Building Level {level} indexes")
        
        # Preprocess corpus texts
        print(f"Preprocessing Level {level} corpus texts")
        processed_corpus = []
        for i, text in enumerate(corpus_texts):
            tokens = preprocess_text(text)
            processed_corpus.append(tokens)
            
            if (i + 1) % PROGRESS_PRINT_FREQ == 0:
                print(f"Preprocessed {i + 1} documents")
        
        # Build dictionary and corpus for gensim
        dictionary = corpora.Dictionary(processed_corpus)
        dictionary.filter_extremes(no_below=TFIDF_MIN_DF, no_above=TFIDF_MAX_DF)
        corpus = [dictionary.doc2bow(tokens) for tokens in processed_corpus]
        
        # Build TF-IDF model
        print(f"Building TF-IDF model for Level {level}")
        tfidf_model = TfidfModel(corpus)
        tfidf_corpus = tfidf_model[corpus]
        tfidf_index = similarities.SparseMatrixSimilarity(tfidf_corpus, num_features=len(dictionary))
        
        # Build BM25 model
        print(f"Building BM25 model for Level {level}")
        bm25_model = OkapiBM25Model(corpus)
        bm25_corpus = bm25_model[corpus]
        bm25_index = SparseMatrixSimilarity(bm25_corpus, num_features=len(dictionary))
        
        indexes[f"level{level}"] = {
            "dictionary": dictionary,
            "corpus": corpus,
            "processed_corpus": processed_corpus,
            "tfidf_model": tfidf_model,
            "tfidf_index": tfidf_index,
            "bm25_model": bm25_model,
            "bm25_index": bm25_index
        }
        
        print(f"Level {level} indexes built successfully")
    
    print("Traditional IR indexes created successfully")
    return indexes

def retrieve_with_tfidf(query_text, level_index, corpus_ids, k):
    """Retrieve documents using TF-IDF"""
    dictionary = level_index["dictionary"]
    tfidf_model = level_index["tfidf_model"]
    tfidf_index = level_index["tfidf_index"]
    
    # Preprocess query
    query_tokens = preprocess_text(query_text)
    query_bow = dictionary.doc2bow(query_tokens)
    query_tfidf = tfidf_model[query_bow]
    
    # Get similarities
    sims = tfidf_index[query_tfidf]
    sims = list(enumerate(sims))
    sims.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k results
    top_k_results = sims[:k]
    retrieved_ids = [corpus_ids[idx] for idx, score in top_k_results]
    scores = [score for idx, score in top_k_results]
    
    return retrieved_ids, scores

def retrieve_with_bm25(query_text, level_index, corpus_ids, k):
    """Retrieve documents using BM25"""
    dictionary = level_index["dictionary"]
    bm25_model = level_index["bm25_model"]
    bm25_index = level_index["bm25_index"]
    
    # Preprocess query
    query_tokens = preprocess_text(query_text)
    query_bow = dictionary.doc2bow(query_tokens)
    query_bm25 = bm25_model[query_bow]
    
    # Get similarities
    sims = bm25_index[query_bm25]
    sims = list(enumerate(sims))
    sims.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k results
    top_k_results = sims[:k]
    retrieved_ids = [corpus_ids[idx] for idx, score in top_k_results]
    scores = [score for idx, score in top_k_results]
    
    return retrieved_ids, scores

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
        retrieved_ids, scores = results
        for rank, doc_id in enumerate(retrieved_ids):
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
        retrieved_ids, scores = results
        for doc_id, score in zip(retrieved_ids, scores):
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
        retrieved_ids, scores = results
        for doc_id, score in zip(retrieved_ids, scores):
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
        retrieved_ids, scores = results
        for rank, doc_id in enumerate(retrieved_ids):
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

def multilevel_retrieve_traditional(query_texts, indexes, corpus_ids, method="tfidf"):
    """Perform traditional multilevel retrieval"""
    print(f"Starting multilevel {method.upper()} retrieval")
    
    level1_results = []
    level2_results = []
    level3_results = []
    fused_results = []
    
    for i, query_text in enumerate(query_texts):
        if (i + 1) % PROGRESS_PRINT_FREQ == 0:
            print(f"Processing query {i + 1}/{len(query_texts)}")
        
        # Retrieve from all three levels
        if method == "tfidf":
            results1 = retrieve_with_tfidf(query_text, indexes["level1"], corpus_ids, MULTILEVEL_RETRIEVAL_K)
            results2 = retrieve_with_tfidf(query_text, indexes["level2"], corpus_ids, MULTILEVEL_RETRIEVAL_K)
            results3 = retrieve_with_tfidf(query_text, indexes["level3"], corpus_ids, MULTILEVEL_RETRIEVAL_K)
        else:  # bm25
            results1 = retrieve_with_bm25(query_text, indexes["level1"], corpus_ids, MULTILEVEL_RETRIEVAL_K)
            results2 = retrieve_with_bm25(query_text, indexes["level2"], corpus_ids, MULTILEVEL_RETRIEVAL_K)
            results3 = retrieve_with_bm25(query_text, indexes["level3"], corpus_ids, MULTILEVEL_RETRIEVAL_K)
        
        # Store individual level results
        level1_results.append(results1[0])  # only IDs
        level2_results.append(results2[0])
        level3_results.append(results3[0])
        
        # Fuse results
        fused_ids, fused_scores = fuse_retrieval_results(results1, results2, results3, corpus_ids, FUSION_STRATEGY)
        fused_results.append((fused_ids, fused_scores))
    
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

def evaluate_multilevel_retrieval_traditional(query_texts, indexes, test_queries, corpus_ids):
    """Evaluate traditional multilevel retrieval performance"""
    print("Starting traditional multilevel retrieval evaluation")
    
    # Evaluate TF-IDF
    print("Evaluating TF-IDF retrieval")
    tfidf_level1, tfidf_level2, tfidf_level3, tfidf_fused = multilevel_retrieve_traditional(
        query_texts, indexes, corpus_ids, method="tfidf"
    )
    
    tfidf_level1_metrics, tfidf_level1_per_query = evaluate_single_level_results(tfidf_level1, test_queries)
    tfidf_level2_metrics, tfidf_level2_per_query = evaluate_single_level_results(tfidf_level2, test_queries)
    tfidf_level3_metrics, tfidf_level3_per_query = evaluate_single_level_results(tfidf_level3, test_queries)
    tfidf_fused_metrics, tfidf_fused_per_query = evaluate_fused_results(tfidf_fused, test_queries)
    
    # Evaluate BM25
    print("Evaluating BM25 retrieval")
    bm25_level1, bm25_level2, bm25_level3, bm25_fused = multilevel_retrieve_traditional(
        query_texts, indexes, corpus_ids, method="bm25"
    )
    
    bm25_level1_metrics, bm25_level1_per_query = evaluate_single_level_results(bm25_level1, test_queries)
    bm25_level2_metrics, bm25_level2_per_query = evaluate_single_level_results(bm25_level2, test_queries)
    bm25_level3_metrics, bm25_level3_per_query = evaluate_single_level_results(bm25_level3, test_queries)
    bm25_fused_metrics, bm25_fused_per_query = evaluate_fused_results(bm25_fused, test_queries)
    
    # Collect all per-query results
    all_per_query_results = {
        "tfidf_level1": tfidf_level1_per_query,
        "tfidf_level2": tfidf_level2_per_query,
        "tfidf_level3": tfidf_level3_per_query,
        "tfidf_fused": tfidf_fused_per_query,
        "bm25_level1": bm25_level1_per_query,
        "bm25_level2": bm25_level2_per_query,
        "bm25_level3": bm25_level3_per_query,
        "bm25_fused": bm25_fused_per_query
    }
    
    return ((tfidf_level1_metrics, tfidf_level2_metrics, tfidf_level3_metrics, tfidf_fused_metrics,
             bm25_level1_metrics, bm25_level2_metrics, bm25_level3_metrics, bm25_fused_metrics),
            all_per_query_results)

def run_traditional_evaluation():
    """Run traditional IR evaluation using TF-IDF and BM25"""
    print("Starting traditional IR evaluation")
    
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        print("Warning: Could not download NLTK data")
    
    # Load models
    models = load_traditional_models()
    
    # Load corpus and test data
    corpus_ids = load_corpus_ids()
    test_queries = load_test_queries()
    query_texts = [query["query_text"] for query in test_queries]
    
    # Build multilevel corpus
    level1_texts, level2_texts, level3_texts = build_multilevel_corpus(corpus_ids)
    
    # Build traditional indexes
    indexes = build_traditional_indexes([level1_texts, level2_texts, level3_texts], corpus_ids)
    
    # Evaluate traditional retrieval
    aggregated_results, all_per_query_results = evaluate_multilevel_retrieval_traditional(query_texts, indexes, test_queries, corpus_ids)
    
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
    print("Starting traditional IR multilevel embedding evaluation")
    
    results = run_traditional_evaluation()
    
    (tfidf_level1_metrics, tfidf_level2_metrics, tfidf_level3_metrics, tfidf_fused_metrics,
     bm25_level1_metrics, bm25_level2_metrics, bm25_level3_metrics, bm25_fused_metrics) = results
    
    # Print TF-IDF results
    print_results("TF-IDF Level 1 (title+abstract)", tfidf_level1_metrics)
    print_results("TF-IDF Level 2 (title+abstract+intro)", tfidf_level2_metrics)
    print_results("TF-IDF Level 3 (title+abstract+intro+conclusion)", tfidf_level3_metrics)
    print_results(f"TF-IDF Multi-level Fusion ({FUSION_STRATEGY})", tfidf_fused_metrics)
    
    # Print BM25 results
    print_results("BM25 Level 1 (title+abstract)", bm25_level1_metrics)
    print_results("BM25 Level 2 (title+abstract+intro)", bm25_level2_metrics)
    print_results("BM25 Level 3 (title+abstract+intro+conclusion)", bm25_level3_metrics)
    print_results(f"BM25 Multi-level Fusion ({FUSION_STRATEGY})", bm25_fused_metrics)
    
    print("Traditional IR multilevel evaluation completed")

if __name__ == "__main__":
    main()
