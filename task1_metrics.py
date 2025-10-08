#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 Metrics Evaluation Module
Implements recall@k and ndcg@k evaluation for reference generation at multiple top-k values
"""

import numpy as np
from typing import List, Dict, Any
from difflib import SequenceMatcher

# Configuration constants - support multiple top-k values
TOP_K_EVALUATE = [10, 20, 40]  # Multiple k values for evaluation
TOP_K_GENERATE = 20  # Default k value for backward compatibility
EPSILON = 1e-8
SIMILARITY_THRESHOLD = 0.90  # 90% similarity threshold

def calculate_title_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity between two titles using SequenceMatcher
    
    Args:
        title1: First title
        title2: Second title
    
    Returns:
        Similarity score between 0 and 1
    """
    return SequenceMatcher(None, title1.lower().strip(), title2.lower().strip()).ratio()

def find_similar_titles(pred_title: str, label_titles: List[str], threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """
    Check if predicted title has a similar match in label titles
    
    Args:
        pred_title: Predicted title
        label_titles: List of ground truth titles
        threshold: Similarity threshold (default: 0.90)
    
    Returns:
        True if similar title found, False otherwise
    """
    for label_title in label_titles:
        similarity = calculate_title_similarity(pred_title, label_title)
        if similarity >= threshold:
            return True
    return False

def calculate_metrics(top_k: int, label_titles: List[str], prediction_titles: List[str]) -> Dict[str, Any]:
    """
    Calculate recall@k and ndcg@k metrics with title similarity
    
    Args:
        top_k: Number of top predictions to consider
        label_titles: Ground truth reference titles
        prediction_titles: Generated reference titles
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Calculating evaluation metrics for k={top_k}...")
    
    # Normalize titles for comparison
    label_titles_normalized = [title.lower().strip() for title in label_titles]
    prediction_titles_normalized = [title.lower().strip() for title in prediction_titles[:top_k]]
    
    # Calculate hits (exact matches + similar matches)
    hits = 0
    hit_positions = []
    
    for i, pred_title in enumerate(prediction_titles_normalized):
        # First check for exact match
        if pred_title in label_titles_normalized:
            hits += 1
            hit_positions.append(i)
        else:
            # Check for similar titles (similarity > 90%)
            if find_similar_titles(pred_title, label_titles_normalized):
                hits += 1
                hit_positions.append(i)
    
    # Calculate recall@k
    recall_at_k = hits / len(label_titles_normalized) if label_titles_normalized else 0.0
    
    # Calculate NDCG@k
    ndcg_at_k = calculate_ndcg_at_k(hit_positions, top_k, len(label_titles_normalized))
    
    print(f"Metrics for k={top_k} - Hits: {hits}, Recall@{top_k}: {recall_at_k:.4f}, NDCG@{top_k}: {ndcg_at_k:.4f}")
    
    return {
        "recall_at_k": recall_at_k,
        "ndcg_at_k": ndcg_at_k,
        "hits": hits,
        "generated_count": len(prediction_titles_normalized),
        "reference_count": len(label_titles_normalized)
    }

def calculate_ndcg_at_k(hit_positions: List[int], k: int, num_references: int) -> float:
    """
    Calculate NDCG@k for reference generation
    
    Args:
        hit_positions: Positions of hits in prediction list
        k: Number of top predictions
        num_references: Number of ground truth references
    
    Returns:
        NDCG@k score
    """
    if num_references == 0:
        return 0.0
    
    # Create relevance scores (1 for hits, 0 for misses)
    relevance_scores = np.zeros(k)
    for pos in hit_positions:
        if pos < k:
            relevance_scores[pos] = 1.0
    
    # Calculate DCG
    dcg = np.sum(relevance_scores / np.log2(np.arange(2, k + 2)))
    
    # Calculate IDCG (ideal case: all hits at the beginning)
    ideal_relevance = np.ones(min(k, num_references))
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
    # Calculate NDCG
    ndcg = dcg / (idcg + EPSILON)
    
    return float(ndcg)

def evaluate_generation_multi_topk(generated_titles: List[str], reference_titles: List[str]) -> Dict[str, Any]:
    """
    Evaluate generation quality using multiple top-k values
    
    Args:
        generated_titles: Generated reference titles
        reference_titles: Ground truth reference titles
    
    Returns:
        Dictionary containing evaluation metrics for multiple k values
    """
    print("Evaluating generation quality with multiple top-k values...")
    
    if not generated_titles:
        print("No generated titles to evaluate")
        # Return metrics for all k values with 0 scores
        metrics = {
            "recall_at_k": {},
            "ndcg_at_k": {},
            "hits": {},
            "generated_count": len(generated_titles),
            "reference_count": len(reference_titles)
        }
        
        for k in TOP_K_EVALUATE:
            metrics["recall_at_k"][f"k={k}"] = 0.0
            metrics["ndcg_at_k"][f"k={k}"] = 0.0
            metrics["hits"][f"k={k}"] = 0
        
        return metrics
    
    # Calculate metrics for each k value
    metrics = {
        "recall_at_k": {},
        "ndcg_at_k": {},
        "hits": {},
        "generated_count": len(generated_titles),
        "reference_count": len(reference_titles)
    }
    
    for k in TOP_K_EVALUATE:
        k_metrics = calculate_metrics(k, reference_titles, generated_titles)
        metrics["recall_at_k"][f"k={k}"] = k_metrics["recall_at_k"]
        metrics["ndcg_at_k"][f"k={k}"] = k_metrics["ndcg_at_k"]
        metrics["hits"][f"k={k}"] = k_metrics["hits"]
    
    print("Multi top-k evaluation completed")
    return metrics

def evaluate_generation(generated_titles: List[str], reference_titles: List[str]) -> Dict[str, Any]:
    """
    Evaluate generation quality using single top-k value (backward compatibility)
    
    Args:
        generated_titles: Generated reference titles
        reference_titles: Ground truth reference titles
    
    Returns:
        Dictionary containing evaluation metrics for default k value
    """
    print("Evaluating generation quality with single top-k value...")
    
    if not generated_titles:
        print("No generated titles to evaluate")
        return {
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "hits": 0,
            "generated_count": 0,
            "reference_count": len(reference_titles)
        }
    
    # Use default TOP_K_GENERATE for backward compatibility
    metrics = calculate_metrics(TOP_K_GENERATE, reference_titles, generated_titles)
    print("Single top-k evaluation completed")
    
    return metrics
