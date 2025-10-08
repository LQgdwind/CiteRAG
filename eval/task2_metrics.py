#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 Metrics Evaluation Module
Implements pacc@k evaluation for citation prediction task at multiple top-k values
"""

import numpy as np
from typing import List, Dict, Any

# Configuration constants - define here to avoid circular import
TOP_K_EVALUATE = [10, 20, 40]
EPSILON = 1e-8

def calculate_pacc_at_k(prediction_ranks: List[int], top_k: int) -> float:
    """Calculate pacc@k for citation prediction"""
    if not prediction_ranks:
        return 0.0
    
    total_score = 0.0
    for rank in prediction_ranks:
        if rank <= top_k:
            score = 1.0 - (rank - 1) / top_k
            total_score += score
    
    return total_score / len(prediction_ranks)

def find_correct_predictions(predicted_titles: List[str], reference_titles: List[str]) -> List[int]:
    """Find ranks of correct predictions in the predicted list"""
    correct_ranks = []
    reference_titles_normalized = [title.lower().strip() for title in reference_titles]
    
    for i, pred_title in enumerate(predicted_titles):
        pred_title_normalized = pred_title.lower().strip()
        if pred_title_normalized in reference_titles_normalized:
            correct_ranks.append(i + 1)
    
    return correct_ranks

def calculate_article_metrics(predictions: List[List[str]], labels: List[List[str]], top_k_values: List[int] = TOP_K_EVALUATE) -> Dict[str, Any]:
    """Calculate metrics for a single article at multiple top-k values"""
    if len(predictions) != len(labels):
        return {
            "pacc_at_k": {k: 0.0 for k in top_k_values},
            "total_refs": 0,
            "correct_predictions": 0,
            "prediction_count": len(predictions)
        }
    
    total_refs = len(predictions)
    correct_predictions = 0
    all_ranks = []
    
    for i, (pred_list, label_list) in enumerate(zip(predictions, labels)):
        if not pred_list or not label_list:
            continue
        
        correct_ranks = find_correct_predictions(pred_list, label_list)
        
        if correct_ranks:
            correct_predictions += 1
            all_ranks.extend(correct_ranks)
    
    # Calculate pacc@k for each top-k value
    pacc_at_k = {}
    for top_k in top_k_values:
        if all_ranks:
            pacc_at_k[top_k] = calculate_pacc_at_k(all_ranks, top_k)
        else:
            pacc_at_k[top_k] = 0.0
    
    return {
        "pacc_at_k": pacc_at_k,
        "total_refs": total_refs,
        "correct_predictions": correct_predictions,
        "prediction_count": len(predictions)
    }

def evaluate_citation_prediction(predictions: List[List[str]], labels: List[List[str]]) -> Dict[str, Any]:
    """Evaluate citation prediction quality using pacc@k at multiple top-k values"""
    if not predictions or not labels or len(predictions) != len(labels):
        return {
            "pacc_at_k": {k: 0.0 for k in TOP_K_EVALUATE},
            "total_articles": 0,
            "total_refs": 0,
            "correct_predictions": 0
        }
    
    total_articles = len(predictions)
    total_pacc = {k: 0.0 for k in TOP_K_EVALUATE}
    total_refs = 0
    total_correct = 0
    
    print(f"Processing {total_articles} articles...")
    
    for article_predictions, article_labels in zip(predictions, labels):
        article_metrics = calculate_article_metrics(article_predictions, article_labels)
        
        for top_k in TOP_K_EVALUATE:
            total_pacc[top_k] += article_metrics["pacc_at_k"][top_k]
        
        total_refs += article_metrics["total_refs"]
        total_correct += article_metrics["correct_predictions"]
    
    # Calculate average pacc@k for each top-k value
    avg_pacc_at_k = {}
    for top_k in TOP_K_EVALUATE:
        avg_pacc_at_k[top_k] = total_pacc[top_k] / total_articles if total_articles > 0 else 0.0
    
    final_metrics = {
        "pacc_at_k": avg_pacc_at_k,
        "total_articles": total_articles,
        "total_refs": total_refs,
        "correct_predictions": total_correct
    }
    
    print(f"SUCCESS: Evaluation completed")
    for top_k in TOP_K_EVALUATE:
        print(f"Final metrics - avg pacc@{top_k}: {avg_pacc_at_k[top_k]:.4f}")
    print(f"Total articles: {total_articles}, total refs: {total_refs}")
    
    return final_metrics
