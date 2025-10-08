#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Citation Prediction Module for Task 2
Predicts citations for [ref] markers in paper sections
"""

import os
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from fluxllm.clients import FluxOpenAI
from task2_metrics import evaluate_citation_prediction

# Configuration constants
API_KEY = ""
BASE_URL = ""
MAX_RETRIES = 10
MAX_PARALLEL_SIZE = 64
TEMPERATURE = 0.6
MAX_TOKENS = 16000
TIMEOUT = 30.0
TOP_K_GENERATE = 40  # Number of citations to generate for each [ref]
TOP_K_EVALUATE = [10, 20, 40]  # List of top-k values for evaluation
TOP_R = 20  # Number of retrieved papers from retriever
BATCH_SIZE = 32
MAX_REF_MARKERS = 1

# Configuration
TEST_DATA_DIR = "./task2_dataset"

# System prompt for citation prediction
SYSTEM_PROMPT = """You are a professional academic citation prediction expert. Your task is to predict the most likely citations for [ref] markers in academic paper sections based on the paper content and retrieved relevant papers.

RETRIEVER INTERFACE:
- Input: paper title + abstract + section text with [ref] markers
- Output: top {top_r} retrieved papers with title and abstract
- Format: List of dictionaries with 'title' and 'abstract' fields

CRITICAL FORMAT REQUIREMENT: You must respond with ONLY valid JSON. Do not include any explanatory text, markdown formatting, or other content outside the JSON structure.

TASK DESCRIPTION:
- Input: Paper section text with [ref] markers + retrieved relevant papers
- Output: top {top_k} predicted reference titles for each [ref] marker with reasoning
- Format: JSON object with "citations" array containing objects with "ref_index", "titles", and "reasoning" fields

REQUIREMENTS:
1. Analyze the context around each [ref] marker to understand what reference is needed
2. Consider the retrieved papers as potential references (when available)
3. Consider the surrounding text, research area, and topic
4. Generate realistic and relevant reference titles that would fit the context
5. Provide detailed reasoning for each prediction to prevent hallucination
6. Output format: JSON object with "citations" array
7. Generate exactly {top_k} most relevant reference titles for each [ref]
8. Ensure titles are realistic and appropriate for the research area
9. Include reasoning that explains why each title is relevant to the specific [ref] context
10. Use proper JSON escaping for special characters (escape quotes with \\")
11. Ensure all strings are properly quoted and escaped

EXAMPLE:
Input:
"Federated Learning aims to train models in massively distributed networks [ref]38 at a large scale [ref]5, over multiple sources of heterogeneous data [ref]30."

Retrieved papers (top {top_r}):
1. Title: Communication-efficient learning of deep networks from decentralized data
   Abstract: We report on a series of experiments with federated learning systems for decentralized data training.

2. Title: Federated Learning: Challenges, Methods, and Future Directions
   Abstract: A comprehensive survey of federated learning approaches and challenges in distributed settings.

Output:
{{
  "citations": [
    {{
      "ref_index": 38,
      "titles": [
        "Communication-efficient learning of deep networks from decentralized data",
        "Federated Learning: Challenges, Methods, and Future Directions",
        "Federated optimization in heterogeneous networks",
        "Towards federated learning at scale: System design",
        "A performance evaluation of federated learning algorithms",
        "Distributed machine learning: A survey",
        "Communication protocols for distributed learning",
        "Scalable federated learning systems",
        "Efficient distributed training algorithms",
        "Large-scale distributed learning frameworks"
      ],
      "reasoning": "This [ref]38 appears in the context of 'massively distributed networks' for federated learning, so it likely refers to foundational papers on federated learning systems, distributed training, and communication-efficient methods in distributed networks."
    }},
    {{
      "ref_index": 5,
      "titles": [
        "Bagging predictors",
        "Ensemble methods in machine learning",
        "Random forests for classification",
        "Bootstrap methods and their application",
        "Combining multiple classifiers",
        "Large-scale machine learning systems",
        "Distributed ensemble methods",
        "Scalable classification algorithms",
        "Massive data processing techniques",
        "High-performance machine learning frameworks"
      ],
      "reasoning": "This [ref]5 appears in the context of 'at a large scale', suggesting it refers to papers on scaling machine learning methods, ensemble techniques, or methods that can handle large-scale data and distributed settings."
    }},
    {{
      "ref_index": 30,
      "titles": [
        "Learning fair representations",
        "Fair machine learning: A survey",
        "Bias in machine learning systems",
        "Addressing fairness in AI systems",
        "Fair representation learning",
        "Heterogeneous data handling methods",
        "Multi-source learning algorithms",
        "Cross-domain representation learning",
        "Fair learning from diverse data",
        "Bias mitigation in heterogeneous datasets"
      ],
      "reasoning": "This [ref]30 appears in the context of 'heterogeneous data' and 'fair representation learning', indicating it likely refers to papers on heterogenous data handling, bias mitigation, and learning representations that are fair across different data distributions."
    }}
  ]
}}

CRITICAL: Return ONLY the JSON object. Do not include any text before or after the JSON. Do not use markdown formatting. Ensure all quotes and special characters are properly escaped.

IMPORTANT NOTES ON REASONING:
- The reasoning field is crucial for preventing hallucination
- Each reasoning should be specific to the context around the [ref] marker
- Explain the connection between the predicted titles and the research topic
- This helps validate that predictions are grounded in the actual content
- Reasoning should be concise but informative (2-3 sentences)
- Consider retrieved papers when available to improve prediction accuracy"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_citation_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RAG Citation Prediction Module for Task 2')
    parser.add_argument(
        '--model', 
        type=str, 
        default='gpt-4.1-2025-04-14',
        help='Model name to use for prediction (default: gpt-4.1-2025-04-14)'
    )
    parser.add_argument(
        '--test-data-dir',
        type=str,
        default=TEST_DATA_DIR,
        help=f'Directory containing test data (default: {TEST_DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    return parser.parse_args()

def setup_output_directory(model_name: str, output_dir: str = 'outputs') -> str:
    """Setup output directory structure and return output file path"""
    outputs_path = Path(output_dir)
    outputs_path.mkdir(exist_ok=True)
    
    model_dir = outputs_path / model_name.replace('/', '_').replace(':', '_')
    model_dir.mkdir(exist_ok=True)
    
    output_file = model_dir / f"TASK2_batch_evaluation_results.json"
    
    print(f"Output directory: {model_dir}")
    print(f"Output file: {output_file}")
    
    return str(output_file)

def create_flux_client():
    """Create FluxLLM client with configuration"""
    print("Creating FluxLLM client...")
    client = FluxOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=MAX_RETRIES,
        max_parallel_size=MAX_PARALLEL_SIZE
    )
    print("SUCCESS: FluxLLM client created")
    return client

def format_retrieved_papers(retrieved_papers: List[Dict[str, Any]]) -> str:
    """Format retrieved papers for prompt with title and abstract"""
    if not retrieved_papers:
        return "No retrieved papers available."
    
    formatted = f"Retrieved relevant papers (top {TOP_R}):\n"
    for i, paper in enumerate(retrieved_papers, 1):
        title = paper.get('title', 'Unknown Title')
        abstract = paper.get('abstract', 'No abstract available')
        
        formatted += f"{i}. Title: {title}\n"
        formatted += f"   Abstract: {abstract}\n\n"
    
    return formatted

def extract_ref_markers(section_text: str) -> List[Tuple[int, int]]:
    """Extract [ref] markers and their positions from section text"""
    ref_pattern = r'\[ref\](\d+)'
    refs = []
    
    for match in re.finditer(ref_pattern, section_text):
        ref_index = int(match.group(1))
        start_pos = match.start()
        refs.append((ref_index, start_pos))
    
    return refs

def create_context_window(section_text: str, ref_pos: int, window_size: int = 200) -> str:
    """Create context window around a [ref] marker"""
    start = max(0, ref_pos - window_size)
    end = min(len(section_text), ref_pos + window_size)
    
    context = section_text[start:end]
    
    if start > 0:
        first_space = context.find(' ')
        if first_space != -1:
            context = context[first_space + 1:]
    
    if end < len(section_text):
        last_space = context.rfind(' ')
        if last_space != -1:
            context = context[:last_space]
    
    return context.strip()

def mock_retriever(title: str, abstract: str, section_text: str) -> List[Dict[str, Any]]:
    """Mock retriever that returns empty results with title and abstract format"""
    print("Mock retriever called - returning empty results")
    print(f"Input: title={title[:50]}..., abstract={abstract[:100]}..., section_length={len(section_text)}")
    print(f"TOP_R parameter: {TOP_R}")
    return []

def create_request_data(section_text: str, ref_markers: List[Tuple[int, int]], retrieved_papers: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Create request data for citation prediction"""
    contexts = []
    for ref_index, ref_pos in ref_markers:
        context = create_context_window(section_text, ref_pos)
        contexts.append(f"[ref]{ref_index}: {context}")
    
    contexts_text = "\n\n".join(contexts)
    retrieved_text = format_retrieved_papers(retrieved_papers)
    
    user_prompt = f"""Paper Section Text:
{section_text}

Reference markers found: {[ref[0] for ref in ref_markers]}

Context windows for each [ref]:
{contexts_text}

{retrieved_text}

Generate exactly {TOP_K_GENERATE} most likely reference titles for each [ref] marker. Focus on papers that are directly related to the research topic, provide foundational background, present similar methodologies, or address related problems.

CRITICAL REQUIREMENTS:
1. For each [ref] marker, provide detailed reasoning explaining why the predicted titles are relevant
2. The reasoning should analyze the specific context around each [ref] marker
3. Explain how each title connects to the research topic, methodology, or problem being discussed
4. This reasoning helps prevent hallucination by grounding predictions in the actual context
5. Consider retrieved papers when available to improve prediction accuracy

CRITICAL FORMAT REQUIREMENT: You must respond with ONLY valid JSON. Do not include any explanatory text, markdown formatting, or other content outside the JSON structure.

IMPORTANT: You must respond with ONLY valid JSON in this exact format:
{{
    "citations": [
        {{
            "ref_index": 38,
            "titles": [
                "Reference Title 1",
                "Reference Title 2",
                "Reference Title 3",
                "Reference Title 4",
                "Reference Title {TOP_K_GENERATE}"
            ],
            "reasoning": "Detailed explanation of why these titles are relevant to [ref]38 based on the surrounding context, research area, and specific topic being discussed."
        }}
    ]
}}

CRITICAL: Return ONLY the JSON object. Do not include any text before or after the JSON. Do not use markdown formatting. Ensure all quotes and special characters are properly escaped. Do not include authors, venues, or years - only the paper titles. Generate exactly {TOP_K_GENERATE} titles for each [ref] marker with reasoning."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(top_k=TOP_K_GENERATE, top_r=TOP_R)},
        {"role": "user", "content": user_prompt}
    ]
    
    request_data = {
        "model": model_name,
        "messages": messages,
        "timeout": TIMEOUT
    }
    
    # 为 o3-2025-04-16 模型使用特殊参数配置
    if model_name == "o3-2025-04-16":
        # o3 模型只支持默认 temperature (1)，不支持自定义值
        pass  # 不添加 temperature 参数，使用默认值
    else:
        # 其他模型使用自定义参数
        request_data["temperature"] = TEMPERATURE
        request_data["max_tokens"] = MAX_TOKENS
    
    return request_data

def parse_citation_response(response_content: str, ref_markers: List[Tuple[int, int]]) -> List[List[str]]:
    """Parse response content to extract citation predictions"""
    if not response_content or not isinstance(response_content, str):
        return []
    
    content = response_content.strip()
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return []
    
    json_content = content[start_idx:end_idx + 1]
    
    try:
        parsed = json.loads(json_content)
    except json.JSONDecodeError:
        return []
    
    if not isinstance(parsed, dict) or 'citations' not in parsed:
        return []
    
    citations = parsed['citations']
    if not isinstance(citations, list):
        return []
    
    predictions = []
    ref_indices = [ref[0] for ref in ref_markers]
    
    for ref_index in ref_indices:
        citation = None
        for cit in citations:
            if isinstance(cit, dict) and cit.get('ref_index') == ref_index:
                citation = cit
                break
        
        if citation and 'titles' in citation and isinstance(citation['titles'], list):
            titles = citation['titles'][:TOP_K_GENERATE]
            predictions.append(titles)
        else:
            predictions.append([])
    
    return predictions

def generate_citations_batch(client, papers_data: List[Dict[str, Any]], model_name: str) -> List[List[List[str]]]:
    """Generate citations for multiple papers in batch using high concurrency"""
    print(f"Starting batch citation generation for {len(papers_data)} papers...")
    
    all_predictions = []
    total_papers = len(papers_data)
    
    for batch_start in range(0, total_papers, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_papers)
        batch_papers = papers_data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}: papers {batch_start+1}-{batch_end}")
        
        batch_requests = []
        batch_ref_markers = []
        
        for paper_data in batch_papers:
            sections = paper_data.get('sections', {})
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            all_ref_markers = []
            
            for section_name, section_text in sections.items():
                refs = extract_ref_markers(section_text)
                all_ref_markers.extend(refs)
            
            unique_refs = list(dict.fromkeys(all_ref_markers))
            actual_ref_count = len(unique_refs)
            refs_to_process = min(actual_ref_count, MAX_REF_MARKERS)
            limited_refs = unique_refs[:refs_to_process]
            
            if limited_refs:
                combined_text = " ".join(sections.values())
                retrieved_papers = mock_retriever(title, abstract, combined_text)
                request_data = create_request_data(combined_text, limited_refs, retrieved_papers, model_name)
                batch_requests.append(request_data)
                batch_ref_markers.append(limited_refs)
            else:
                batch_requests.append(None)
                batch_ref_markers.append([])
        
        print(f"Created {len(batch_requests)} requests for current batch")
        
        print("Executing batch request to FluxLLM...")
        responses = client.request([req for req in batch_requests if req is not None])
        print(f"Received {len(responses)} responses from FluxLLM")
        
        response_index = 0
        for i, (paper_data, ref_markers) in enumerate(zip(batch_papers, batch_ref_markers)):
            paper_index = batch_start + i
            print(f"Processing response {paper_index+1}/{total_papers}")
            
            if not ref_markers or batch_requests[i] is None:
                all_predictions.append([])
                continue
            
            if response_index >= len(responses):
                all_predictions.append([])
                continue
            
            response = responses[response_index]
            response_index += 1
            
            if response is None:
                all_predictions.append([])
                continue
            
            if isinstance(response, dict):
                response_content = response['choices'][0]['message']['content']
            else:
                response_content = response.choices[0].message.content
            
            if not response_content:
                print(f"WARNING: Empty response content for paper {paper_index+1}, skipping...")
                all_predictions.append([])
                continue
            
            response_content = response_content.strip()
            
            predictions = parse_citation_response(response_content, ref_markers)
            all_predictions.append(predictions)
        
        print(f"Completed batch {batch_start//BATCH_SIZE + 1}")
    
    print(f"SUCCESS: Batch generation completed for {len(all_predictions)} papers")
    return all_predictions

def load_test_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load test data from directory"""
    print(f"Loading test data from {data_dir}...")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory {data_dir} does not exist")
        return []
    
    test_data = []
    json_files = list(data_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
            test_data.append(paper_data)
    
    print(f"SUCCESS: Loaded {len(test_data)} papers")
    return test_data

def extract_labels_from_papers(papers_data: List[Dict[str, Any]]) -> List[List[List[str]]]:
    """Extract ground truth labels from papers data"""
    print("Extracting ground truth labels from papers...")
    
    all_labels = []
    
    for paper_idx, paper_data in enumerate(papers_data):
        sections = paper_data.get('sections', {})
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        reference_labels = paper_data.get('reference_labels', [])
        index_to_title = {ref['index']: ref['title'] for ref in reference_labels}
        
        all_ref_markers = []
        for section_name, section_text in sections.items():
            refs = extract_ref_markers(section_text)
            all_ref_markers.extend(refs)
        
        unique_refs = list(dict.fromkeys(all_ref_markers))
        actual_ref_count = len(unique_refs)
        refs_to_process = min(actual_ref_count, MAX_REF_MARKERS)
        limited_refs = unique_refs[:refs_to_process]
        
        paper_labels = []
        for ref_index, _ in limited_refs:
            if ref_index in index_to_title:
                label_title = index_to_title[ref_index]
                paper_labels.append([label_title])
            else:
                paper_labels.append([])
        
        all_labels.append(paper_labels)
    
    print(f"SUCCESS: Extracted labels for {len(all_labels)} papers")
    return all_labels

def process_batch_papers(client, papers_data: List[Dict[str, Any]], model_name: str) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
    """Process multiple papers in batch using high concurrency"""
    print(f"Processing batch of {len(papers_data)} papers with high concurrency...")
    
    all_predictions = generate_citations_batch(client, papers_data, model_name)
    all_labels = extract_labels_from_papers(papers_data)
    
    print("SUCCESS: Batch processing completed")
    return all_predictions, all_labels

def save_results(predictions: List[List[List[str]]], labels: List[List[List[str]]], metrics: Dict[str, Any], output_file: str):
    """Save batch results and metrics to file"""
    print(f"Saving results to {output_file}...")
    
    output_data = {
        'metrics': metrics,
        'predictions': predictions,
        'labels': labels
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("SUCCESS: Results saved to file")

if __name__ == "__main__":
    print("RAG Citation Prediction Module - Starting...")
    
    args = parse_arguments()
    model_name = args.model
    test_data_dir = args.test_data_dir
    output_dir = args.output_dir
    
    print(f"Model: {model_name}")
    print(f"Test data directory: {test_data_dir}")
    print(f"Output directory: {output_dir}")
    
    output_file = setup_output_directory(model_name, output_dir)
    client = create_flux_client()
    test_data = load_test_data(test_data_dir)
    
    if not test_data:
        print("ERROR: No test data loaded")
        exit(1)
    
    print(f"Starting batch evaluation of {len(test_data)} papers...")
    predictions, labels = process_batch_papers(client, test_data, model_name)
    
    print("Evaluating citation prediction results...")
    metrics = evaluate_citation_prediction(predictions, labels)
    
    save_results(predictions, labels, metrics, output_file)
    
    print("=" * 50)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Total papers processed: {len(predictions)}")
    print(f"Generated citations per [ref]: {TOP_K_GENERATE}")
    print(f"Evaluation top-k values: {TOP_K_EVALUATE}")
    
    # Display metrics for each top-k value
    pacc_at_k = metrics.get('pacc_at_k', {})
    if isinstance(pacc_at_k, dict):
        for top_k in TOP_K_EVALUATE:
            if top_k in pacc_at_k:
                print(f"Average pacc@{top_k}: {pacc_at_k[top_k]:.4f}")
    else:
        print(f"Average pacc@{TOP_K_GENERATE}: {pacc_at_k:.4f}")
    
    print(f"Total [ref] markers: {metrics.get('total_refs', 0)}")
    print(f"Correct predictions: {metrics.get('correct_predictions', 0)}")
    print(f"Results saved to: {output_file}")
    print("=" * 50)
    
    print("SUCCESS: RAG Citation Prediction Module - Completed")
