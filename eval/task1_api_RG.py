#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Generation Module for Task 1
Predicts reference list based on paper title, abstract and retrieval results
"""

import os
import json
import logging
import argparse
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from fluxllm.clients import FluxOpenAI
from task1_metrics import evaluate_generation, evaluate_generation_multi_topk

# Configuration constants
API_KEY = ""
BASE_URL = ""
MAX_RETRIES = 10
MAX_PARALLEL_SIZE = 64
TEMPERATURE = 0.6
MAX_TOKENS = 16000
TIMEOUT = 120.0
TOP_K_GENERATE = 40  # Number of references to generate (should be >= max evaluation k)
BATCH_SIZE = 32  # Batch size for high concurrency processing
TOP_K_EVALUATE = [10, 20, 40]

# Retriever API configuration
RETRIEVER_TIMEOUT = 30.0
RETRIEVER_BATCH_SIZE = 10 # Parallel batch size for retriever API calls
RETRIEVER_MAX_RETRIES = 5  # Maximum number of retry attempts for retriever API
RETRIEVER_RETRY_DELAY = 1.0  # Initial delay between retries (in seconds) 

# Configuration
TEST_DATA_DIR = "./task1"

# System prompt for reference generation
SYSTEM_PROMPT = """You are a professional academic reference prediction expert. Your task is to predict the most likely references for a given academic paper based on its title, abstract, and retrieved relevant papers.

RETRIEVER INTERFACE:
- Input: paper title + abstract
- Output: top {top_r} retrieved papers with title and abstract
- Format: List of dictionaries with 'title' and 'abstract' fields

REQUIREMENTS:
1. Analyze the paper's title and abstract to understand its research area and focus
2. Consider the retrieved papers as potential references (when available)
3. Generate a list of predicted reference titles that would be most relevant and likely to be cited
4. Provide reasoning for each prediction to prevent hallucination
5. Output format: JSON object with "titles" array and "reasoning" string
6. Generate exactly {top_k} most relevant reference titles (this should be at least 40 to cover all evaluation metrics)
7. Ensure titles are realistic and appropriate for the research area
8. Consider citation patterns in the field

EXAMPLE:
Input:
Title: "Question Type Classification Methods Comparison"
Abstract: "The paper presents a comparative study of state-of-the-art approaches for question classification task: Logistic Regression, Convolutional Neural Networks (CNN), Long Short-Term Memory Network (LSTM) and Quasi-Recurrent Neural Networks (QRNN). All models use pre-trained GLoVe word embeddings and trained on human-labeled data. The best accuracy is achieved using CNN model with five convolutional layers and various kernel sizes stacked in parallel, followed by one fully connected layer. The model reached 90.7% accuracy on TREC 10 test set."

Retrieved papers (top {top_r}):
1. Title: Convolutional Neural Networks for Sentence Classification
   Abstract: We report on a series of experiments with convolutional neural networks trained on top of pre-trained word vectors for sentence-level classification tasks.

2. Title: Glove: Global vectors for word representation
   Abstract: We present a new global log-bilinear regression model that combines the advantages of the two major model families in the literature.

Output:
{{
  "titles": [
    "Convolutional Neural Networks for Sentence Classification",
    "Glove: Global vectors for word representation",
    "Quasi-Recurrent Neural Networks",
    "A Comparative Study of Neural Network Models for Sentence Classification",
    "Deep lstm based feature mapping for query classification"
  ],
  "reasoning": "Based on the paper's focus on question classification using neural networks, I predict these references because: 1) CNN paper is directly relevant to the convolutional approach used; 2) GloVe embeddings are mentioned as pre-trained features; 3) QRNN is one of the compared methods; 4) Comparative study papers are common in this field; 5) LSTM-based query classification is closely related to question classification. Additional references cover related neural network architectures, NLP applications, deep learning methodologies, optimization techniques, and advanced neural network methods that would be relevant for this research area."
}}

CRITICAL: Each title must appear exactly once. Do not repeat any title.

Return JSON object only."""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RAG Generation Module for Task 1')
    parser.add_argument(
        '--model', 
        type=str, 
        default='gpt-4.1-2025-04-14',
        help='Model name to use for generation (default: gpt-4.1-2025-04-14)'
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
    parser.add_argument(
        '--top-r',
        type=int,
        default=5,
        help='retrieved papers from retriever (default: 5)'
    )
    parser.add_argument(
        '--url',
        type=str,
        default="",
        help='Retriever API URL'
    )

    return parser.parse_args()

def setup_output_directory(model_name: str, top_r: int, output_dir: str = 'outputs') -> str:
    """Setup output directory structure and return output file path"""
    # Create outputs directory if it doesn't exist
    outputs_path = Path(output_dir)
    outputs_path.mkdir(exist_ok=True)
    
    # Create model-specific subdirectory
    model_dir = outputs_path / model_name.replace('/', '_').replace(':', '_')
    model_dir.mkdir(exist_ok=True)
    
    # Generate output file path
    output_file = model_dir / f"r-{top_r}-TASK1_batch_evaluation_results.json"
    
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

def create_request_data(title: str, abstract: str, retrieved_papers: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Create request data for a single paper"""
    retrieved_text = format_retrieved_papers(retrieved_papers)
    
    user_prompt = f"""Paper Title: {title}

Paper Abstract: {abstract}

{retrieved_text}

Generate exactly {TOP_K_GENERATE} most likely reference titles for this paper. Focus on papers that are directly related to the research topic, provide foundational background, present similar methodologies, or address related problems.

IMPORTANT: You must respond with ONLY valid JSON in this exact format:
{{
    "titles": [
        "Reference Title 1",
        "Reference Title 2",
        "Reference Title 3",
        "Reference Title 4"
        "Reference Title {TOP_K_GENERATE}"
    ],
    "reasoning": "Explain why you chose these specific references based on the paper's title, abstract, and retrieved papers. This helps prevent hallucination and ensures relevance."
}}

CRITICAL: Each title must appear exactly once. Do not repeat any title.

Do not include any text before or after the JSON. Do not include authors, venues, or years - only the paper titles. The reasoning field is required to explain your prediction logic."""

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

def call_retriever_api_batch(titles_abstracts: List[Tuple[str, str]], retriever_url: str) -> List[List[Dict[str, Any]]]:
    """Call retriever API for multiple papers in parallel using requests"""
    print(f"Calling retriever API for {len(titles_abstracts)} papers in parallel...")
    
    import concurrent.futures
    import threading
    
    def single_retrieval(title_abstract):
        title, abstract = title_abstract
        return call_retriever_api(title, abstract, retriever_url)
    
    # Use ThreadPoolExecutor for parallel API calls with configured batch size
    with concurrent.futures.ThreadPoolExecutor(max_workers=RETRIEVER_BATCH_SIZE) as executor:
        # Submit all retrieval tasks
        future_to_index = {
            executor.submit(single_retrieval, title_abstract): i 
            for i, title_abstract in enumerate(titles_abstracts)
        }
        
        # Collect results in order
        results = [None] * len(titles_abstracts)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    
    print(f"SUCCESS: Completed parallel retrieval for {len(results)} papers")
    return results

def generate_references_batch(client, papers_data: List[Dict[str, Any]], model_name: str) -> List[List[str]]:
    """Generate references for multiple papers in batch using high concurrency"""
    print(f"Starting batch reference generation for {len(papers_data)} papers...")
    
    all_generated_titles = []
    total_papers = len(papers_data)
    
    # Process papers in batches
    for batch_start in range(0, total_papers, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_papers)
        batch_papers = papers_data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}: papers {batch_start+1}-{batch_end}")
        
        # Prepare titles and abstracts for parallel retrieval
        titles_abstracts = []
        for paper_data in batch_papers:
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            titles_abstracts.append((title, abstract))
        
        # Call retriever API in parallel for all papers in batch
        print("Calling retriever API in parallel...")
        all_retrieved_papers = call_retriever_api_batch(titles_abstracts, retriever_url)
        
        # Create batch requests for current batch
        batch_requests = []
        for i, (paper_data, retrieved_papers) in enumerate(zip(batch_papers, all_retrieved_papers)):
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            request_data = create_request_data(title, abstract, retrieved_papers, model_name)
            batch_requests.append(request_data)
        
        print(f"Created {len(batch_requests)} requests for current batch")
        
        # Execute batch request
        print("Executing batch request to FluxLLM...")
        responses = client.request(batch_requests)
        print(f"Received {len(responses)} responses from FluxLLM")
        
        # Process responses for current batch
        for i, response in enumerate(responses):
            paper_index = batch_start + i
            print(f"Processing response {paper_index+1}/{total_papers}")
            
            # Extract response content with robust error handling
            response_content = extract_response_content(response, paper_index+1)
            
            if not response_content:
                print(f"WARNING: Empty response content for paper {paper_index+1}, skipping...")
                all_generated_titles.append([])
                continue
            
            # Clean and parse response
            response_content = response_content.strip()
            generated_titles = parse_response_titles(response_content)
            all_generated_titles.append(generated_titles)
            
            if len(generated_titles) > 0:
                print(f"SUCCESS: Generated {len(generated_titles)} titles for paper {paper_index+1}")
            else:
                print(f"FAILED: No titles generated for paper {paper_index+1}")
        
        print(f"Completed batch {batch_start//BATCH_SIZE + 1}")
    
    print(f"SUCCESS: Batch generation completed for {len(all_generated_titles)} papers")
    return all_generated_titles

def generate_references(client, title: str, abstract: str, retrieved_papers: List[Dict[str, Any]], model_name: str) -> List[str]:
    """Generate predicted reference titles using FluxLLM with retry logic"""
    print("Starting reference generation...")
    
    # Create request data
    request_data = create_request_data(title, abstract, retrieved_papers, model_name)
    
    # Retry logic - 3 attempts
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        print(f"Attempt {attempt + 1}/{MAX_RETRIES}")
        
        # Call the model
        print("Calling FluxLLM for reference generation...")
        
        responses = client.request([request_data])
        response = responses[0]
        
        # Check if response is None or failed
        if response is None:
            print(f"ERROR: API call failed on attempt {attempt + 1}")
            continue
            
        # Extract response content with robust error handling
        print("Extracting response content...")
        response_content = extract_response_content(response, 1)
        
        if not response_content:
            print(f"ERROR: Failed to extract response content on attempt {attempt + 1}")
            continue
        
        # Parse response
        generated_titles = parse_response_titles(response_content)
        if len(generated_titles) > 0:
            print(f"SUCCESS: Generated {len(generated_titles)} reference titles")
            return generated_titles
        else:
            print(f"Attempt {attempt + 1} failed, retrying...")
    
    # All attempts failed, return empty list
    print("ERROR: All 3 attempts failed, returning empty list")
    return []

def clean_response_content(content: str) -> str:
    """Clean response content by removing markdown, code blocks, and extra text"""
    # Remove markdown code blocks
    if '```json' in content:
        start = content.find('```json') + 7
        end = content.find('```', start)
        if end != -1:
            content = content[start:end].strip()
    elif '```' in content:
        start = content.find('```') + 3
        end = content.find('```', start)
        if end != -1:
            content = content[start:end].strip()
    
    # Find the first { and last } to extract JSON
    start_brace = content.find('{')
    end_brace = content.rfind('}')
    
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        content = content[start_brace:end_brace + 1]
    
    # Remove leading/trailing whitespace and newlines
    content = content.strip()
    
    return content

def parse_response_titles(response_content: str) -> List[str]:
    """Parse response content to extract reference titles"""
    print("Parsing response content...")
    
    # Validate input
    if not response_content or not isinstance(response_content, str):
        print("ERROR: Invalid response content type or empty")
        return []
    
    # Clean response content - remove markdown and extra text
    cleaned_content = clean_response_content(response_content)
    
    # Check if content looks like JSON
    if not cleaned_content.strip().startswith('{'):
        print("FAILED: Response does not start with '{'")
        return []
    
    # Attempt to parse JSON with error handling
    try:
        parsed = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"FAILED: Invalid JSON format - {e}")
        return []
    
    # Validate parsed structure
    if not isinstance(parsed, dict):
        print("FAILED: Parsed content is not a dictionary")
        return []
    
    if 'titles' not in parsed:
        print("FAILED: Missing 'titles' field in response")
        return []
    
    titles = parsed['titles']
    reasoning = parsed.get('reasoning', 'No reasoning provided')
    print(f"SUCCESS: Parsed {len(titles)} titles from JSON")
    print(f"Reasoning: {reasoning[:200]}...")
    return titles[:TOP_K_GENERATE]

def extract_response_content(response, paper_index: int) -> Optional[str]:
    """Extract response content from various response formats with robust error handling"""
    print(f"Extracting response content for paper {paper_index}...")
    
    # Handle None response
    if response is None:
        print(f"ERROR: Response for paper {paper_index} is None")
        return None
    
    # Handle dict format (standard OpenAI format)
    if isinstance(response, dict):
        try:
            if 'choices' in response and len(response['choices']) > 0:
                choice = response['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    if content and isinstance(content, str):
                        return content
        except (KeyError, IndexError, AttributeError) as e:
            print(f"ERROR: Failed to extract content from dict response for paper {paper_index}: {e}")
            return None
    
    # Handle object format (FluxLLM object)
    try:
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
                if content and isinstance(content, str):
                    return content
    except (AttributeError, IndexError) as e:
        print(f"ERROR: Failed to extract content from object response for paper {paper_index}: {e}")
        return None
    
    # Handle string format (direct content)
    if isinstance(response, str):
        return response
    
    print(f"ERROR: Unknown response format for paper {paper_index}: {type(response)}")
    return None

def call_retriever_api(title: str, abstract: str, retriever_url: str) -> List[Dict[str, Any]]:
    """Call retriever API to get relevant papers using requests with retry mechanism"""
    print("Calling retriever API...")
    print(f"Input: title={title[:50]}..., abstract={abstract[:100]}...")
    
    # Prepare payload for retrieval API
    payload = {
        "title": title,
        "abstract": abstract,
        "top_k": TOP_R
    }
    
    last_exception = None
    
    for attempt in range(RETRIEVER_MAX_RETRIES):
        try:
            if attempt > 0:
                # Calculate exponential backoff delay
                delay = RETRIEVER_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{RETRIEVER_MAX_RETRIES})")
                time.sleep(delay)
            
            print(f"Sending request to retriever API (attempt {attempt + 1}/{RETRIEVER_MAX_RETRIES})...")
            response = requests.post(retriever_url, json=payload, timeout=RETRIEVER_TIMEOUT)
            
            if response.status_code != 200:
                print(f"API call failed with status code: {response.status_code}")
                # For non-2xx status codes, don't retry immediately - might be a persistent error
                if attempt == RETRIEVER_MAX_RETRIES - 1:
                    print("Max retries reached for non-200 status code")
                    return []
                continue
            
            response_data = response.json()
            print(f"Received response with {len(response_data.get('results', []))} papers")
            
            # Extract papers from results and limit to TOP_R
            results = response_data.get('results', [])
            retrieved_papers = []
            
            for i, paper in enumerate(results[:TOP_R]):
                retrieved_papers.append({
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'paper_id': paper.get('paper_id', ''),
                    'score': paper.get('score', 0.0)
                })
            
            print(f"Successfully retrieved {len(retrieved_papers)} papers")
            return retrieved_papers
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            print(f"Timeout error on attempt {attempt + 1}: {str(e)}")
            if attempt == RETRIEVER_MAX_RETRIES - 1:
                print("Max retries reached due to timeout")
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            print(f"Connection error on attempt {attempt + 1}: {str(e)}")
            if attempt == RETRIEVER_MAX_RETRIES - 1:
                print("Max retries reached due to connection error")
        except requests.exceptions.RequestException as e:
            last_exception = e
            print(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt == RETRIEVER_MAX_RETRIES - 1:
                print("Max retries reached due to request error")
        except Exception as e:
            last_exception = e
            print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == RETRIEVER_MAX_RETRIES - 1:
                print("Max retries reached due to unexpected error")
    
    print(f"Retriever API call failed after {RETRIEVER_MAX_RETRIES} attempts. Last error: {str(last_exception)}")
    print("Continuing with empty retrieved papers list...")
    return []

def mock_retriever(title: str, abstract: str) -> List[Dict[str, Any]]:
    """Mock retriever that returns empty results with title and abstract format"""
    print("Mock retriever called - returning empty results")
    print(f"Input: title={title[:50]}..., abstract={abstract[:350]}...")
    print(f"TOP_R parameter: {TOP_R}")
    return []



def load_test_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load test data from directory"""
    print(f"Loading test data from {data_dir}...")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory {data_dir} does not exist")
        return []
    
    test_data = []
    
    # Load all JSON files in the directory
    json_files = list(data_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
            test_data.append(paper_data)
    
    print(f"SUCCESS: Loaded {len(test_data)} papers")
    return test_data

def calculate_average_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate average metrics across all papers for multiple top-k values"""
    print("Calculating average metrics across all papers for multiple top-k values...")
    
    # Initialize metrics for each k value
    from task1_metrics import TOP_K_EVALUATE
    total_recall = {f"k={k}": 0.0 for k in TOP_K_EVALUATE}
    total_ndcg = {f"k={k}": 0.0 for k in TOP_K_EVALUATE}
    total_hits = {f"k={k}": 0 for k in TOP_K_EVALUATE}
    total_generated = 0
    total_references = 0
    valid_papers = 0
    
    for result in results:
        if result.get('success') and 'evaluation' in result:
            evaluation = result['evaluation']
            
            # Sum metrics for each k value
            for k in TOP_K_EVALUATE:
                k_key = f"k={k}"
                total_recall[k_key] += evaluation.get('recall_at_k', {}).get(k_key, 0.0)
                total_ndcg[k_key] += evaluation.get('ndcg_at_k', {}).get(k_key, 0.0)
                total_hits[k_key] += evaluation.get('hits', {}).get(k_key, 0)
            
            total_generated += evaluation.get('generated_count', 0)
            total_references += evaluation.get('reference_count', 0)
            valid_papers += 1
    
    if valid_papers == 0:
        print("ERROR: No valid papers found for averaging")
        return {}
    
    # Calculate averages for each k value
    avg_recall_at_k = {k_key: total_recall[k_key] / valid_papers for k_key in total_recall}
    avg_ndcg_at_k = {k_key: total_ndcg[k_key] / valid_papers for k_key in total_ndcg}
    
    avg_metrics = {
        'avg_recall_at_k': avg_recall_at_k,
        'avg_ndcg_at_k': avg_ndcg_at_k,
        'total_hits': total_hits,
        'total_generated': total_generated,
        'total_references': total_references,
        'valid_papers': valid_papers
    }
    
    print(f"SUCCESS: Average metrics calculated for {valid_papers} papers")
    for k in TOP_K_EVALUATE:
        k_key = f"k={k}"
        print(f"Average Recall@{k}: {avg_metrics['avg_recall_at_k'][k_key]:.4f}")
        print(f"Average NDCG@{k}: {avg_metrics['avg_ndcg_at_k'][k_key]:.4f}")
    
    return avg_metrics

def process_batch_papers(client, papers: List[Dict[str, Any]], model_name: str, retriever_url: str) -> List[Dict[str, Any]]:
    """Process multiple papers in batch using high concurrency"""
    print(f"Processing batch of {len(papers)} papers with high concurrency...")
    
    # Generate all references in batch
    all_generated_titles = generate_references_batch(client, papers, model_name)
    
    # Process results
    results = []
    for i, (paper_data, generated_titles) in enumerate(zip(papers, all_generated_titles)):
        print(f"Processing results for paper {i+1}/{len(papers)}")
        
        # Extract paper information
        paper_id = paper_data.get('paper_id', 'unknown')
        title = paper_data.get('title', '')
        reference_labels = paper_data.get('reference_labels', [])
        reference_titles = [ref.get('title', '') for ref in reference_labels if ref.get('title')]
        
        # Check if generation failed
        if len(generated_titles) == 0:
            print(f"WARNING: Generation failed for paper {i+1}, setting metrics to 0")
            evaluation_metrics = {
                'recall_at_k': {},
                'ndcg_at_k': {},
                'hits': {},
                'generated_count': 0,
                'reference_count': len(reference_titles)
            }
            # Set metrics for all k values to 0
            from task1_metrics import TOP_K_EVALUATE
            for k in TOP_K_EVALUATE:
                evaluation_metrics["recall_at_k"][f"k={k}"] = 0.0
                evaluation_metrics["ndcg_at_k"][f"k={k}"] = 0.0
                evaluation_metrics["hits"][f"k={k}"] = 0
        else:
            # Evaluate generation with multiple top-k values
            evaluation_metrics = evaluate_generation_multi_topk(generated_titles, reference_titles)
        
        # Prepare result
        result = {
            'success': True,
            'paper_id': paper_id,
            'title': title,
            'generated_titles': generated_titles,
            'evaluation': evaluation_metrics
        }
        
        results.append(result)
    
    print("SUCCESS: Batch processing completed")
    return results

def save_results(results: List[Dict[str, Any]], avg_metrics: Dict[str, float], output_file: str):
    """Save batch results and average metrics to file"""
    print(f"Saving results to {output_file}...")
    
    output_data = {
        'average_metrics': avg_metrics,
        'individual_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("SUCCESS: Results saved to file")

if __name__ == "__main__":
    print("RAG Generation Module - Starting...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    model_name = args.model
    test_data_dir = args.test_data_dir
    output_dir = args.output_dir
    top_r = args.top_r
    retriever_url = args.url
    
    # Override global TOP_R for functions that use it
    TOP_R = top_r

    print(f"Model: {model_name}")
    print(f"Test data directory: {test_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Retrieved papers from retriever: {top_r}")
    print(f"Retriever URL: {retriever_url}")

    # Setup output directory and file
    output_file = setup_output_directory(model_name, top_r, output_dir)
    
    # Create client
    client = create_flux_client()
    
    # Load test data
    test_data = load_test_data(test_data_dir)
    
    if not test_data:
        print("ERROR: No test data loaded")
        exit(1)
    
    # Process all papers in batch
    print(f"Starting batch evaluation of {len(test_data)} papers...")
    batch_results = process_batch_papers(client, test_data, model_name, retriever_url)
    
    # Calculate average metrics
    avg_metrics = calculate_average_metrics(batch_results)
    
    # Save results
    save_results(batch_results, avg_metrics, output_file)
    
    # Print final summary
    print("=" * 50)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Total papers processed: {len(batch_results)}")
    print(f"Valid papers evaluated: {avg_metrics.get('valid_papers', 0)}")
    
    # Print metrics for each k value
    from task1_metrics import TOP_K_EVALUATE
    for k in TOP_K_EVALUATE:
        k_key = f"k={k}"
        avg_recall = avg_metrics.get('avg_recall_at_k', {}).get(k_key, 0.0)
        avg_ndcg = avg_metrics.get('avg_ndcg_at_k', {}).get(k_key, 0.0)
        total_hits = avg_metrics.get('total_hits', {}).get(k_key, 0)
        print(f"Average Recall@{k}: {avg_recall:.4f}")
        print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
        print(f"Total hits@{k}: {total_hits}")
    
    print(f"Results saved to: {output_file}")
    print("=" * 50)
    
    print("SUCCESS: RAG Generation Module - Completed")
