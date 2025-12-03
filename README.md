## Project Structure

```
CiteRAG/
├── corpus/                          # Academic corpus data
│   └── hierarchical_corpus.tar.gz.* # Multi-volume compressed corpus (large size)
├── data/                           # Dataset files
│   ├── trainQA/                    # Training data
│   └── test/                       # Test data
├── retriever/                      # Retrieval system
│   ├── api.py                      # Retrieval API
│   ├── train.sh                    # Training script
│   ├── train_example.jsonl         # Training data example
│   └── eval/                       # Retrieval evaluation
├── eval/                          # Evaluation scripts
│   ├── task1_api_G.py             # Task 1 generation evaluation
│   ├── task1_api_RG.py            # Task 1 RAG evaluation
│   ├── task1_metrics.py           # Task 1 metrics
│   ├── task2_api_G.py             # Task 2 generation evaluation
│   ├── task2_api_RG.py            # Task 2 RAG evaluation
│   └── task2_metrics.py           # Task 2 metrics
├── server/                        # Server configurations
│   ├── serve_retriever.sh         # Retrieval service script
│   └── serve_generator.sh         # Generator service script
├── README.md                      # Project documentation
└── requirements.txt               # requirements
```

**Note**: The corpus is compressed using multi-volume compression due to its large size.

