# Khmer Word Search with BERT

Semantic search engine for Khmer text using Khmer Wikipedia data and XLM-RoBERTa embeddings.

## What It Does

1. **Processes Khmer Wikipedia** - Extracts and cleans text from XML dumps
2. **Creates embeddings** - Uses `channudam/khmer-xlm-roberta-base` model to generate semantic vectors
3. **Enables semantic search** - Find similar sentences using cosine similarity
4. **Interactive interface** - Command-line search with real-time results

## Features

- **Khmer-specific preprocessing**: Handles Khmer sentence markers (។ ៕) and Unicode ranges
- **Smart text cleaning**: Removes English text, HTML tags, and wiki markup
- **BERT embeddings**: Uses pre-trained Khmer XLM-RoBERTa for semantic understanding
- **Efficient processing**: Batch processing with progress bars and memory optimization
- **Cross-platform**: Supports CPU and Mac GPU (MPS) acceleration

## Installation

```bash
pip install torch transformers tqdm khmernltk numpy
```

## Quick Start

### 1. Download Data
Get Khmer Wikipedia XML dump from [Wikimedia Downloads](https://dumps.wikimedia.org/kmwiki/)

### 2. Configure Paths
Edit the file paths in `main.py`:
```python
xml_path = "kmwiki-latest-pages-articles.xml"
```

### 3. Run Processing
```python
# First run: Process Wikipedia data
process_new_data = True

# Subsequent runs: Use cached data
process_new_data = False
```

### 4. Start Searching
```bash
python main.py
```

Enter Khmer queries in the interactive interface:
```
> ប្រទេសកម្ពុជា
> អាហារខ្មែរ
> ប្រវត្តិសាស្ត្រ
```

## File Structure

```
├── main.py                      # Main script
├── kmwiki-latest-pages-articles.xml  # Wikipedia XML dump
├── khmer_processed_data.pkl     # Processed sentences
├── khmer_embeddings.pkl         # Generated embeddings
├── khmer_embedding_db.pkl       # Search database
└── khmer_cleaned_sentences.txt  # Human-readable output
```

## Key Functions

- **`preprocess_khmer_wiki()`** - Complete preprocessing pipeline
- **`setup_bert_model()`** - Load Khmer XLM-RoBERTa model
- **`generate_sentence_embeddings()`** - Create semantic vectors
- **`khmer_word_search()`** - Perform similarity search
- **`interactive_search_interface()`** - Command-line interface

## Model Details

- **Base Model**: `channudam/khmer-xlm-roberta-base`
- **Training**: Fine-tuned on 26K+ Khmer sentences
- **Pooling**: CLS token extraction (configurable to mean pooling)
- **Similarity**: Cosine similarity with temperature scaling

## Configuration Options

```python
# Processing
max_pages = None        # Process all pages (or set limit)
batch_size = 16        # Embedding batch size
max_length = 128       # Token sequence length

# Search
top_n = 5             # Number of results
pooling_method = 'cls' # 'cls' or 'mean'
temperature = 0.01     # Similarity scaling
```

## Performance

- **Processing**: ~1000 sentences/minute (CPU), faster with GPU
- **Search**: Sub-second response times
- **Memory**: Efficient batch processing prevents OOM errors
- **Storage**: Embeddings cached for instant subsequent runs

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- khmernltk
- NumPy
- tqdm

## License

Open source - modify as needed for your use case.
