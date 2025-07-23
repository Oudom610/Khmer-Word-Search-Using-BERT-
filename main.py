import xml.etree.ElementTree as ET
import re
import pickle
from tqdm import tqdm 
import torch
import numpy as np
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import os
from time import time
from khmernltk import word_tokenize

def load_khmer_wiki_data(xml_file_path, max_pages=None):
    """
    Load and extract text content from Khmer Wikipedia XML dump
    
    Args:
        xml_file_path (str): Path to the XML file
        max_pages (int, optional): Maximum number of pages to process. If None, process all pages.
        
    Returns:
        list: List of text content from each article
    """
    print("Loading XML file. This may take a while...")
    
    # Parse the XML namespaces
    namespaces = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
    
    # List to store text content from all pages
    all_text_content = []
    
    # Create an iterator for the XML to avoid loading the entire file into memory
    context = ET.iterparse(xml_file_path, events=('end',))
    
    # Count for progress reporting
    article_count = 0
    
    for event, elem in tqdm(context):
        # Look for page elements
        if elem.tag.endswith('page'):
            # Extract the namespace
            ns_page = elem.tag.split('}')[0] + '}'
            
            # Find the text content (located in revision/text)
            text_elem = elem.find(f'.//{ns_page}revision/{ns_page}text')
            
            if text_elem is not None and text_elem.text:
                all_text_content.append(text_elem.text)
                article_count += 1
                
                # Check if we've reached the maximum number of pages
                if max_pages is not None and article_count >= max_pages:
                    print(f"Reached maximum number of pages ({max_pages})")
                    break
            
            # Clear the element to save memory
            elem.clear()
    
    print(f"Extracted text content from {article_count} articles")
    return all_text_content


def segment_khmer_text(text_list):
    """
    Segment Khmer text into sentences based on Khmer-specific sentence markers
    
    Args:
        text_list (list): List of text content
        
    Returns:
        list: List of sentences
    """
    all_sentences = []
    
    # Khmer sentence markers
    # Khan (។) is the primary sentence ending marker in Khmer
    # Bariyoosan (៕) marks the end of a chapter or text
    # include Western punctuation that might be used
    sentence_markers = r'[។៕\.\?\!]'
    
    for text in tqdm(text_list, desc="Segmenting text"):
        # Split the text by sentence markers
        # Look ahead to include the marker in the split result
        raw_sentences = re.split(f'({sentence_markers})', text)
        
        # Combine sentences with their markers
        sentences = []
        for i in range(0, len(raw_sentences) - 1, 2):
            if i + 1 < len(raw_sentences):
                sentence = raw_sentences[i] + raw_sentences[i + 1]
                sentences.append(sentence.strip())
        
        # Handle any remaining text without markers
        if len(raw_sentences) % 2 == 1 and raw_sentences[-1].strip():
            sentences.append(raw_sentences[-1].strip())
        
        all_sentences.extend(sentences)
    
    return all_sentences


def clean_khmer_sentences(sentences):
    """
    Clean Khmer sentences by removing English text, unnecessary symbols, and extra whitespace
    
    Args:
        sentences (list): List of sentences
        
    Returns:
        list: List of cleaned sentences
    """
    cleaned_sentences = []
    
    # Pattern to identify Latin script (English text)
    english_pattern = r'[a-zA-Z0-9]+'
    
    # Pattern for HTML tags and Wiki markup
    html_wiki_pattern = r'<.*?>|\[\[.*?\]\]|\{\{.*?\}\}|\[\w+:.+?\]'
    
    # Special characters to remove (excluding Khmer characters and punctuation)
    special_chars_pattern = r'[^\u1780-\u17FF\u19E0-\u19FF។៕៖ៗ\?\!\.\s]+'
    
    for sentence in tqdm(sentences, desc="Cleaning sentences"):
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # Remove HTML tags and Wiki markup
        cleaned = re.sub(html_wiki_pattern, ' ', sentence)
        
        # Remove English text
        cleaned = re.sub(english_pattern, ' ', cleaned)
        
        # Remove special characters
        cleaned = re.sub(special_chars_pattern, ' ', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Count Khmer words (approximately)
        khmer_words = re.findall(r'[\u1780-\u17FF]+', cleaned)
        
        # Skip if the cleaned sentence is too short, empty, or has too few Khmer words
        if len(cleaned) > 10 and len(khmer_words) >= 3:  # Increased minimum length and require at least 3 Khmer words
            cleaned_sentences.append(cleaned)
    
    return cleaned_sentences

def tokenize_khmer_sentences(cleaned_sentences):
    """
    Tokenize Khmer sentences using khmer-nltk
    
    Args:
        cleaned_sentences (list): List of cleaned Khmer sentences
        
    Returns:
        list: List of tokenized sentences (each sentence is a list of tokens)
    """
    tokenized_sentences = []
    
    for sentence in tqdm(cleaned_sentences, desc="Tokenizing sentences"):
        try:
            # Use khmer-nltk to tokenize the sentence
            tokens = word_tokenize(sentence)
            
            # Only keep sentences with meaningful tokens
            if tokens and len(tokens) >= 3:  # Require at least 3 tokens
                tokenized_sentences.append(tokens)
        except Exception as e:
            print(f"Error tokenizing sentence: {e}")
            continue
    
    print(f"Successfully tokenized {len(tokenized_sentences)} sentences")
    return tokenized_sentences


def store_processed_data(original_sentences, cleaned_sentences, tokenized_sentences, output_file):
    """
    Store processed data in a structured format
    
    Args:
        original_sentences (list): List of original sentences
        cleaned_sentences (list): List of cleaned sentences
        tokenized_sentences (list): List of tokenized sentences
        output_file (str): Path to save the processed data
        
    Returns:
        dict: Dictionary containing the processed data
    """
    # Create a data structure to store processed data
    processed_data = {
        'original_sentences': original_sentences,
        'cleaned_sentences': cleaned_sentences,
        'tokenized_sentences': tokenized_sentences
    }
    
    # Save the data structure to a file using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Saved processed data to {output_file}")
    
    return processed_data


def load_processed_data(input_file):
    """
    Load previously processed data
    
    Args:
        input_file (str): Path to the processed data file
        
    Returns:
        dict: Dictionary containing the processed data
    """
    with open(input_file, 'rb') as f:
        processed_data = pickle.load(f)
    
    print(f"Loaded processed data from {input_file}")
    print(f"Number of original sentences: {len(processed_data['original_sentences'])}")
    print(f"Number of cleaned sentences: {len(processed_data['cleaned_sentences'])}")
    print(f"Number of tokenized sentences: {len(processed_data['tokenized_sentences'])}")
    
    return processed_data


def preprocess_khmer_wiki(xml_file_path, text_output_file=None, data_output_file=None, max_pages=None):
    """
    Main function to load, segment, clean, and tokenize Khmer Wikipedia text
    
    Args:
        xml_file_path (str): Path to the XML file
        text_output_file (str, optional): Path to save cleaned sentences as text
        data_output_file (str, optional): Path to save processed data structure
        max_pages (int, optional): Maximum number of pages to process. If None, process all pages.
        
    Returns:
        dict: Dictionary containing the processed data
    """
    # Step 1: Load the wiki data
    wiki_text = load_khmer_wiki_data(xml_file_path, max_pages=max_pages)
    print(f"Loaded {len(wiki_text)} articles")
    
    # Step 2: Segment into sentences
    original_sentences = segment_khmer_text(wiki_text)
    print(f"Segmented into {len(original_sentences)} sentences")
    
    # Step 3: Clean the sentences
    cleaned_sentences = clean_khmer_sentences(original_sentences)
    print(f"Cleaned {len(cleaned_sentences)} sentences")
    
    # Optionally save cleaned sentences to text file
    if text_output_file:
        with open(text_output_file, 'w', encoding='utf-8') as f:
            for sentence in cleaned_sentences:
                f.write(f"{sentence}\n")
        print(f"Saved cleaned sentences to {text_output_file}")
    
    # Step 4: Tokenize the sentences
    tokenized_sentences = tokenize_khmer_sentences(cleaned_sentences)
    
    # Step 5: Store the processed data
    if data_output_file:
        processed_data = store_processed_data(
            original_sentences, 
            cleaned_sentences, 
            tokenized_sentences, 
            data_output_file
        )
    else:
        processed_data = {
            'original_sentences': original_sentences,
            'cleaned_sentences': cleaned_sentences,
            'tokenized_sentences': tokenized_sentences
        }
    
    return processed_data

def setup_bert_model():
    """
    Set up the pretrained Khmer XLM-RoBERTa model and tokenizer
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    print("Setting up Khmer-specific XLM-RoBERTa model...")
    
    # Check if Mac GPU is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on macOS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load pretrained Khmer-specific XLM-RoBERTa tokenizer and model
    # This model is fine-tuned on 26K+ Khmer sentences
    model_name = 'channudam/khmer-xlm-roberta-base'
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaModel.from_pretrained(model_name)
    
    # Move model to the device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Khmer-specific XLM-RoBERTa model '{model_name}' loaded successfully")
    return model, tokenizer, device

def generate_sentence_embeddings(model, tokenizer, device, sentences, batch_size=16, max_length=128, 
                                 embeddings_file=None, pooling_method='cls'):
    """
    Generate embeddings for a list of Khmer sentences using Khmer-specific XLM-RoBERTa
    
    Args:
        model: The loaded Khmer-specific XLM-RoBERTa model
        tokenizer: The loaded Khmer-specific XLM-RoBERTa tokenizer
        device: The device (CPU/GPU) to run inference on
        sentences (list): List of Khmer sentences
        batch_size (int): Number of sentences to process in a batch
        max_length (int): Maximum sequence length for tokenization
        embeddings_file (str, optional): Path to save embeddings
        pooling_method (str): 'cls' for CLS token or 'mean' for mean pooling
    
    Returns:
        numpy.ndarray: Array of sentence embeddings (num_sentences, embedding_dim)
    """
    print(f"\nGenerating sentence embeddings using '{pooling_method}' pooling method...")
    print(f"Processing {len(sentences)} sentences in batches of {batch_size}")
    
    # Check if embeddings file exists
    if embeddings_file and os.path.exists(embeddings_file):
        print(f"Loading pre-computed embeddings from {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            embeddings_data = pickle.load(f)
            return embeddings_data['embeddings']
    
    # List to store all embeddings
    all_embeddings = []
    
    # Process sentences in batches
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    
    # Track time
    start_time = time()
    
    for i in tqdm(range(num_batches), desc="Generating embeddings"):
        # Get the current batch of sentences
        batch_sentences = sentences[i * batch_size:(i + 1) * batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(
            batch_sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model output (without computing gradients)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embeddings based on pooling method
        if pooling_method == 'cls':
            # Option 1: Extract [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        else:  # 'mean' pooling
            # Option 2: Mean/Average pooling over all tokens (excluding padding)
            # Get attention mask to exclude padding tokens
            attention_mask = inputs['attention_mask'].cpu().numpy()
            
            # Get all token embeddings and move to CPU
            token_embeddings = outputs.last_hidden_state.cpu().numpy()
            
            # Initialize an array to store the mean-pooled embeddings
            batch_embeddings = np.zeros((token_embeddings.shape[0], token_embeddings.shape[2]))
            
            # Compute mean pooling for each sentence in the batch
            for j in range(token_embeddings.shape[0]):
                # Get valid token positions (where attention mask is 1)
                valid_tokens = attention_mask[j] == 1
                
                # Compute mean over valid tokens
                if valid_tokens.sum() > 0:  # Avoid division by zero
                    batch_embeddings[j] = token_embeddings[j, valid_tokens, :].mean(axis=0)
        
        # Add batch embeddings to the list
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings
    embeddings = np.vstack(all_embeddings)
    
    # Calculate time taken
    time_taken = time() - start_time
    print(f"Embedding generation complete. Time taken: {time_taken:.2f} seconds")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save embeddings to file if specified
    if embeddings_file:
        embedding_data = {'embeddings': embeddings, 'pooling_method': pooling_method}
        print(f"Saving embeddings to {embeddings_file}")
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embedding_data, f)
    
    return embeddings

def create_embedding_database(embeddings, sentences, index_file=None):
    """
    Create an embedding database with index mapping to original sentences
    
    Args:
        embeddings (numpy.ndarray): Array of sentence embeddings
        sentences (list): List of original sentences
        index_file (str, optional): Path to save the embedding database
    
    Returns:
        dict: Embedding database
    """
    print("\nCreating embedding database...")
    
    # Check that embeddings and sentences align
    assert len(embeddings) == len(sentences), "Number of embeddings must match number of sentences"
    
    # Create the database
    embedding_db = {
        'embeddings': embeddings,
        'sentences': sentences,
        'embedding_dim': embeddings.shape[1],
        'num_sentences': len(sentences)
    }
    
    print(f"Created embedding database with {embedding_db['num_sentences']} entries")
    print(f"Embedding dimension: {embedding_db['embedding_dim']}")
    
    # Save database to file if specified
    if index_file:
        print(f"Saving embedding database to {index_file}")
        with open(index_file, 'wb') as f:
            pickle.dump(embedding_db, f)
    
    return embedding_db

def clean_khmer_query(query):
    """
    Clean a Khmer query by applying the same preprocessing as the corpus
    
    Args:
        query (str): Input Khmer query
        
    Returns:
        str: Cleaned query
    """
    # Pattern to identify Latin script (English text)
    english_pattern = r'[a-zA-Z0-9]+'
    
    # Pattern for HTML tags and Wiki markup
    html_wiki_pattern = r'<.*?>|\[\[.*?\]\]|\{\{.*?\}\}|\[\w+:.+?\]'
    
    # Special characters to remove (excluding Khmer characters and punctuation)
    special_chars_pattern = r'[^\u1780-\u17FF\u19E0-\u19FF។៕៖ៗ\?\!\.\s]+'
    
    # Skip empty query
    if not query.strip():
        return ""
    
    # Remove HTML tags and Wiki markup
    cleaned = re.sub(html_wiki_pattern, ' ', query)
    
    # Remove English text
    cleaned = re.sub(english_pattern, ' ', cleaned)
    
    # Remove special characters
    cleaned = re.sub(special_chars_pattern, ' ', cleaned)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def validate_khmer_input(input_text):
    """
    Validate that the input contains Khmer characters
    
    Args:
        input_text (str): Input text to validate
        
    Returns:
        bool: True if the input contains Khmer characters, False otherwise
    """
    # Check if input is empty
    if not input_text or not input_text.strip():
        print("Error: Input is empty.")
        return False
    
    # Khmer Unicode range: \u1780-\u17FF and \u19E0-\u19FF
    khmer_pattern = re.compile(r'[\u1780-\u17FF\u19E0-\u19FF]')
    
    # Check if input contains at least one Khmer character
    if not khmer_pattern.search(input_text):
        print("Error: Input does not contain Khmer characters.")
        return False
    
    return True

def compute_query_embedding(model, tokenizer, device, query_text, max_length=128, pooling_method='cls'):
    """
    Compute embedding for a query string using the same method as for corpus sentences
    
    Args:
        model: The loaded Khmer-specific XLM-RoBERTa model
        tokenizer: The loaded Khmer-specific XLM-RoBERTa tokenizer
        device: The device (CPU/GPU) to run inference on
        query_text (str): Cleaned query text
        max_length (int): Maximum sequence length for tokenization
        pooling_method (str): 'cls' for CLS token or 'mean' for mean/average pooling
        
    Returns:
        numpy.ndarray: Query embedding vector
    """
    # Tokenize the query
    inputs = tokenizer(
        query_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model output (without computing gradients)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get embedding based on pooling method
    if pooling_method == 'cls':
        # Option 1: Extract [CLS] token embedding (first token)
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    else:  # 'mean' pooling
        # Option 2: Mean pooling over all tokens (excluding padding)
        # Get attention mask to exclude padding tokens
        attention_mask = inputs['attention_mask'].cpu().numpy()[0]
        
        # Get all token embeddings and move to CPU
        token_embeddings = outputs.last_hidden_state.cpu().numpy()[0]
        
        # Get valid token positions (where attention mask is 1)
        valid_tokens = attention_mask == 1
        
        # Compute mean over valid tokens
        if valid_tokens.sum() > 0:  # Avoid division by zero
            query_embedding = token_embeddings[valid_tokens, :].mean(axis=0)
        else:
            # Fallback if no valid tokens (shouldn't happen with proper inputs)
            query_embedding = np.zeros(token_embeddings.shape[1])
    
    return query_embedding

def compute_cosine_similarities(query_embedding, corpus_embeddings, temperature=0.01):
    """
    Compute cosine similarities between a query embedding and all corpus embeddings,
    with temperature scaling to spread out the similarity distribution
    
    Args:
        query_embedding (numpy.ndarray): Query embedding vector
        corpus_embeddings (numpy.ndarray): Matrix of corpus embeddings
        temperature (float): Temperature parameter (lower values spread out the distribution)
        
    Returns:
        numpy.ndarray: Array of similarity scores
    """
    # Normalize query embedding (L2 norm)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        normalized_query = query_embedding / query_norm
    else:
        normalized_query = query_embedding
    
    # Normalize corpus embeddings (L2 norm)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    normalized_corpus = corpus_embeddings / (corpus_norms + epsilon)
    
    # Compute dot products (vectorized operation)
    similarities = np.dot(normalized_corpus, normalized_query)
    
    # Apply power transformation to spread out similarity distribution
    similarities = np.power(similarities, 1/temperature)
    
    return similarities

def get_top_matches(similarities, sentences, top_n=5):
    """
    Get the top N matches based on similarity scores
    
    Args:
        similarities (numpy.ndarray): Array of similarity scores
        sentences (list): List of original sentences
        top_n (int): Number of top matches to return
        
    Returns:
        list: List of tuples (sentence, score)
    """
    # Get indices of top N similarity scores
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Get corresponding sentences and scores
    top_matches = [(sentences[i], float(similarities[i])) for i in top_indices]
    
    return top_matches

def format_search_results(query, matches):
    """
    Format search results for display
    
    Args:
        query (str): Original query text
        matches (list): List of tuples (sentence, score)
        
    Returns:
        str: Formatted search results
    """
    # Create header
    result = f"\n{'=' * 60}\n"
    result += f"SEARCH RESULTS FOR: '{query}'\n"
    result += f"{'=' * 60}\n\n"
    
    # Format each match
    for i, (sentence, score) in enumerate(matches):
        # Limit sentence display length if too long
        display_sentence = sentence
        if len(display_sentence) > 100:
            display_sentence = display_sentence[:97] + "..."
            
        result += f"{i+1}. {display_sentence}\n"
        result += f"   Score: {score:.4f}\n\n"
    
    return result


def khmer_word_search(query, embedding_db, model, tokenizer, device, top_n=5, pooling_method='cls'):
    """
    Main function for performing Khmer word/phrase search
    
    Args:
        query (str): Input Khmer query
        embedding_db (dict): Embedding database
        model: The loaded Khmer-specific XLM-RoBERTa model
        tokenizer: The loaded Khmer-specific XLM-RoBERTa tokenizer
        device: The device (CPU/GPU) to run inference on
        top_n (int): Number of top matches to return
        pooling_method (str): 'cls' for CLS token or 'mean' for mean pooling
        
    Returns:
        tuple: (formatted_results, matches)
    """
    print(f"\nProcessing query: '{query}'")
    
    # Step 1: Validate input
    if not validate_khmer_input(query):
        return "Invalid input. Please enter Khmer text.", []
    
    # Step 2: Clean query
    cleaned_query = clean_khmer_query(query)
    print(f"Cleaned query: '{cleaned_query}'")
    
    if not cleaned_query:
        return "Invalid input after cleaning. Please enter valid Khmer text.", []
    
    # Step 3: Compute query embedding
    start_time = time()
    query_embedding = compute_query_embedding(
        model=model,
        tokenizer=tokenizer,
        device=device,
        query_text=cleaned_query,
        pooling_method=pooling_method
    )
    
    # Step 4: Compute similarities
    similarities = compute_cosine_similarities(
        query_embedding=query_embedding,
        corpus_embeddings=embedding_db['embeddings']
    )
    
    # Step 5: Get top matches
    matches = get_top_matches(
        similarities=similarities,
        sentences=embedding_db['sentences'],
        top_n=top_n
    )
    
    # Calculate time taken
    time_taken = time() - start_time
    print(f"Search completed in {time_taken:.2f} seconds")
    
    # Step 6: Format results
    formatted_results = format_search_results(query, matches)
    
    return formatted_results, matches

def interactive_search_interface(embedding_db_path):
    """
    Interactive command-line interface for Khmer word search
    
    Args:
        embedding_db_path (str): Path to the embedding database file
    """
    print("\n" + "=" * 80)
    print("KHMER WORD SEARCH USING BERT".center(80))
    print("=" * 80)
    
    # Load embedding database
    with open(embedding_db_path, 'rb') as f:
        embedding_db = pickle.load(f)
    
    print(f"Loaded embedding database with {embedding_db['num_sentences']} sentences")
    
    # Setup BERT model
    model, tokenizer, device = setup_bert_model()
    
    # Main search loop
    while True:
        print("\nEnter a Khmer word or phrase to search (or 'exit' to quit):")
        query = input("> ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Exiting search interface...")
            break
        
        # Perform search
        formatted_results, matches = khmer_word_search(
            query=query,
            embedding_db=embedding_db,
            model=model,
            tokenizer=tokenizer,
            device=device,
            pooling_method='cls' 
        )
        
        # Display results
        print(formatted_results)

if __name__ == "__main__":
    xml_path = "kmwiki-latest-pages-articles.xml"
    text_output_path = "khmer_cleaned_sentences.txt"
    data_output_path = "khmer_processed_data.pkl"
    embeddings_output_path = "khmer_embeddings.pkl"
    embedding_db_path = "khmer_embedding_db.pkl"
    
    # Option 1: Process new data
    process_new_data = False  
    
    if process_new_data:
        # Process the data
        processed_data = preprocess_khmer_wiki(
            xml_path, 
            text_output_file=text_output_path,
            data_output_file=data_output_path,
            max_pages=None
        )
    else:
        # Option 2: Load previously processed data
        processed_data = load_processed_data(data_output_path)
    
    # Setup the BERT model
    model, tokenizer, device = setup_bert_model()
    
    # Check if embedding database exists
    if not os.path.exists(embedding_db_path) or process_new_data:
        print("Generating new embeddings...")
        
        # Generate embeddings for sentences
        embeddings = generate_sentence_embeddings(
            model=model,
            tokenizer=tokenizer,
            device=device,
            sentences=processed_data['cleaned_sentences'],
            batch_size=16,  
            max_length=128,
            embeddings_file=embeddings_output_path,
            pooling_method='cls'  
        )
        
        # Create embedding database
        embedding_db = create_embedding_database(
            embeddings=embeddings,
            sentences=processed_data['cleaned_sentences'],
            index_file=embedding_db_path
        )
    
    # Launch interactive search interface
    interactive_search_interface(embedding_db_path)