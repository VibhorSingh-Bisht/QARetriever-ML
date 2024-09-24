#Dataset_preparation.py

import datasets
from transformers import AutoTokenizer

def load_and_preprocess_dataset(dataset_name, config_name, tokenizer_name, max_length=512):
    # Load dataset with the specified config
    dataset = datasets.load_dataset(f"BeIR/{dataset_name}", config_name)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
    # Apply preprocessing to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    return tokenized_dataset

def prepare_beir_datasets(dataset_names, embedding_models, max_length=512, chunk_size=200):
    prepared_datasets = {}
    
    for dataset_name in dataset_names:
        dataset_dict = {}
        for emb_size, emb_model in embedding_models.items():
            # Load the 'corpus' and 'queries' configurations
            corpus_dataset = load_and_preprocess_dataset(dataset_name, 'corpus', emb_model, max_length)
            queries_dataset = load_and_preprocess_dataset(dataset_name, 'queries', emb_model, max_length)
            
            # Chunk the dataset if necessary
            if chunk_size:
                # Implement chunking logic here
                pass
            
            dataset_dict[emb_size] = {
                'corpus': corpus_dataset,
                'queries': queries_dataset
            }
        
        # Retrieve relevance judgments or answers if available
        relevance_judgments = None
        if 'answers' in corpus_dataset:
            relevance_judgments = corpus_dataset['answers']
        elif 'answers' in queries_dataset:
            relevance_judgments = queries_dataset['answers']
        
        prepared_datasets[dataset_name] = {
            'corpus': dataset_dict,
            'answers': relevance_judgments
        }
    
    return prepared_datasets

# Example usage
if __name__ == "__main__":
    dataset_names = ['nq', 'hotpotqa', 'fiqa']
    embedding_models = {
        'small': 'sentence-transformers/all-MiniLM-L6-v2',
        'large': 'intfloat/e5-large-v2'
    }
    prepared_datasets = prepare_beir_datasets(dataset_names, embedding_models)
    for name, dataset in prepared_datasets.items():
        print(f"{name} dataset prepared with the following keys: {dataset.keys()}")