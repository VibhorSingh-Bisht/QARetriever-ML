import datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm

def load_beir_dataset(dataset_name):
    """Load a dataset from the BEIR benchmark."""
    return datasets.load_dataset(f"BeIR/{dataset_name}")

def preprocess_dataset(dataset, tokenizer, max_length=512, chunk_size=None):
    """Preprocess the dataset by tokenizing and optionally chunking."""
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
    def chunk_text(text, chunk_size):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    def chunk_function(examples):
        if chunk_size:
            chunked_texts = [chunk_text(text, chunk_size) for text in examples['text']]
            return {
                'text': sum(chunked_texts, []),
                'id': sum([[id_]*len(chunks) for id_, chunks in zip(examples['id'], chunked_texts)], [])
            }
        return examples
    
    # First, chunk the text if necessary
    if chunk_size:
        dataset = dataset.map(chunk_function, batched=True, remove_columns=dataset['corpus'].column_names)
    
    # Then, tokenize the text
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    return tokenized_dataset

def prepare_beir_datasets(dataset_names, tokenizer_names, max_length=512, chunk_size=None):
    """Prepare multiple BEIR datasets."""
    prepared_datasets = {}
    
    for name in tqdm(dataset_names, desc="Preparing datasets"):
        dataset = load_beir_dataset(name)
        prepared_datasets[name] = {}
        
        for tokenizer_key, tokenizer_name in tokenizer_names.items():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            prepared_datasets[name][tokenizer_key] = {
                'corpus': preprocess_dataset(dataset['corpus'], tokenizer, max_length, chunk_size),
                'queries': preprocess_dataset(dataset['queries'], tokenizer, max_length)
            }
        
        if 'answers' in dataset:
            prepared_datasets[name]['answers'] = dataset['answers']
    
    return prepared_datasets

# Usage
dataset_names = ['nq', 'hotpotqa', 'fiqa']
tokenizer_names = {
    'small': 'sentence-transformers/all-MiniLM-L6-v2',
    'large': 'intfloat/e5-large-v2'
}

prepared_datasets = prepare_beir_datasets(dataset_names, tokenizer_names, max_length=512, chunk_size=200)

# Print some information about the prepared datasets
for name, dataset in prepared_datasets.items():
    print(f"\n{name} dataset:")
    for tokenizer_key in tokenizer_names.keys():
        print(f"  {tokenizer_key.capitalize()} tokenizer:")
        print(f"    Corpus size: {len(dataset[tokenizer_key]['corpus'])}")
        print(f"    Number of queries: {len(dataset[tokenizer_key]['queries'])}")
    if 'answers' in dataset:
        print(f"  Number of answers: {len(dataset['answers'])}")
    print(f"  Example query: {dataset[list(tokenizer_names.keys())[0]]['queries'][0]['text']}")
    print(f"  Example corpus item: {dataset[list(tokenizer_names.keys())[0]]['corpus'][0]['text'][:100]}...")