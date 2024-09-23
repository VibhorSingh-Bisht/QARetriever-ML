#Dataset_preparation.py

import datasets
from transformers import AutoTokenizer

def load_and_preprocess_dataset(dataset_name, tokenizer_name, max_length=512):
    dataset = datasets.load_dataset(f"BeIR/{dataset_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    return tokenized_dataset


nq_dataset = load_and_preprocess_dataset('nq', 'sentence-transformers/all-MiniLM-L6-v2')
hotpotqa_dataset = load_and_preprocess_dataset('hotpotqa', 'sentence-transformers/all-MiniLM-L6-v2')
fiqa_dataset = load_and_preprocess_dataset('fiqa', 'sentence-transformers/all-MiniLM-L6-v2')

print(nq_dataset)
print(hotpotqa_dataset)
print(fiqa_dataset)