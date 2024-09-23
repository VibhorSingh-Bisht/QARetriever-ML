#Retrieval_Pipeline.py

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import faiss
import json  
import pandas as pd  # For CSV handling

class RetrievalPipeline:
    def __init__(self, embedding_model_name, ranking_model_name):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.ranking_model = AutoModelForSequenceClassification.from_pretrained(ranking_model_name)
        self.ranking_tokenizer = AutoTokenizer.from_pretrained(ranking_model_name)
        self.index = None
        self.documents = []  # Initialize documents list

    def index_documents(self, documents):
        self.documents = documents  # Store documents
        embeddings = self.embedding_model.encode(documents)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_candidates(self, query, k=100):
        query_embedding = self.embedding_model.encode([query])
        _, I = self.index.search(query_embedding, k)
        return I[0]

    def rerank_candidates(self, query, candidate_docs):
        inputs = self.ranking_tokenizer([(query, doc) for doc in candidate_docs], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.ranking_model(**inputs).logits.squeeze(-1)
        ranked_indices = torch.argsort(scores, descending=True)
        return [candidate_docs[i] for i in ranked_indices]

    def retrieve(self, query, k=10):
        candidate_indices = self.retrieve_candidates(query, k*10)
        candidate_docs = [self.documents[i] for i in candidate_indices]
        reranked_docs = self.rerank_candidates(query, candidate_docs)
        return reranked_docs[:k]

# Function to load documents from a text file
def load_documents_from_txt(file_path):
    with open(file_path, 'r') as file:
        documents = file.readlines()
    return [doc.strip() for doc in documents]

# Function to load documents from a CSV file
def load_documents_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['text_column'].tolist()  

# Function to load documents from a JSON file
def load_documents_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [item['text'] for item in data]  


file_path = 'fast_food.txt'  
documents = load_documents_from_txt(file_path)  

pipeline = RetrievalPipeline('sentence-transformers/all-MiniLM-L6-v2', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
pipeline.index_documents(documents)

# Query input and retrieval
query = input("Enter your query: ")
results = pipeline.retrieve(query)

# Print results
print("Retrieved Documents:")
for result in results:
    print(result)
