# Retrieval_pipeline.py

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RetrievalPipeline:
    def __init__(self, embedding_model_name, ranking_model_name):
        self.embedding_model = self.load_embedding_model(embedding_model_name)
        self.ranking_model, self.ranking_tokenizer = self.load_ranking_model(ranking_model_name)
        self.index = None
        self.documents = None

    def load_embedding_model(self, model_name):
        return SentenceTransformer(model_name)

    def load_ranking_model(self, model_name):
        return (
            AutoModelForSequenceClassification.from_pretrained(model_name),
            AutoTokenizer.from_pretrained(model_name)
        )

    def index_documents(self, documents):
        self.documents = documents
        embeddings = self.embedding_model.encode(documents)
        # Implement indexing logic here (e.g., using FAISS)

    def retrieve_candidates(self, query, k=100):
        query_embedding = self.embedding_model.encode([query])
        # Implement retrieval logic here
        # Return indices of top k documents

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

# Usage example
embedding_models = {
    'small': 'sentence-transformers/all-MiniLM-L6-v2',
    'large': 'intfloat/e5-large-v2'  # Replacement for nvidia/nv-embedqa-e5-v5
}

ranking_models = [
    'cross-encoder/ms-marco-MiniLM-L-12-v2',
    'cross-encoder/ms-marco-MiniLM-L-6-v2'  # A smaller alternative to compare
]

# Create pipelines
pipelines = {}
for emb_size, emb_model in embedding_models.items():
    for rank_model in ranking_models:
        key = f"{emb_size}_{rank_model.split('/')[-1]}"
        pipelines[key] = RetrievalPipeline(emb_model, rank_model)

print("Created pipelines:", list(pipelines.keys()))