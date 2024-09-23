import numpy as np
from Retrieval_pipeline import RetrievalPipeline, load_documents_from_txt

# Function to calculate NDCG
def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)

def ndcg_at_k(relevances, k):
    best_relevances = sorted(relevances, reverse=True)
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(best_relevances, k)
    return dcg / idcg if idcg > 0 else 0.

def evaluate_retrieval(pipeline, query, true_relevance, k=10):
    retrieved_docs = pipeline.retrieve(query, k)
    retrieved_relevances = [true_relevance.get(doc, 0) for doc in retrieved_docs]
    return ndcg_at_k(retrieved_relevances, k)

# Example usage
file_path = 'path/to/your/fast_food_data.txt'  # Change to your actual file path
documents = load_documents_from_txt(file_path)

# Initialize the pipeline
pipeline = RetrievalPipeline('sentence-transformers/all-MiniLM-L6-v2', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
pipeline.index_documents(documents)

# Sample true relevance for a specific query (you can adjust this as needed)
true_relevance = {"What is a popular fast food item?": {"burger": 3, "fries": 2, "salad": 1}}

# Runtime query input
query = input("Enter your query: ")
ndcg_score = evaluate_retrieval(pipeline, query, true_relevance, k=10)

# Print the NDCG score
print(f"NDCG@10 for the query '{query}': {ndcg_score}")

# Optionally print retrieved documents
results = pipeline.retrieve(query)
print("Retrieved Documents:")
for result in results:
    print(result)
