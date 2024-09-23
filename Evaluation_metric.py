import numpy as np

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

def evaluate_retrieval(pipeline, queries, true_relevances, k=10):
    ndcg_scores = []
    for query, true_relevance in zip(queries, true_relevances):
        retrieved_docs = pipeline.retrieve(query, k)
        retrieved_relevances = [true_relevance.get(doc, 0) for doc in retrieved_docs]
        ndcg_scores.append(ndcg_at_k(retrieved_relevances, k))
    return np.mean(ndcg_scores)

# Example usage
queries = ["query1", "query2", "query3"]
true_relevances = [{"doc1": 3, "doc2": 2, "doc3": 1}, {"doc1": 2, "doc2": 3, "doc3": 1}, {"doc1": 1, "doc2": 2, "doc3": 3}]
ndcg_10 = evaluate_retrieval(pipeline, queries, true_relevances, k=10)
print(f"NDCG@10: {ndcg_10}")