#Evaluation_metric.py

import numpy as np
from tqdm import tqdm

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

def evaluate_pipeline(pipeline, queries, true_relevances, k=10):
    ndcg_scores = []
    for query, true_relevance in tqdm(zip(queries, true_relevances), total=len(queries)):
        retrieved_docs = pipeline.retrieve(query, k)
        retrieved_relevances = [true_relevance.get(doc, 0) for doc in retrieved_docs]
        ndcg_scores.append(ndcg_at_k(retrieved_relevances, k))
    return np.mean(ndcg_scores)

def compare_pipelines(pipelines, queries, true_relevances, k=10):
    results = {}
    for name, pipeline in pipelines.items():
        print(f"Evaluating {name}")
        results[name] = evaluate_pipeline(pipeline, queries, true_relevances, k)
    
    # Calculate improvement percentages
    baseline = min(results.values())
    for name, score in results.items():
        improvement = ((score - baseline) / baseline) * 100
        print(f"{name}: NDCG@{k} = {score:.4f} (Improvement: {improvement:.2f}%)")

    return results

# You would then run this comparison for your different pipeline configurations