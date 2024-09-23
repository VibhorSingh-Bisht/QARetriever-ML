#main.py

import os
import json
from Dataset_preparation import prepare_beir_datasets
from Retrieval_Pipeline import RetrievalPipeline, embedding_models, ranking_models
from Evaluation_Metric import compare_pipelines

# Configuration
dataset_names = ['nq', 'hotpotqa', 'fiqa']
max_length = 512
chunk_size = 200
top_k = 10

def main():
    # Step 1: Prepare datasets
    print("Preparing datasets...")
    prepared_datasets = prepare_beir_datasets(dataset_names, embedding_models, max_length, chunk_size)

    # Step 2: Create pipelines
    print("Creating retrieval pipelines...")
    pipelines = {}
    for emb_size, emb_model in embedding_models.items():
        for rank_model in ranking_models:
            key = f"{emb_size}_{rank_model.split('/')[-1]}"
            pipelines[key] = RetrievalPipeline(emb_model, rank_model)

    # Step 3: Run experiments for each dataset
    results = {}
    for dataset_name, dataset in prepared_datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        dataset_results = {}

        for pipeline_name, pipeline in pipelines.items():
            print(f"Running pipeline: {pipeline_name}")
            emb_size = pipeline_name.split('_')[0]
            
            # Index documents
            pipeline.index_documents(dataset[emb_size]['corpus']['text'])

            # Prepare queries and relevance judgments
            queries = dataset[emb_size]['queries']['text']
            relevance_judgments = dataset['answers']  # Assume this is in the correct format

            # Run evaluation
            ndcg_score = compare_pipelines({pipeline_name: pipeline}, queries, relevance_judgments, k=top_k)
            dataset_results[pipeline_name] = ndcg_score[pipeline_name]

        results[dataset_name] = dataset_results

    # Step 4: Save results
    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Step 5: Print summary
    print("\nExperiment completed. Summary of results:")
    for dataset, dataset_results in results.items():
        print(f"\n{dataset} dataset:")
        for pipeline, score in dataset_results.items():
            print(f"  {pipeline}: NDCG@{top_k} = {score:.4f}")

if __name__ == "__main__":
    main()