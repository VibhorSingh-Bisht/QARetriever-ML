import torch
from sentence_transformers import SentenceTransformer

# Initialize query and passages
query = "Your query here"  # Replace with your query
passages = ["Passage 1", "Passage 2", "Passage 3", ... ]  # List of passages
k = 5  # Number of top passages to retrieve

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode query and passages (return PyTorch tensors directly)
query_embedding = model.encode([query], convert_to_tensor=True)  # Query is a list
passage_embeddings = model.encode(passages, convert_to_tensor=True)

# Compute cosine similarity using PyTorch
# Cosine similarity: dot product of normalized vectors
query_embedding_norm = query_embedding / query_embedding.norm(dim=1)  # Normalize query
passage_embeddings_norm = passage_embeddings / passage_embeddings.norm(dim=1, keepdim=True)  # Normalize passages

cosine_scores = torch.matmul(query_embedding_norm, passage_embeddings_norm.T)  # Matrix multiplication for similarity

# Retrieve the top-k highest scoring passages
top_k_indices = torch.topk(cosine_scores, k=k, dim=1).indices.squeeze().tolist()  # Get top-k indices
top_k_passages = [passages[i] for i in top_k_indices]

# Output the top-k passages
for i, idx in enumerate(top_k_indices):
    print(f"Rank {i+1}: Passage {idx} with score {cosine_scores[0, idx].item()}")
    print(f"Passage: {passages[idx]}")
