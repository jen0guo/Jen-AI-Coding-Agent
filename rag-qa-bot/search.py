import json
import sys

import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 2:
    print("Usage: python search.py <query>")
    sys.exit(1)

query = sys.argv[1]

# Load the previously saved index, including the embeddings and original documents.
doc_embeddings = np.load("doc_embeddings.npy")

with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load the same embedding model used to build the index.
# Both operations must use the same model; otherwise, the embedding
# dimensions or vector spaces may not match.
model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
model = SentenceTransformer(model_dir)

# Encode the user's query as a semantic vector.
query_embedding = model.encode([query], prompt_name="query")
similarities = model.similarity(query_embedding, doc_embeddings).numpy()[0]

# Retrieve the three documents with the highest similarity scores.
top3 = np.argsort(-similarities)[:3]

print(f"\nQuery: {query}")

for rank, idx in enumerate(top3, 1):
    doc = documents[idx]
    score = similarities[idx]

    print(f"  Top {rank} | Similarity: {score:.3f} | Index: {idx}")
    print(f"    Title: {doc['title']}")
    print(f"    URL: {doc['url']}")
    print(f"    Excerpt: {doc['text']}")