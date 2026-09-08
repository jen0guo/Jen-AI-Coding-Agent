# search.py
import sys

import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download

if len(sys.argv) < 2:
    print("Usage: python search.py <query>")
    sys.exit(1)

query = sys.argv[1]

# Load the same embedding model used to encode the documents.
model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir,
)

# Open the previously created index using the same database path
# and collection name.
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(
    name="example_product_docs",
    embedding_function=embed_fn,
)

# Search in one step: pass the query text, automatically encode it as a
# vector, and return the top-k most similar documents.
results = collection.query(
    query_texts=[query],
    n_results=3,
)

# Extract the results for this query.
docs = results["documents"][0]
metas = results["metadatas"][0]
dists = results["distances"][0]

print(f"\nQuery: {query}")

for rank, (text, metadata, distance) in enumerate(
    zip(docs, metas, dists),
    start=1,
):
    # Cosine distance = 1 - cosine similarity.
    similarity = 1 - distance

    print(f"  Top {rank} | Similarity: {similarity:.3f}")
    print(f"    Title: {metadata['title']}")
    print(f"    URL: {metadata['url']}")
    print(f"    Excerpt: {text}")