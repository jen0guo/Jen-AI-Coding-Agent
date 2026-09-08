import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download

# Reuse the previously downloaded Qwen3-Embedding model weights.
model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir,
)

# Create a locally persisted vector database at ./chroma_db.
client = chromadb.PersistentClient(path="./chroma_db")

# Chroma uses L2 distance by default. Configure the collection to use
# cosine distance so it remains consistent with the previous implementation.
collection = client.get_or_create_collection(
    name="example_product_docs",
    embedding_function=embed_fn,
    configuration={"hnsw": {"space": "cosine"}},
)

# The same eight documentation excerpts used previously.
from documents import documents

# Insert the data in batches. The documents, metadata, and IDs correspond
# to one another by position. Chroma automatically uses embed_fn to encode
# the document text as vectors.
collection.add(
    documents=[doc["text"] for doc in documents],
    metadatas=[
        {"title": doc["title"], "url": doc["url"]}
        for doc in documents
    ],
    ids=[f"doc-{i}" for i in range(len(documents))],
)

print(f"Insertion complete. Collection size: {collection.count()}")