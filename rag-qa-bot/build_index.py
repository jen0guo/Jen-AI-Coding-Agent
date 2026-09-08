import json

import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

# On the first run, approximately 1.1 GB of model weights will be downloaded
# from ModelScope and cached in ~/.cache/modelscope/.
# On subsequent runs, snapshot_download will return the cached model path
# without downloading the model again.
model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
model = SentenceTransformer(model_dir)

# Simulated documentation excerpts for a SaaS product
from documents import documents
# Encode only the text field of each document.
# Metadata is not included in the embedding calculation.
texts = [doc["text"] for doc in documents]

# encode returns a NumPy array by default, which can be passed directly to np.save.
doc_embeddings = model.encode(texts)
print(
    f"Number of documents: {len(documents)}, "
    f"embedding shape: {doc_embeddings.shape}"
)

# Save the embeddings together with the original documents and their metadata.
# The vector at row i of doc_embeddings must correspond exactly to the
# document dictionary at index i of documents.
np.save("doc_embeddings.npy", doc_embeddings)

with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print("Index saved to doc_embeddings.npy and documents.json")