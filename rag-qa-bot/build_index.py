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
documents = [
    {
        "title": "Pro Subscription",
        "url": "https://docs.example.com/billing/pro",
        "text": "The Pro subscription is available in monthly and annual plans. The monthly plan costs ¥38, while the annual plan costs ¥388, making the annual plan approximately 15% cheaper. You can cancel your subscription at any time and retain access until the end of the current billing period.",
    },
    {
        "title": "Free Plan Limits",
        "url": "https://docs.example.com/billing/free-limits",
        "text": "Free users can create up to three projects per month, with each project limited to 100 MB. To access higher limits, users must upgrade to a Pro subscription.",
    },
    {
        "title": "Data Export",
        "url": "https://docs.example.com/data/export",
        "text": "You can export all account data with one click from the Settings → Data page. The export is provided as a compressed JSON archive. A download link will be sent by email within 24 hours of the export request.",
    },
    {
        "title": "Account Deletion",
        "url": "https://docs.example.com/account/deletion",
        "text": "To delete your account, go to Settings → Account and click Permanently Delete Account. Your account data will be retained for 30 days after deletion, during which you can restore the account by logging in. After 30 days, the data will be permanently deleted and cannot be recovered.",
    },
    {
        "title": "Changing Your Email Address",
        "url": "https://docs.example.com/account/email",
        "text": "You can change your email address under Settings → Account → Login Email. The change takes effect only after you click the confirmation links sent to both your current and new email addresses.",
    },
    {
        "title": "Resetting Your Password",
        "url": "https://docs.example.com/account/reset-password",
        "text": "Click Forgot Password on the login page and enter your registered email address to receive a password reset link. The link is valid for 30 minutes. If it expires, you will need to request a new one.",
    },
    {
        "title": "API Rate Limits",
        "url": "https://docs.example.com/api/rate-limit",
        "text": "Free users can make up to 60 API requests per minute, while Pro users can make up to 600 requests per minute. Requests exceeding the limit will receive a 429 status code. Clients are advised to retry using exponential backoff.",
    },
    {
        "title": "Team Collaboration",
        "url": "https://docs.example.com/team/collaboration",
        "text": "Pro users can create team workspaces and invite members to collaborate on editing. Projects within a team workspace are visible to all members. Three permission levels are available: read-only, write, and administrator.",
    },
]

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