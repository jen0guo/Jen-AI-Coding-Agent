import os

import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not base_url:
    raise RuntimeError(
        "Please set the API_KEY and BASE_URL environment variables first."
    )

llm_client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# Open the vector database when the application starts.
model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir,
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(
    name="example_product_docs",
    embedding_function=embed_fn,
)


def retrieve(query, top_k=3):
    """Return the top-k relevant documents and their metadata for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    docs = results["documents"][0]
    metadata = results["metadatas"][0]

    return [
        {
            "title": meta["title"],
            "url": meta["url"],
            "text": text,
        }
        for meta, text in zip(metadata, docs)
    ]


def build_system_prompt(retrieved_docs):
    """Add the retrieved excerpts and metadata to the system prompt."""
    blocks = [
        f"[{doc['title']}]({doc['url']})\n{doc['text']}"
        for doc in retrieved_docs
    ]
    context = "\n\n".join(blocks)

    return (
        "You are a customer support assistant for a software product. "
        "Answer the user's question based on the reference materials below.\n"
        "If the reference materials do not contain the answer, clearly tell "
        "the user, \"The documentation does not contain relevant information.\" "
        "Do not make up an answer.\n"
        "At the end of your response, list the titles of the documents you "
        "referenced as Markdown links.\n\n"
        f"Reference materials:\n{context}"
    )


print("Enter a message to start chatting. Enter q to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "q":
        break

    # Retrieve relevant documents before sending the request to the LLM.
    retrieved = retrieve(user_input)
    print(f"  [Retrieval] Matches: {[doc['title'] for doc in retrieved]}")

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(retrieved),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    response = llm_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )

    print(f"AI: {response.choices[0].message.content}\n")