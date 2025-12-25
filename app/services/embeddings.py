from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Project root (adjust parents[] if needed)
BASE_DIR = Path(__file__).resolve().parents[2]

# Single FAISS index directory
INDEX_DIR = BASE_DIR / "vector_store" / "faiss"


def store_embeddings(documents):
    """
    Store embeddings in a single shared FAISS index.
    Automatically creates required directories.
    """

    # 1️⃣ Ensure directory exists (safe & idempotent)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    index_file = INDEX_DIR / "index.faiss"

    # 2️⃣ Load or create index
    if index_file.exists():
        store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        store.add_documents(documents)
    else:
        store = FAISS.from_documents(documents, embeddings)

    # 3️⃣ Persist index
    store.save_local(str(INDEX_DIR))
