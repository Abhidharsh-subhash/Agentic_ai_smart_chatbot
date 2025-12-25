from pathlib import Path
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import threading
from app.core.config import settings
from pathlib import Path

# Thread lock for FAISS operations (FAISS is not thread-safe)
_faiss_lock = threading.Lock()


class EmbeddingService:
    """Service for managing document embeddings with shared FAISS index."""

    def __init__(self):
        self.index_dir = Path(settings.vector_store_dir)  # Single shared directory
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        # Ensure directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_file(self) -> Path:
        """Get the index file path."""
        return self.index_dir / "index.faiss"

    def store_documents(self, documents: List[Document]) -> int:
        """
        Store document embeddings in shared FAISS index.
        Creates new index if none exists, otherwise adds to existing.

        Args:
            documents: List of Document objects to embed

        Returns:
            Number of documents stored
        """
        if not documents:
            return 0

        with _faiss_lock:
            if self.index_file.exists():
                # Load existing and add new documents
                vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                vector_store.add_documents(documents)
            else:
                # Create new index
                vector_store = FAISS.from_documents(documents, self.embeddings)

            # Persist
            vector_store.save_local(str(self.index_dir))

        return len(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results
            filter_dict: Optional metadata filters (e.g., {"admin_id": "xxx"})

        Returns:
            List of matching Documents
        """
        if not self.index_file.exists():
            return []

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            if filter_dict:
                results = vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict,
                )
            else:
                results = vector_store.similarity_search(query, k=k)

        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        if not self.index_file.exists():
            return []

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            results = vector_store.similarity_search_with_score(query, k=k)

        return results

    def delete_by_file_id(self, file_id: str) -> bool:
        """
        Delete all embeddings for a specific file.

        Note: FAISS doesn't support deletion directly.
        This requires rebuilding the index without the deleted documents.
        """
        if not self.index_file.exists():
            return False

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Get all documents (workaround for FAISS)
            all_docs = vector_store.similarity_search("", k=100000)

            # Filter out documents with matching file_id
            remaining_docs = [
                doc for doc in all_docs if doc.metadata.get("file_id") != file_id
            ]

            if len(remaining_docs) == len(all_docs):
                return False  # Nothing was deleted

            if remaining_docs:
                # Rebuild index without deleted documents
                new_store = FAISS.from_documents(remaining_docs, self.embeddings)
                new_store.save_local(str(self.index_dir))
            else:
                # No documents left, remove index files
                import shutil

                shutil.rmtree(self.index_dir)
                self.index_dir.mkdir(parents=True, exist_ok=True)

        return True

    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        if not self.index_file.exists():
            return {"exists": False, "total_documents": 0}

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Get sample to count
            all_docs = vector_store.similarity_search("", k=100000)

            # Collect unique files
            files = set()
            admins = set()
            for doc in all_docs:
                files.add(doc.metadata.get("file_id", "unknown"))
                admins.add(doc.metadata.get("admin_id", "unknown"))

        return {
            "exists": True,
            "total_documents": len(all_docs),
            "unique_files": len(files),
            "unique_admins": len(admins),
        }


# Singleton instance
embedding_service = EmbeddingService()


# =====================
# Convenience Functions
# =====================


def store_embeddings(documents: List[Document]) -> int:
    """Store embeddings in shared index."""
    return embedding_service.store_documents(documents)


def search_embeddings(
    query: str,
    k: int = 5,
    filter_dict: Optional[dict] = None,
) -> List[Document]:
    """Search embeddings in shared index."""
    return embedding_service.search(query, k, filter_dict)


def search_embeddings_with_scores(
    query: str,
    k: int = 5,
) -> List[tuple[Document, float]]:
    """Search embeddings with relevance scores."""
    return embedding_service.search_with_scores(query, k)


def delete_file_embeddings(file_id: str) -> bool:
    """Delete embeddings for a specific file."""
    return embedding_service.delete_by_file_id(file_id)
