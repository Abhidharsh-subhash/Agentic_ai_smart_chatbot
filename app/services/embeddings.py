# app/services/embeddings.py
from pathlib import Path
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import threading
from app.core.config import settings

# Thread lock for FAISS operations (FAISS is not thread-safe)
_faiss_lock = threading.Lock()


class EmbeddingService:
    """Service for managing document embeddings with FAISS."""

    def __init__(self):
        self.index_dir = Path(settings.vector_store_dir)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    def _get_index_path(self, admin_id: str) -> Path:
        """Get admin-specific index path."""
        # Option 1: Shared index for all admins
        # return self.index_dir

        # Option 2: Separate index per admin (recommended for multi-tenancy)
        return self.index_dir / admin_id

    def store_documents(
        self,
        documents: List[Document],
        admin_id: str,
    ) -> int:
        """
        Store document embeddings in FAISS.
        Creates new index if none exists, otherwise adds to existing.

        Args:
            documents: List of Document objects to embed
            admin_id: Admin ID for index isolation

        Returns:
            Number of documents stored
        """
        if not documents:
            return 0

        index_path = self._get_index_path(admin_id)
        index_path.mkdir(parents=True, exist_ok=True)

        index_file = index_path / "index.faiss"

        with _faiss_lock:
            if index_file.exists():
                # Load existing and add new documents
                vector_store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                vector_store.add_documents(documents)
            else:
                # Create new index
                vector_store = FAISS.from_documents(documents, self.embeddings)

            # Persist
            vector_store.save_local(str(index_path))

        return len(documents)

    def search(
        self,
        query: str,
        admin_id: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            admin_id: Admin ID for index isolation
            k: Number of results
            filter_dict: Optional metadata filters

        Returns:
            List of matching Documents
        """
        index_path = self._get_index_path(admin_id)
        index_file = index_path / "index.faiss"

        if not index_file.exists():
            return []

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(index_path),
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

    def delete_by_file_id(self, file_id: str, admin_id: str) -> bool:
        """
        Delete all embeddings for a specific file.

        Note: FAISS doesn't support deletion directly.
        This requires rebuilding the index without the deleted documents.
        """
        index_path = self._get_index_path(admin_id)
        index_file = index_path / "index.faiss"

        if not index_file.exists():
            return False

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Get all documents
            # Note: This is a workaround since FAISS doesn't support deletion
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
                new_store.save_local(str(index_path))
            else:
                # No documents left, remove index files
                import shutil

                shutil.rmtree(index_path)

        return True


# Singleton instance
embedding_service = EmbeddingService()


def store_embeddings(documents: List[Document], admin_id: str) -> int:
    """Convenience function to store embeddings."""
    return embedding_service.store_documents(documents, admin_id)


def search_embeddings(
    query: str,
    admin_id: str,
    k: int = 5,
    filter_dict: Optional[dict] = None,
) -> List[Document]:
    """Convenience function to search embeddings."""
    return embedding_service.search(query, admin_id, k, filter_dict)
