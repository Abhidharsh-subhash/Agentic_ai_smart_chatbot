from pathlib import Path
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import threading
import shutil
from app.core.config import settings

# Thread lock for FAISS operations
_faiss_lock = threading.Lock()


class EmbeddingService:
    """Service for managing document embeddings with shared FAISS index."""

    def __init__(self):
        self.index_dir = Path(settings.vector_store_dir)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_file(self) -> Path:
        """Get the index file path."""
        return self.index_dir / "index.faiss"

    def store_documents(self, documents: List[Document]) -> int:
        """Store document embeddings in shared FAISS index."""
        if not documents:
            return 0

        with _faiss_lock:
            if self.index_file.exists():
                vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                vector_store.add_documents(documents)
            else:
                vector_store = FAISS.from_documents(documents, self.embeddings)

            vector_store.save_local(str(self.index_dir))

        return len(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> List[Document]:
        """Search for similar documents."""
        if not self.index_file.exists():
            return []

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            if filter_dict:
                results = vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                results = vector_store.similarity_search(query, k=k)

        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple[Document, float]]:
        """Search with relevance scores."""
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

        Note: FAISS doesn't support direct deletion.
        This rebuilds the index without the deleted documents.
        """
        if not self.index_file.exists():
            return False

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Get all documents
            all_docs = vector_store.similarity_search("", k=100000)

            # Filter out documents with matching file_id
            remaining_docs = [
                doc for doc in all_docs if doc.metadata.get("file_id") != file_id
            ]

            # Check if anything was deleted
            if len(remaining_docs) == len(all_docs):
                return False

            if remaining_docs:
                # Rebuild index without deleted documents
                new_store = FAISS.from_documents(remaining_docs, self.embeddings)
                new_store.save_local(str(self.index_dir))
            else:
                # No documents left, remove index
                shutil.rmtree(self.index_dir)
                self.index_dir.mkdir(parents=True, exist_ok=True)

        return True

    def delete_by_file_ids(self, file_ids: List[str]) -> dict:
        """
        Delete embeddings for multiple files at once.
        More efficient than calling delete_by_file_id multiple times.

        Args:
            file_ids: List of file IDs to delete

        Returns:
            Dict with deletion statistics
        """
        if not self.index_file.exists():
            return {
                "deleted_count": 0,
                "remaining_count": 0,
                "file_ids_found": [],
            }

        file_ids_set = set(file_ids)

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Get all documents
            all_docs = vector_store.similarity_search("", k=100000)
            original_count = len(all_docs)

            # Track which file_ids were found
            found_file_ids = set()

            # Filter out documents with matching file_ids
            remaining_docs = []
            for doc in all_docs:
                doc_file_id = doc.metadata.get("file_id")
                if doc_file_id in file_ids_set:
                    found_file_ids.add(doc_file_id)
                else:
                    remaining_docs.append(doc)

            deleted_count = original_count - len(remaining_docs)

            if deleted_count > 0:
                if remaining_docs:
                    # Rebuild index without deleted documents
                    new_store = FAISS.from_documents(remaining_docs, self.embeddings)
                    new_store.save_local(str(self.index_dir))
                else:
                    # No documents left, remove index
                    shutil.rmtree(self.index_dir)
                    self.index_dir.mkdir(parents=True, exist_ok=True)

        return {
            "deleted_count": deleted_count,
            "remaining_count": len(remaining_docs),
            "file_ids_found": list(found_file_ids),
            "file_ids_not_found": list(file_ids_set - found_file_ids),
        }

    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        if not self.index_file.exists():
            return {
                "exists": False,
                "total_documents": 0,
                "unique_files": 0,
                "unique_admins": 0,
            }

        with _faiss_lock:
            vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            all_docs = vector_store.similarity_search("", k=100000)

            files = set()
            admins = set()
            file_types = {}

            for doc in all_docs:
                files.add(doc.metadata.get("file_id", "unknown"))
                admins.add(doc.metadata.get("admin_id", "unknown"))

                ft = doc.metadata.get("file_type", "unknown")
                file_types[ft] = file_types.get(ft, 0) + 1

        return {
            "exists": True,
            "total_documents": len(all_docs),
            "unique_files": len(files),
            "unique_admins": len(admins),
            "documents_by_type": file_types,
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


def delete_files_embeddings(file_id: str) -> bool:
    """Delete embeddings for a single file."""
    return embedding_service.delete_by_file_id(file_id)


def delete_multiple_files_embeddings(file_ids: List[str]) -> dict:
    """Delete embeddings for multiple files at once."""
    return embedding_service.delete_by_file_ids(file_ids)
