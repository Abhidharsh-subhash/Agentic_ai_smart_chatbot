from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime
from typing import List, Dict, Any
from app.core.config import settings


def process_pdf(
    file_path: str,
    file_metadata: Dict[str, Any],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Process PDF file and return chunked Documents with metadata.

    Args:
        file_path: Path to PDF file
        file_metadata: Dict containing file_id, admin_id, folder_id, etc.
        chunk_size: Size of text chunks (defaults to settings)
        chunk_overlap: Overlap between chunks (defaults to settings)

    Returns:
        List of Document objects with enriched metadata
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            return []

        # Enrich metadata for each page
        for doc in documents:
            doc.metadata.update(
                {
                    "file_id": file_metadata.get("file_id"),
                    "admin_id": file_metadata.get("admin_id"),
                    "folder_id": file_metadata.get("folder_id"),
                    "original_filename": file_metadata.get("original_filename"),
                    "unique_name": file_metadata.get("unique_name"),
                    "file_type": "pdf",
                    "indexed_at": datetime.utcnow().isoformat(),
                }
            )

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        # Add chunk index to metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    except Exception as e:
        raise RuntimeError(f"Error processing PDF {file_path}: {str(e)}")
