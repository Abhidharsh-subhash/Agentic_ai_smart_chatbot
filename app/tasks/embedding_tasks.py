from celery import shared_task
from typing import List, Dict, Any
import logging
from pathlib import Path
from app.core.config import settings
from app.services.ingestion.dispatcher import ingest_file
from app.services.embeddings import store_embeddings

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(settings.upload_dir)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    acks_late=True,
)
def process_uploaded_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple uploaded files and create embeddings.

    Args:
        files: List of file metadata dicts containing:
            - file_id: UUID of the file record
            - unique_name: Unique filename on disk
            - original_filename: Original uploaded filename
            - extension: File extension (e.g., '.pdf')
            - admin_id: Admin who uploaded the file
            - folder_id: Folder containing the file

    Returns:
        Dict with processing results
    """
    results = {
        "processed": [],
        "failed": [],
        "total_documents": 0,
    }

    for file_info in files:
        file_path = UPLOAD_DIR / file_info["unique_name"]

        try:
            # Validate file exists
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                results["failed"].append(
                    {
                        "file_id": file_info["file_id"],
                        "filename": file_info["original_filename"],
                        "error": "File not found on disk",
                    }
                )
                continue

            logger.info(f"Processing: {file_info['original_filename']}")

            # Ingest file (extract text and create Documents)
            documents = ingest_file(str(file_path), file_info)

            if not documents:
                logger.warning(
                    f"No content extracted from: {file_info['original_filename']}"
                )
                results["failed"].append(
                    {
                        "file_id": file_info["file_id"],
                        "filename": file_info["original_filename"],
                        "error": "No content extracted",
                    }
                )
                continue

            # Store embeddings
            doc_count = store_embeddings(documents, file_info["admin_id"])

            logger.info(
                f"✅ Processed {file_info['original_filename']}: "
                f"{doc_count} documents embedded"
            )

            results["processed"].append(
                {
                    "file_id": file_info["file_id"],
                    "filename": file_info["original_filename"],
                    "documents_created": doc_count,
                }
            )
            results["total_documents"] += doc_count

        except Exception as e:
            logger.error(
                f"❌ Error processing {file_info['original_filename']}: {str(e)}"
            )
            results["failed"].append(
                {
                    "file_id": file_info["file_id"],
                    "filename": file_info["original_filename"],
                    "error": str(e),
                }
            )

    return results


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 2},
)
def process_single_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single uploaded file and create embeddings.
    Useful for immediate processing without batching.
    """
    file_path = UPLOAD_DIR / file_info["unique_name"]

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Ingest and embed
    documents = ingest_file(str(file_path), file_info)

    if not documents:
        raise ValueError(f"No content extracted from: {file_info['original_filename']}")

    doc_count = store_embeddings(documents, file_info["admin_id"])

    return {
        "file_id": file_info["file_id"],
        "filename": file_info["original_filename"],
        "documents_created": doc_count,
        "status": "success",
    }


@shared_task(bind=True)
def delete_file_embeddings(self, file_id: str, admin_id: str) -> Dict[str, Any]:
    """
    Delete embeddings for a specific file.
    Called when a file is deleted from the system.
    """
    from app.services.embeddings import embedding_service

    success = embedding_service.delete_by_file_id(file_id, admin_id)

    return {
        "file_id": file_id,
        "deleted": success,
    }
