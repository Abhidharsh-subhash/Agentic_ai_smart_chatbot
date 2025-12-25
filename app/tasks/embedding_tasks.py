from typing import List, Dict, Any
import logging
from pathlib import Path
from app.core.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(settings.upload_dir)


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    acks_late=True,
    name="tasks.process_uploaded_files",
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
    # Import inside to avoid circular imports
    from app.services.ingestion.dispatcher import ingest_file
    from app.services.embeddings import store_embeddings

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

            # ✅ Store embeddings (no admin_id needed - shared index)
            doc_count = store_embeddings(documents)

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


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 2},
    name="tasks.process_single_file",
)
def process_single_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single uploaded file and create embeddings."""
    from app.services.ingestion.dispatcher import ingest_file
    from app.services.embeddings import store_embeddings

    file_path = UPLOAD_DIR / file_info["unique_name"]

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Ingest and embed
    documents = ingest_file(str(file_path), file_info)

    if not documents:
        raise ValueError(f"No content extracted from: {file_info['original_filename']}")

    # ✅ Store embeddings (shared index)
    doc_count = store_embeddings(documents)

    return {
        "file_id": file_info["file_id"],
        "filename": file_info["original_filename"],
        "documents_created": doc_count,
        "status": "success",
    }


@celery_app.task(bind=True, name="tasks.delete_file_embeddings")
def delete_file_embeddings(self, file_id: str) -> Dict[str, Any]:
    """
    Delete embeddings for a specific file.
    Called when a file is deleted from the system.
    """
    from app.services.embeddings import delete_file_embeddings

    # ✅ No admin_id needed - delete by file_id only
    success = delete_file_embeddings(file_id)

    return {
        "file_id": file_id,
        "deleted": success,
    }
