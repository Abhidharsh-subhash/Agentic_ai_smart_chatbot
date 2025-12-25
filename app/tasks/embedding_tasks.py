from typing import List, Dict, Any
import logging
from pathlib import Path
from app.core.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(settings.upload_dir)


# =====================
# FILE PROCESSING TASKS
# =====================


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    acks_late=True,
    name="tasks.process_uploaded_files",
)
def process_uploaded_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process multiple uploaded files and create embeddings."""
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

            documents = ingest_file(str(file_path), file_info)

            if not documents:
                results["failed"].append(
                    {
                        "file_id": file_info["file_id"],
                        "filename": file_info["original_filename"],
                        "error": "No content extracted",
                    }
                )
                continue

            doc_count = store_embeddings(documents)

            logger.info(
                f"✅ Processed {file_info['original_filename']}: {doc_count} documents"
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


# =====================
# FILE DELETION TASKS
# =====================


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 3},
    acks_late=True,
    name="tasks.delete_files_embeddings",
)
def delete_files_embeddings_task(
    self,
    file_ids: List[str],
) -> Dict[str, Any]:
    """
    Delete embeddings for multiple files from vector DB.

    Args:
        file_ids: List of file IDs to delete embeddings for

    Returns:
        Dict with deletion results
    """
    from app.services.embeddings import delete_multiple_files_embeddings

    result = {
        "success": False,
        "deleted_count": 0,
        "file_ids_found": [],
        "file_ids_not_found": [],
        "error": None,
    }

    try:
        deletion_result = delete_multiple_files_embeddings(file_ids)

        result["success"] = True
        result["deleted_count"] = deletion_result.get("deleted_count", 0)
        result["file_ids_found"] = deletion_result.get("file_ids_found", [])
        result["file_ids_not_found"] = deletion_result.get("file_ids_not_found", [])

        logger.info(
            f"✅ Deleted {result['deleted_count']} embeddings "
            f"for {len(result['file_ids_found'])} file(s)"
        )

    except Exception as e:
        logger.error(f"❌ Error deleting embeddings: {str(e)}")
        result["error"] = str(e)

    return result
