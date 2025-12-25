from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

from .pdf import process_pdf
from .docx import process_docx
from .excel import process_excel
from .csv import process_csv


SUPPORTED_EXTENSIONS = {
    ".pdf": process_pdf,
    ".docx": process_docx,
    ".xlsx": process_excel,
    ".xls": process_excel,
    ".csv": process_csv,
}


def ingest_file(file_path: str, file_metadata: Dict[str, Any]) -> List[Document]:
    """
    Dispatch file processing based on extension.

    Args:
        file_path: Path to the file
        file_metadata: Dict containing file info (file_id, admin_id, etc.)

    Returns:
        List of Document objects ready for embedding

    Raises:
        ValueError: If file type is not supported
    """
    ext = Path(file_path).suffix.lower()

    processor = SUPPORTED_EXTENSIONS.get(ext)
    if not processor:
        raise ValueError(f"Unsupported file type: {ext}")

    return processor(file_path, file_metadata)
