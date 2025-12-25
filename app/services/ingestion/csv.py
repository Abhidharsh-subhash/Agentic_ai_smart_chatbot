import pandas as pd
from langchain_core.documents import Document
from datetime import datetime
from typing import List, Dict, Any, Optional


def process_csv(
    file_path: str,
    file_metadata: Dict[str, Any],
    text_columns: Optional[List[str]] = None,
    row_format: str = "structured",
) -> List[Document]:
    """
    Process CSV file - each row becomes a separate document (NO chunking).

    Args:
        file_path: Path to CSV file
        file_metadata: Dict containing file_id, admin_id, folder_id, etc.
        text_columns: Specific columns to include (None = all columns)
        row_format: 'structured' (key: value) or 'natural' (sentence-like)

    Returns:
        List of Document objects (one per row)
    """
    try:
        df = pd.read_csv(file_path)

        # Filter columns if specified
        if text_columns:
            available_cols = [col for col in text_columns if col in df.columns]
            if available_cols:
                df = df[available_cols]

        documents = []

        for idx, row in df.iterrows():
            # Convert row to text based on format
            if row_format == "structured":
                text_parts = [
                    f"{col}: {val}"
                    for col, val in row.items()
                    if pd.notna(val) and str(val).strip()
                ]
                text_content = " | ".join(text_parts)
            else:
                text_parts = [
                    f"{col} is {val}"
                    for col, val in row.items()
                    if pd.notna(val) and str(val).strip()
                ]
                text_content = ", ".join(text_parts)

            # Skip empty rows
            if not text_content.strip():
                continue

            # Build column metadata
            column_metadata = {
                f"col_{col}": str(val) for col, val in row.items() if pd.notna(val)
            }

            doc = Document(
                page_content=text_content,
                metadata={
                    "file_id": file_metadata.get("file_id"),
                    "admin_id": file_metadata.get("admin_id"),
                    "folder_id": file_metadata.get("folder_id"),
                    "original_filename": file_metadata.get("original_filename"),
                    "unique_name": file_metadata.get("unique_name"),
                    "file_type": "csv",
                    "row_index": int(idx),
                    "indexed_at": datetime.utcnow().isoformat(),
                    **column_metadata,
                },
            )
            documents.append(doc)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error processing CSV {file_path}: {str(e)}")
