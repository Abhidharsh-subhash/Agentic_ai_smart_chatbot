import pandas as pd
from langchain_core.documents import Document
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


def process_excel(
    file_path: str,
    file_metadata: Dict[str, Any],
    sheet_name: Optional[Union[str, int, List]] = None,
    text_columns: Optional[List[str]] = None,
    row_format: str = "structured",
) -> List[Document]:
    """
    Process Excel file - each row becomes a separate document (NO chunking).

    This ensures each Excel record is treated as an atomic unit for better
    retrieval accuracy.

    Args:
        file_path: Path to Excel file
        file_metadata: Dict containing file_id, admin_id, folder_id, etc.
        sheet_name: Specific sheet(s) to process (None = all sheets)
        text_columns: Specific columns to include (None = all columns)
        row_format: 'structured' (key: value) or 'natural' (sentence-like)

    Returns:
        List of Document objects (one per row)
    """
    try:
        path = Path(file_path)
        excel_file = pd.ExcelFile(path)

        # Determine sheets to process
        if sheet_name is None:
            sheet_names = excel_file.sheet_names
        else:
            sheet_names = (
                [sheet_name] if isinstance(sheet_name, (str, int)) else sheet_name
            )

        documents = []

        for sheet in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)

            # Filter columns if specified
            if text_columns:
                available_cols = [col for col in text_columns if col in df.columns]
                if available_cols:
                    df = df[available_cols]

            for idx, row in df.iterrows():
                # Convert row to text based on format
                if row_format == "structured":
                    # Format: "Column1: Value1 | Column2: Value2 | ..."
                    text_parts = [
                        f"{col}: {val}"
                        for col, val in row.items()
                        if pd.notna(val) and str(val).strip()
                    ]
                    text_content = " | ".join(text_parts)
                else:
                    # Natural format: "The Column1 is Value1, Column2 is Value2..."
                    text_parts = [
                        f"{col} is {val}"
                        for col, val in row.items()
                        if pd.notna(val) and str(val).strip()
                    ]
                    text_content = ", ".join(text_parts)

                # Skip empty rows
                if not text_content.strip():
                    continue

                # Build column metadata (store original values for filtering)
                column_metadata = {
                    f"col_{col}": str(val) for col, val in row.items() if pd.notna(val)
                }

                # Create document with rich metadata
                doc = Document(
                    page_content=text_content,
                    metadata={
                        # File identification
                        "file_id": file_metadata.get("file_id"),
                        "admin_id": file_metadata.get("admin_id"),
                        "folder_id": file_metadata.get("folder_id"),
                        "original_filename": file_metadata.get("original_filename"),
                        "unique_name": file_metadata.get("unique_name"),
                        # Excel-specific metadata
                        "file_type": "excel",
                        "sheet_name": str(sheet),
                        "row_index": int(idx),
                        "indexed_at": datetime.utcnow().isoformat(),
                        # Column values as metadata
                        **column_metadata,
                    },
                )
                documents.append(doc)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error processing Excel {file_path}: {str(e)}")
