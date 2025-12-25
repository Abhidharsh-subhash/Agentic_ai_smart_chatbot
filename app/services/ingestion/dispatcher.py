from pathlib import Path
from .pdf import process_pdf
from .docx import process_docx
from .excel import process_excel
from app.services.embeddings import store_embeddings


def ingest_file(file_path: str, admin_id: str):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        docs = process_pdf(file_path)
    elif ext == ".docx":
        docs = process_docx(file_path)
    elif ext in [".xlsx", ".xls"]:
        docs = process_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    store_embeddings(docs, admin_id)
