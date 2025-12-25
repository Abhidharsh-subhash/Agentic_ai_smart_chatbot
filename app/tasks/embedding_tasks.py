from celery import shared_task
from pathlib import Path

UPLOAD_DIR = Path("uploads")


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
)
def process_uploaded_files(self, files: list[dict]):
    for file in files:
        file_path = UPLOAD_DIR / file["unique_name"]
        ext = file["extension"]

        if not file_path.exists():
            continue

        # Dispatch by extension
        # if ext == ".pdf":
        #     process_pdf(file_path)
        # elif ext == ".docx":
        #     process_docx(file_path)
        # elif ext in [".xlsx", ".xls"]:
        #     process_excel(file_path)
