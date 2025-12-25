import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


def process_excel(file_path: str):
    """
    Converts Excel rows into semantic documents.
    Each row becomes a searchable text chunk.
    """
    path = Path(file_path)
    documents = []

    # Load all sheets
    excel_file = pd.ExcelFile(path)

    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name).fillna("")

        for index, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in df.columns)

            doc = Document(
                page_content=content,
                metadata={
                    "sheet": sheet_name,
                    "row_index": index,
                    "source": path.name,
                },
            )
            documents.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)
