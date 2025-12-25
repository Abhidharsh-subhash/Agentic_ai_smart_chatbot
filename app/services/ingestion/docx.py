from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


def process_docx(file_path: str):
    """
    Extracts text from a DOCX file and returns chunked Documents
    """
    path = Path(file_path)

    loader = Docx2txtLoader(str(path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(documents)
