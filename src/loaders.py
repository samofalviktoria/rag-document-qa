from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config


def load_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix == ".txt":
        docs = TextLoader(str(path), encoding = "utf-8").load()
    elif suffix == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    else:
        raise ValueError(f"Не підтримується розширення файлу {suffix}")
    # Склеюємо слова розділені переносами
    for d in docs:
        d.page_content = d.page_content.replace("-\n", "")
        d.metadata["source"] = path.name
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError("У файлі не знайдено тексту (можливо це скан без розпізнавання)")
    return docs


def split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)