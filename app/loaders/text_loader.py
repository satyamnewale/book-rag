from pathlib import Path
from langchain_community.document_loaders import TextLoader

BASE_DIR = Path(__file__).resolve().parents[2]   # book-rag

file_path = BASE_DIR / "books" / "Title The Adventures of Sherlock Holmes.txt"

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

documents = loader.load()
