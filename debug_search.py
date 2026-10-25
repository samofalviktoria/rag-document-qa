import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src import config
from src.vectordb import get_db

question = " ".join(sys.argv[1:]) or "Коли була прийнята Конституція України?"
db = get_db()
print("Запитання:", question)
for i, d in enumerate(db.similarity_search(question, k=config.TOP_K), 1):
    name = Path(d.metadata["source"]).name
    page = d.metadata.get("page")
    page = page + 1 if page is not None else "-"
    text = d.page_content[:300].replace("\n", " ")
    print(f"\n[{i}] {name}, стор. {page}\n{text}")