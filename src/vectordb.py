import time
import hashlib
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src import config
from src.loaders import load_file, split_documents


def get_db():
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(persist_directory=str(config.DB_DIR), embedding_function=embeddings)


def file_hash(path):
    """Відбиток вмісту файлу: зміниться хоч на один байт, зміниться і хеш."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed_files(db):
    """Функція повертає словник {ім'я файлу: хеш}"""
    data = db.get(where={"kind": "marker"}, include=["metadatas"])
    return {m["source"]: m["hash"] for m in data["metadatas"]}


def delete_file_records(db, name):
    """Видаляє всі старі записи з файлу"""
    res = db.get(where={"source": name})
    if res["ids"]:
        db.delete(ids=res["ids"])


def add_marker(db, name, hash_value):
    marker = Document(
        page_content=name,
        metadata={"kind": "marker", "source": name, "hash": hash_value}
    )
    for attempt in range(5):
        try:
            db.add_documents([marker], ids=[f"indexed:{name}"])
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f" Ліміт запитів, чекаю {config.PAUSE_SECONDS} с...")
                time.sleep(config.PAUSE_SECONDS)
            else:
                raise
    else:
        raise RuntimeError("Не вдалося записати маркер після 5 спроб")   


def add_chunks(db, chunks):
    """Додає чанки порціями, чекає і повторює спробу при ліміті запитів (429)."""
    batch_size = config.BATCH_SIZE
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [f"{c.metadata['source']}:{n}" for n, c in enumerate(batch, start=i)]
        for attempt in range(5):
            try:
                db.add_documents(batch, ids=ids)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  Ліміт запитів, чекаю {config.PAUSE_SECONDS} с...")
                    time.sleep(config.PAUSE_SECONDS)
                else:
                    raise
        else:
            raise RuntimeError("не вдалося додати порцію після 5 спроб")
        print(f"  Додано {min(i + batch_size, len(chunks))}/{len(chunks)}")
        if i + batch_size < len(chunks):
            time.sleep(config.PAUSE_SECONDS)


def index_new_files():
    config.DATA_DIR.mkdir(exist_ok=True)
    files = sorted(
        p for p in config.DATA_DIR.iterdir()
        if p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"У папці {config.DATA_DIR} немає файлів PDF або TXT.")
        return

    db = get_db()
    done = indexed_files(db)
    disk_names = {f.name for f in files}
    orphans = set(done) - disk_names
    for name in orphans:
        print(f"Видаляю осиротілі записи файлу {name}")
        delete_file_records(db, name)
    hashes = {f: file_hash(f) for f in files}
    new_files = [f for f in files if done.get(f.name) != hashes[f]]
    if not new_files:
        print("Нових або змінених файлів немає, уже все проіндексовано.")
        return
    for f in new_files:
        print(f"Обробляю {f.name}...")
        try:
            docs = load_file(f)
        except Exception as e:
            print(f"  Пропускаю {f.name}: {e}")
            continue
        chunks = split_documents(docs)
        for c in chunks:
            c.metadata["kind"] = "chunk"
        print(f"  Сторінок: {len(docs)}, чанків: {len(chunks)}")
        delete_file_records(db, f.name)
        add_chunks(db, chunks)
        add_marker(db, f.name, hashes[f])
    print("Готово.")