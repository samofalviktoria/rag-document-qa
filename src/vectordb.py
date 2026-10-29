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
    """Fingerprint of the file content: change a single byte and the hash changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed_files(db):
    """Returns a dict {file name: hash}."""
    data = db.get(where={"kind": "marker"}, include=["metadatas"])
    return {m["source"]: m["hash"] for m in data["metadatas"]}


def delete_file_records(db, name):
    """Deletes all old records of the file."""
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
                print(f" Rate limit reached, waiting {config.PAUSE_SECONDS} s...")
                time.sleep(config.PAUSE_SECONDS)
            else:
                raise
    else:
        raise RuntimeError("Failed to write the marker after 5 attempts") 


def add_chunks(db, chunks):
    """Adds chunks in batches; waits and retries when the rate limit (429) is hit."""
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
                    print(f"  Rate limit reached, waiting {config.PAUSE_SECONDS} s...")
                    time.sleep(config.PAUSE_SECONDS)
                else:
                    raise
        else:
            raise RuntimeError("Failed to add the batch after 5 attempts")
        print(f"  Added {min(i + batch_size, len(chunks))}/{len(chunks)}")
        if i + batch_size < len(chunks):
            time.sleep(config.PAUSE_SECONDS)


def index_new_files():
    config.DATA_DIR.mkdir(exist_ok=True)
    files = sorted(
        p for p in config.DATA_DIR.iterdir()
        if p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"No PDF, TXT or DOCX files found in {config.DATA_DIR}.")
        return

    db = get_db()
    done = indexed_files(db)
    disk_names = {f.name for f in files}
    orphans = set(done) - disk_names
    for name in orphans:
        print(f"Removing orphaned records for {name}")
        delete_file_records(db, name)
    hashes = {f: file_hash(f) for f in files}
    new_files = [f for f in files if done.get(f.name) != hashes[f]]
    if not new_files:
        print("No new or changed files. Everything is already indexed.")
        return
    for f in new_files:
        print(f"Processing {f.name}...")
        try:
            docs = load_file(f)
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
            continue
        chunks = split_documents(docs)
        for c in chunks:
            c.metadata["kind"] = "chunk"
        print(f"  Documents: {len(docs)}, chunks: {len(chunks)}")
        delete_file_records(db, f.name)
        add_chunks(db, chunks)
        add_marker(db, f.name, hashes[f])
    print("Done.")