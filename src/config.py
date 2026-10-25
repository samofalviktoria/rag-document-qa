from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_pdf"
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

BATCH_SIZE = 50
PAUSE_SECONDS = 65

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")