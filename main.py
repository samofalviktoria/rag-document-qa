import os
import sys
from dotenv import load_dotenv

load_dotenv()

USAGE = """Usage:
  python main.py index - index new files from the data folder
  python main.py ask   - ask questions about your documents"""


def check_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("Error: GOOGLE_API_KEY not found. Create a .env file (see .env.example).")
    if not key.isascii():
        sys.exit("Error: the key in .env contains invalid characters. Retype it manually.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("index", "ask"):
        print(USAGE)
        return
    check_key()
    if sys.argv[1] == "index":
        from src.vectordb import index_new_files
        index_new_files()
    else:
        from src.qa import chat_loop
        chat_loop()


if __name__ == "__main__":
    main()