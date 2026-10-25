import os
import sys
from dotenv import load_dotenv

load_dotenv()

USAGE = """Використання:
  python main.py index - проіндексувати нові файли з папки data
  python main.py ask   - ставити запитання по документах"""


def check_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("Помилка: не знайдено GOOGLE_API_KEY. Створіть файл .env (див. .env. example).")
    if not key.isascii():
        sys.exit("Помилка: ключ у .env містить недопустимі символи. Перепишіть його вручну.")


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