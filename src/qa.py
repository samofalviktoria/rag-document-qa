import re
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from src import config
from src.vectordb import get_db

NOT_FOUND = "Не знайшов цієї інформації в документах."

PROMPT = """Ти асистент, який відповідає на запитання по документах.
Використовуй ЛИШЕ фрагменти нижче. Якщо в них є лише часткова або непряма відповідь,
скажи, що саме знайдено, а чого бракує. Якщо фрагменти не стосуються запитання, скажи лише: "{not_found}"
Відповідай мовою запитання, коротко й точно.
Пиши простим текстом, без markdown і без формул LaTeX.
Якщо відповідь є, після неї з нового рядка пиши: ДЖЕРЕЛА: і номери фрагментів, які ти використав, через кому.

Фрагменти:
{context}

Запитання: {question}"""


def format_sources(sources):
    return ", ".join(f"{n}, стор. {p}" if p else n for n, p in sources)


def answer_question(db, llm, question):
    docs = db.similarity_search(question, k=config.TOP_K, filter={"kind": {"$ne": "marker"}})
    context = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    prompt = PROMPT.format(not_found=NOT_FOUND, context=context, question=question)
    raw = llm.invoke(prompt).text
    answer, sources = parse_answer(raw, docs)
    if NOT_FOUND in answer:
        sources = []
    return answer, sources


def parse_answer(raw, docs):
    match = re.search(r"ДЖЕРЕЛА:(.*)", raw, re.S)
    if not match:
        return raw.strip(), []
    answer = raw[:match.start()].strip()
    numbers = {int(n) for n in re.findall(r"\d+", match.group(1))}
    sources = set()
    for n in numbers:
        if 1 <= n <= len(docs):
            d = docs[n - 1]
            name = Path(d.metadata["source"]).name
            page = d.metadata.get("page")
            sources.add((name, page + 1 if page is not None else None))
    return answer, sorted(sources, key=lambda s: (s[0], s[1] or 0))


def friendly_error(e):
    text = str(e)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        msg = "Перевищено ліміт запитів."
        quota = re.search(r"'quotaId': '([^']+)'", text)
        delay = re.search(r"'retryDelay': '([^']+)'", text)
        if quota:
            msg += f" Ліміт: {quota.group(1)}."
        if delay:
            msg += f" Спробуйте через {delay.group(1)}."
        return msg
    if "API_KEY_INVALID" in text or "API key not valid" in text or "PERMISSION_DENIED" in text:
        return "Ключ API недійсний або не має доступу. Перевірте GOOGLE_API_KEY у файлі .env."
    return f"Сталася помилка: {text[:200]}"


def chat_loop():
    db = get_db()
    if not db.get(limit=1)["ids"]:
        print("База порожня. Спочатку виконайте: python main.py index")
        return
    llm = ChatGoogleGenerativeAI(model=config.CHAT_MODEL, temperature=0)
    print("Готово. Введіть запитання (або 'вихід').")
    while True:
        try:
            q = input("\nВаше запитання: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("вихід", "exit"):
            break
        if not q:
            continue
        try:
            answer, sources = answer_question(db, llm, q)
        except Exception as e:
            print(friendly_error(e))
            continue
        print(answer)
        if sources:
            print("Джерела:", format_sources(sources))
