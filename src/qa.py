import re
import logging
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from src import config
from src.vectordb import get_db

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

NOT_FOUND = "I could not find this information in the documents."

PROMPT = """You are an assistant that answers questions about documents.
Use ONLY the fragments below. If they contain only a partial or indirect answer,
say what exactly was found and what is missing. If the fragments are not related to the question, say only: "{not_found}"
Answer in the language of the question, briefly and precisely.
Write plain text, without markdown and without LaTeX formulas.
If there is an answer, after it write on a new line: SOURCES: and the numbers of the fragments you used, separated by commas.

Fragments:
{context}

Question: {question}"""


def format_sources(sources):
    return ", ".join(f"{n}, p. {p}" if p else n for n, p in sources)


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
    match = re.search(r"SOURCES:(.*)", raw, re.S)
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
        msg = "Rate limit exceeded."
        quota = re.search(r"'quotaId': '([^']+)'", text)
        delay = re.search(r"'retryDelay': '([^']+)'", text)
        if quota:
            msg += f" Limit: {quota.group(1)}."
        if delay:
            msg += f" Try again in {delay.group(1)}."
        return msg
    if "API_KEY_INVALID" in text or "API key not valid" in text or "PERMISSION_DENIED" in text:
        return "The API key is invalid or has no access. Check GOOGLE_API_KEY in the .env file."
    return f"An error occurred: {text[:200]}"


def chat_loop():
    db = get_db()
    if not db.get(limit=1)["ids"]:
        print("The database is empty. Run first: python main.py index")
        return
    llm = ChatGoogleGenerativeAI(model=config.CHAT_MODEL)
    print("Ready. Enter a question (or 'exit').")
    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() == "exit":
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
            print("Sources:", format_sources(sources))
