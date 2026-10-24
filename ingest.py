import re
import random
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

from rag import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    get_embedding_model,
)

ROOT_DIR = Path(__file__).resolve().parent
PDF_PATH = ROOT_DIR / "data" / "Constitution_of_Kenya_2010.pdf"
TARGET_WORDS = 350
MAX_WORDS = 500
OVERLAP_WORDS = 50
BATCH_SIZE = 32

ARTICLE_RE = re.compile(
    r"^\s*(?:ARTICLE\s+)?(?P<number>\d{1,3})\s*[.\u2010-\u2015\u2212-]\s*"
    r"(?P<title>\S.*)\s*$",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(
    r"^\s*CHAPTER\s+(?P<number>[A-Z]+|\d+|[IVXLCDM]+)"
    r"(?:\s*[.\u2010-\u2015\u2212-]\s*(?P<title>.*))?\s*$",
    re.IGNORECASE,
)
SCHEDULE_START_RE = re.compile(
    r"^\s*(?:THE\s+)?(?:FIRST|1(?:ST)?)\s+SCHEDULE\b",
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(r"^\s*\((?P<number>\d{1,2})\)\s*(?P<text>.*)$")
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*$", re.IGNORECASE)
TOC_LEADER_RE = re.compile(r"(?:\.{4,}|…{2,})")


class ArticleChunk(TypedDict):
    id: str
    text: str
    article: str
    title: str
    chapter: str
    clause: str


def _join_wrapped_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        if not current:
            current = line
        elif current.endswith("-") and line[:1].islower():
            current = current[:-1] + line
        elif current[-1:] not in ".!?;:" and line[:1].islower():
            current += " " + line
        else:
            paragraphs.append(current)
            current = line
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


def clean_page_texts(page_texts: list[str]) -> str:
    normalized_pages = [
        [re.sub(r"\s+", " ", line).strip() for line in page.splitlines()]
        for page in page_texts
    ]
    repeated_lines = Counter(
        line.casefold()
        for page in normalized_pages
        for line in set(page)
        if line and len(line) < 120
    )
    repeat_threshold = max(3, (len(page_texts) + 1) // 2)
    repeated_headers = {
        line for line, count in repeated_lines.items() if count >= repeat_threshold
    }

    cleaned_pages = []
    for page in normalized_pages:
        lines = [
            line
            for line in page
            if line
            and not PAGE_NUMBER_RE.fullmatch(line)
            and line.casefold() not in repeated_headers
        ]
        cleaned_pages.append(_join_wrapped_lines(lines))
    return "\n".join(cleaned_pages)


def extract_pdf_text(pdf_path: Path = PDF_PATH) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return clean_page_texts([page.extract_text() or "" for page in reader.pages])


def _clause_groups(body_lines: list[str]) -> list[tuple[str, str]]:
    groups: list[tuple[str, list[str]]] = []
    current_clause = ""
    current_lines: list[str] = []
    for line in body_lines:
        clause_match = CLAUSE_RE.match(line)
        if clause_match:
            if current_lines:
                groups.append((current_clause, current_lines))
            current_clause = clause_match.group("number")
            text = clause_match.group("text").strip()
            current_lines = [text] if text else []
        elif current_lines:
            current_lines.append(line)
        elif line.strip():
            current_lines = [line]
    if current_lines:
        groups.append((current_clause, current_lines))
    return [
        (clause, "\n".join(lines).strip())
        for clause, lines in groups
        if "\n".join(lines).strip()
    ]


def _chunk_article(
    number: str,
    title: str,
    chapter: str,
    body_lines: list[str],
) -> list[ArticleChunk]:
    heading = f"Article {number}. {title}".strip()
    body = "\n".join(line for line in body_lines if line.strip()).strip()
    if not body:
        return [
            {
                "id": f"article-{number}-chunk-1",
                "text": heading,
                "article": number,
                "title": title,
                "chapter": chapter,
                "clause": "",
            }
        ]

    body_words = len(body.split())
    if body_words <= MAX_WORDS:
        return [
            {
                "id": f"article-{number}-chunk-1",
                "text": f"{heading}\n{body}",
                "article": number,
                "title": title,
                "chapter": chapter,
                "clause": "",
            }
        ]

    parts = _clause_groups(body_lines)
    if not parts:
        parts = [("", body)]

    words = [
        (word, clause)
        for clause, text in parts
        for word in text.split()
    ]
    chunks: list[ArticleChunk] = []
    start = 0
    while start < len(words):
        end = min(start + TARGET_WORDS, len(words))
        window = words[start:end]
        clauses = ",".join(dict.fromkeys(clause for _, clause in window if clause))
        text = " ".join(word for word, _ in window)
        index = len(chunks) + 1
        chunk_text = f"{heading}\n{text}"
        chunks.append(
            {
                "id": f"article-{number}-chunk-{index}",
                "text": chunk_text,
                "article": number,
                "title": title,
                "chapter": chapter,
                "clause": clauses,
            }
        )
        if end == len(words):
            break
        start = end - OVERLAP_WORDS
    return chunks


def chunk_articles(text: str) -> list[ArticleChunk]:
    articles: list[ArticleChunk] = []
    current_chapter = ""
    current_article: tuple[str, str, list[str]] | None = None
    pending_chapter_number = ""
    pending_chapter_title = False
    previous_article_number = 0

    def finish_article() -> None:
        if current_article is not None:
            number, title, body = current_article
            articles.extend(
                _chunk_article(number, title, current_chapter, body)
            )

    for line in text.splitlines():
        if TOC_LEADER_RE.search(line):
            continue

        if SCHEDULE_START_RE.match(line):
            break

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            finish_article()
            pending_chapter_number = chapter_match.group("number").title()
            chapter_title = (chapter_match.group("title") or "").strip()
            current_chapter = (
                f"{pending_chapter_number} - {chapter_title}".strip(" -")
                if chapter_title
                else pending_chapter_number
            )
            pending_chapter_title = not bool(chapter_title)
            current_article = None
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            article_number = int(article_match.group("number"))
            if article_number <= previous_article_number:
                break
            if pending_chapter_title:
                pending_chapter_title = False
            finish_article()
            current_article = (
                str(article_number),
                article_match.group("title").strip(),
                [],
            )
            previous_article_number = article_number
            continue

        if pending_chapter_title and line.strip():
            current_chapter = (
                f"{pending_chapter_number} - {line.strip().title()}"
            )
            pending_chapter_title = False
            continue

        if current_article is not None:
            current_article[2].append(line)

    finish_article()
    if not articles:
        raise ValueError(
            "No Article headings found in the PDF. Inspect extracted text and "
            "update ARTICLE_RE to match its layout."
        )
    return articles


def ingest_chunks(chunks: list[ArticleChunk]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    old_ids = collection.get(include=[])["ids"]
    if old_ids:
        collection.delete(ids=old_ids)

    model = get_embedding_model()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embeddings = model.encode(
            [f"passage: {chunk['text']}" for chunk in batch],
            normalize_embeddings=True,
        )
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[
                {
                    "article": chunk["article"],
                    "title": chunk["title"],
                    "chapter": chunk["chapter"],
                    "clause": chunk["clause"],
                }
                for chunk in batch
            ],
            embeddings=embeddings.tolist(),
        )


def main() -> None:
    sys.stdout.reconfigure(errors="backslashreplace")
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Constitution PDF not found at {PDF_PATH}. Download it from "
            "https://new.kenyalaw.org/akn/ke/act/2010/constitution/"
            "eng@2010-09-03/source"
        )

    text = extract_pdf_text()
    print("Extracted text preview (first 1000 characters):")
    print(text[:1000])
    chunks = chunk_articles(text)
    print(f"\nCreated {len(chunks)} Article chunks. Sample chunks:")
    for chunk in random.Random(42).sample(chunks, min(5, len(chunks))):
        print(
            f"\n{chunk['id']} | Article {chunk['article']} - {chunk['title']} "
            f"| Chapter {chunk['chapter']} | clause {chunk['clause'] or '-'}"
        )
        print(chunk["text"][:500])

    ingest_chunks(chunks)
    print(
        f"\nIndexed {len(chunks)} chunks in {CHROMA_PATH} "
        f"(collection: {COLLECTION_NAME})."
    )


if __name__ == "__main__":
    main()
