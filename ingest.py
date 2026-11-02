import re
import random
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

from rag import (
    CHROMA_PATH,
    COLLECTION_NAME,
    CONSTITUTION_SOURCE,
    get_embedding_model,
)

ROOT_DIR = Path(__file__).resolve().parent
PDF_PATH = ROOT_DIR / "data" / "Constitution_of_Kenya_2010.pdf"
EMPLOYMENT_ACT_PDF_PATH = (
    ROOT_DIR / "data" / "Employment_Act_2007_2024-04-26.pdf"
)
STATUTE_PDFS: tuple[tuple[str, Path], ...] = (
    ("Criminal Procedure Code", ROOT_DIR / "data" / "Criminal Procedure Code.pdf"),
    ("Penal Code", ROOT_DIR / "data" / "Penal Code.pdf"),
    (
        "Landlord and Tenant (Shops, Hotels and Catering Establishments) Act",
        ROOT_DIR
        / "data"
        / "Landlord and Tenant (Shops Hotels and Catering Establishments) Act.pdf",
    ),
    ("Rent Restriction Act", ROOT_DIR / "data" / "Rent Restriction Act.pdf"),
    ("Small Claims Court Act", ROOT_DIR / "data" / "Small Claims Court Act.pdf"),
    ("Consumer Protection Act", ROOT_DIR / "data" / "Consumer Protection Act.pdf"),
    ("Law of Succession Act", ROOT_DIR / "data" / "Law of Succession Act.pdf"),
    ("Land Act", ROOT_DIR / "data" / "Land Act.pdf"),
    ("Sexual Offences Act", ROOT_DIR / "data" / "Sexual Offences Act.pdf"),
    ("Data Protection Act", ROOT_DIR / "data" / "Data Protection Act.pdf"),
)
STATUTE_DEFAULT_CHAPTERS = {
    "Landlord and Tenant (Shops, Hotels and Catering Establishments) Act":
        "Cap. 301",
    "Rent Restriction Act": "Cap. 296",
    "Law of Succession Act": "Cap. 160",
    "Sexual Offences Act": "Cap. 63A",
}
TARGET_WORDS = 350
MAX_WORDS = 500
OVERLAP_WORDS = 50
BATCH_SIZE = 32

ARTICLE_RE = re.compile(
    r"^\s*(?:ARTICLE\s+)?(?P<number>\d{1,3})\s*"
    r"[.\u2010-\u2015\u2212-]\s*"
    r"(?P<title>\S.*)\s*$",
    re.IGNORECASE,
)
PART_RE = re.compile(
    r"^\s*PART\s+(?P<number>[A-Z]+|\d+)\s*"
    r"[.:\-\u2010-\u2015\u2212\uFFFD]\s*(?P<title>.*)\s*$",
    re.IGNORECASE,
)
STATUTE_SECTION_RE = re.compile(
    r"^\s*(?P<number>\d{1,3}[A-Z]?)\s*[.\u2010-\u2015\u2212-]\s*"
    r"(?P<title>\S.*)\s*$",
    re.IGNORECASE,
)
STATUTE_AMENDMENT_NOTE_RE = re.compile(
    r"^\s*\d{1,3}[A-Z]?\.\s*of\s+(?:18|19|20)\d{2}\b",
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
    source: str
    section: str
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
                "source": CONSTITUTION_SOURCE,
                "section": number,
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
                "source": CONSTITUTION_SOURCE,
                "section": number,
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
                "source": CONSTITUTION_SOURCE,
                "section": number,
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


def _chunk_statute_section(
    number: str,
    title: str,
    body_lines: list[str],
    source: str,
    chapter: str,
) -> list[ArticleChunk]:
    heading = f"Section {number}. {title}".strip()
    source_id = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-")
    body = "\n".join(line for line in body_lines if line.strip()).strip()
    if not body or len(body.split()) <= MAX_WORDS:
        return [
            {
                "id": f"{source_id}-section-{number}-chunk-1",
                "text": f"{heading}\n{body}".strip(),
                "source": source,
                "section": number,
                "article": "",
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
        chunks.append(
            {
                "id": f"{source_id}-section-{number}-chunk-{index}",
                "text": f"{heading}\n{text}",
                "source": source,
                "section": number,
                "article": "",
                "title": title,
                "chapter": chapter,
                "clause": clauses,
            }
        )
        if end == len(words):
            break
        start = end - OVERLAP_WORDS
    return chunks


def chunk_statute_sections(text: str, source: str) -> list[ArticleChunk]:
    sections: list[ArticleChunk] = []
    current_section: tuple[str, str, list[str]] | None = None
    current_chapter = STATUTE_DEFAULT_CHAPTERS.get(source, "")
    pending_chapter_title = False

    def finish_section() -> None:
        if current_section is not None:
            number, title, body = current_section
            sections.extend(
                _chunk_statute_section(
                    number,
                    title,
                    body,
                    source,
                    current_chapter,
                )
            )

    for line in text.splitlines():
        if TOC_LEADER_RE.search(line):
            continue
        if SCHEDULE_START_RE.match(line):
            break

        part_match = PART_RE.match(line)
        if part_match:
            finish_section()
            current_section = None
            title = part_match.group("title").strip(" \uFFFD-:.\u2013\u2014")
            current_chapter = (
                f"Part {part_match.group('number')} - {title}".strip(" -")
            )
            continue

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            finish_section()
            current_section = None
            chapter_number = chapter_match.group("number").title()
            chapter_title = (chapter_match.group("title") or "").strip()
            current_chapter = (
                f"Chapter {chapter_number} - {chapter_title}".strip(" -")
            )
            pending_chapter_title = not bool(chapter_title)
            continue

        if STATUTE_AMENDMENT_NOTE_RE.match(line):
            continue

        section_match = STATUTE_SECTION_RE.match(line)
        if section_match:
            pending_chapter_title = False
            finish_section()
            current_section = (
                section_match.group("number"),
                section_match.group("title").strip(),
                [],
            )
            continue
        if pending_chapter_title and line.strip():
            current_chapter = f"{current_chapter} - {line.strip()}"
            pending_chapter_title = False
            continue
        if current_section is not None:
            current_section[2].append(line)

    finish_section()
    if not sections:
        raise ValueError(
            f"No numbered sections found in the {source} PDF. "
            "Inspect extracted text and update ARTICLE_RE to match its layout."
        )
    return sections


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
                    "source": chunk["source"],
                    "section": chunk["section"],
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
    source_pdfs = [
        (CONSTITUTION_SOURCE, PDF_PATH),
        ("Employment Act", EMPLOYMENT_ACT_PDF_PATH),
        *STATUTE_PDFS,
    ]
    chunks: list[ArticleChunk] = []
    for source, pdf_path in source_pdfs:
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"{source} PDF not found at {pdf_path}. See the README for "
                "the verified source and version."
            )
        text = extract_pdf_text(pdf_path)
        source_chunks = (
            chunk_articles(text)
            if source == CONSTITUTION_SOURCE
            else chunk_statute_sections(text, source)
        )
        chunks.extend(source_chunks)
        print(f"\n{source} text preview (first 1000 characters):")
        print(text[:1000])
        print(f"\nCreated {len(source_chunks)} chunks.")

    print(f"\nSample chunks from {len(chunks)} legal excerpts:")
    for chunk in random.Random(42).sample(chunks, min(5, len(chunks))):
        print(
            f"\n{chunk['id']} | {chunk['source']}, "
            f"{'Article' if chunk['article'] else 's.'} {chunk['section']} "
            f"| {chunk['title']} | subsection {chunk['clause'] or '-'}"
        )
        print(chunk["text"][:500])

    ingest_chunks(chunks)
    print(
        f"\nIndexed {len(chunks)} chunks in {CHROMA_PATH} "
        f"(collection: {COLLECTION_NAME})."
    )


if __name__ == "__main__":
    main()
