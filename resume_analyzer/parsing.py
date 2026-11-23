"""Turning resume files into plain text, plus contact-detail extraction.

PDF goes through pdfplumber, DOCX through python-docx (paragraphs *and*
tables, since a lot of templates put the whole header in a table).
Legacy .doc is not supported - convert it to .docx or PDF first.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Roughly what one page of a resume holds. Only used when the format has no
# real notion of pages (txt/docx).
WORDS_PER_PAGE = 450


@dataclass
class ParsedDocument:
    text: str
    pages: int
    source: str = ""


@dataclass
class ContactInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def found(self) -> list[str]:
        return [k for k, v in asdict(self).items() if v]


class UnsupportedFormatError(ValueError):
    pass


def _clean(text: str) -> str:
    # NFKC folds ligatures (ﬁ -> fi) that pdf extraction loves to produce.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(data: bytes) -> ParsedDocument:
    import pdfplumber

    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            parts.append(page.extract_text(x_tolerance=2, y_tolerance=3) or "")
    text = _clean("\n".join(parts))
    if len(text) < 50:
        # Almost certainly a scanned PDF. OCR is out of scope for now, but
        # saying so is better than returning an empty analysis.
        raise UnsupportedFormatError(
            "Could not extract text from this PDF (it looks scanned). "
            "Export it as a text-based PDF or DOCX and try again."
        )
    return ParsedDocument(text=text, pages=pages)


def _read_docx(data: bytes) -> ParsedDocument:
    import docx

    document = docx.Document(io.BytesIO(data))
    lines = []
    for table in document.tables:
        for row in table.rows:
            seen = set()
            for cell in row.cells:
                # merged cells show up once per grid column
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                if cell.text.strip():
                    lines.append(cell.text.strip())
    for para in document.paragraphs:
        if not para.text.strip():
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        prefix = "• " if "list" in style else ""
        lines.append(prefix + para.text.strip())
    text = _clean("\n".join(lines))
    return ParsedDocument(text=text, pages=max(1, round(len(text.split()) / WORDS_PER_PAGE)))


def _read_text(data: bytes) -> ParsedDocument:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    text = _clean(text)
    return ParsedDocument(text=text, pages=max(1, round(len(text.split()) / WORDS_PER_PAGE)))


def load_document(source: str | Path | bytes, filename: str | None = None) -> ParsedDocument:
    """Read a resume from a path or raw bytes.

    When passing bytes, ``filename`` is needed so we know the format.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        filename = filename or path.name
    else:
        data = source
        if not filename:
            raise ValueError("filename is required when passing raw bytes")

    ext = Path(filename).suffix.lower()
    if ext == ".doc":
        raise UnsupportedFormatError("Legacy .doc files aren't supported. Save as .docx or PDF.")
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported file type: {ext or '(none)'}")

    if ext == ".pdf":
        doc = _read_pdf(data)
    elif ext == ".docx":
        doc = _read_docx(data)
    else:
        doc = _read_text(data)
    doc.source = filename
    return doc


# --- contact details ---------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/[\w\-%]+/?", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?github\.com/[\w\-]+", re.I)
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.(?:dev|io|me|com|org|net|ai|app|in)(?:/[^\s|,]*)?", re.I)
# Candidate phone numbers: a run of digits, spaces, dots, dashes and brackets.
# Validated separately so date ranges like "2018 - 2021" don't sneak in.
PHONE_CANDIDATE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s().\-]{7,18}\d(?![\w/])")
YEAR_RANGE_RE = re.compile(r"^(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}$")

_NOT_A_NAME = re.compile(r"(resume|curriculum|vitae|cv\b|@|\d|http|www\.)", re.I)


def _looks_like_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 10 <= len(digits) <= 15:
        return False
    if YEAR_RANGE_RE.match(candidate.strip()):
        return False
    return True


def _guess_name(lines: list[str]) -> str:
    for line in lines[:6]:
        # header lines are often "Name | email | phone"
        first = re.split(r"[|•·,]", line)[0].strip()
        words = first.split()
        if not 2 <= len(words) <= 4 or _NOT_A_NAME.search(first):
            continue
        if all(w[:1].isupper() for w in words if w.isalpha()) and sum(w.isalpha() for w in words) >= 2:
            return first.title() if first.isupper() else first
    return ""


def extract_contact(text: str) -> ContactInfo:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    info = ContactInfo(name=_guess_name(lines))

    if m := EMAIL_RE.search(text):
        info.email = m.group()
    for m in PHONE_CANDIDATE_RE.finditer(text[:1500]):  # phone lives near the top
        if _looks_like_phone(m.group()):
            info.phone = re.sub(r"\s+", " ", m.group()).strip()
            break
    if m := LINKEDIN_RE.search(text):
        info.linkedin = m.group().rstrip("/")
    if m := GITHUB_RE.search(text):
        info.github = m.group()

    email_domain = info.email.split("@")[-1].lower() if info.email else ""
    for m in URL_RE.finditer(text[:1500]):
        url = m.group().rstrip(".,)")
        low = url.lower()
        if any(s in low for s in ("linkedin.com", "github.com")) or (email_domain and low.endswith(email_domain)):
            continue
        if "@" in text[max(0, m.start() - 1) : m.start()]:
            continue
        info.website = url
        break
    return info
