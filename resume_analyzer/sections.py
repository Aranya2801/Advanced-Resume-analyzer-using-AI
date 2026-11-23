"""Splitting resumes and job descriptions into sections, and reading dates.

The old version used ``re.fullmatch`` against a single regex per section,
which silently failed on very common headings like "WORK EXPERIENCE" or
"Technical Skills:". Here a heading is any short line whose normalised text
is in a known vocabulary (or starts with one of those phrases).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

RESUME_HEADINGS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "professional profile",
                "objective", "career objective", "about me", "about"),
    "experience": ("experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "career history", "internships",
                   "internship experience", "relevant experience"),
    "education": ("education", "academic background", "academics", "education and training",
                  "qualifications", "academic qualifications"),
    "skills": ("skills", "technical skills", "core skills", "skills and tools", "technologies",
               "tech stack", "core competencies", "competencies", "tools", "expertise", "key skills"),
    "projects": ("projects", "personal projects", "academic projects", "key projects",
                 "selected projects", "side projects", "portfolio"),
    "certifications": ("certifications", "certificates", "licenses", "licenses and certifications",
                       "courses", "training"),
    "publications": ("publications", "research", "papers", "research experience",
                     "research and publications"),
    "awards": ("awards", "honors", "honours", "achievements", "awards and honors",
               "accomplishments", "recognition"),
    "languages": ("languages", "spoken languages"),
    "volunteering": ("volunteering", "volunteer experience", "volunteer", "community",
                     "extracurricular", "extracurricular activities", "leadership and activities",
                     "positions of responsibility"),
}

JD_HEADINGS: dict[str, tuple[str, ...]] = {
    "about": ("about us", "about the role", "about the team", "the role", "overview",
              "who we are", "company overview"),
    "responsibilities": ("responsibilities", "what you'll do", "what you will do", "the job",
                         "your role", "key responsibilities", "duties", "in this role you will"),
    "requirements": ("requirements", "qualifications", "minimum qualifications",
                     "basic qualifications", "what you'll need", "what you bring",
                     "what we're looking for", "who you are", "must have", "must-have",
                     "required skills", "you have"),
    "preferred": ("nice to have", "nice-to-have", "preferred qualifications", "preferred",
                  "bonus", "bonus points", "good to have", "pluses", "extra credit"),
    "benefits": ("we offer", "benefits", "perks", "compensation", "what we offer", "why join us"),
}

BULLET_RE = re.compile(r"^\s*(?:[•●▪‣◦■\-–*]|\d{1,2}[.)])\s+")


def normalise_heading(line: str) -> str:
    line = line.strip().strip(":").strip()
    line = re.sub(r"[^\w\s&'/-]", "", line).lower()
    line = line.replace("&", "and")
    return re.sub(r"\s+", " ", line).strip()


def _heading_lookup(vocab: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {phrase: section for section, phrases in vocab.items() for phrase in phrases}


_RESUME_LOOKUP = _heading_lookup(RESUME_HEADINGS)
_JD_LOOKUP = _heading_lookup(JD_HEADINGS)


def match_heading(line: str, lookup: dict[str, str]) -> str | None:
    raw = line.strip()
    if not raw or len(raw) > 60 or BULLET_RE.match(raw):
        return None
    norm = normalise_heading(raw)
    if not norm or len(norm.split()) > 6:
        return None
    if norm in lookup:
        return lookup[norm]
    # "Skills & Interests", "Experience (selected)" etc. Only trust this when
    # the line looks like a heading: caps, title case or trailing colon.
    looks_like_heading = raw.isupper() or raw.endswith(":") or raw.istitle()
    if looks_like_heading:
        for phrase, section in sorted(lookup.items(), key=lambda kv: -len(kv[0])):
            if norm.startswith(phrase + " ") or norm == phrase:
                return section
    return None


def split_sections(text: str, kind: str = "resume") -> dict[str, str]:
    """Return {section_name: text}. Text before the first heading is 'header'
    for resumes and 'about' for job descriptions."""
    lookup = _RESUME_LOOKUP if kind == "resume" else _JD_LOOKUP
    current = "header" if kind == "resume" else "about"
    sections: dict[str, list[str]] = {current: []}
    for line in text.splitlines():
        heading = match_heading(line, lookup)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def bullets(text: str) -> list[str]:
    """Bullet lines, with the marker stripped. Falls back to sentence-ish
    lines when a resume has no bullet characters at all (common with DOCX)."""
    found = [BULLET_RE.sub("", ln).strip() for ln in text.splitlines() if BULLET_RE.match(ln)]
    return [b for b in found if len(b.split()) >= 3]


def units(text: str, min_words: int = 3) -> list[str]:
    """Break text into small matchable units: bullets, or sentences for prose."""
    out = []
    for line in text.splitlines():
        line = BULLET_RE.sub("", line).strip()
        if not line:
            continue
        for piece in re.split(r"(?<=[.;!?])\s+(?=[A-Z])", line):
            piece = piece.strip(" .;")
            if len(piece.split()) >= min_words:
                out.append(piece)
    return out


# --- job description requirements -----------------------------------------

PREFERRED_CUES = re.compile(r"\b(a plus|nice to have|preferred|bonus|ideally|desirable|good to have)\b", re.I)


@dataclass
class Requirement:
    text: str
    weight: float  # 1.0 required, 0.5 preferred


def jd_requirements(jd_text: str, preferred_weight: float = 0.5) -> list[Requirement]:
    """Pull out the lines of a JD that actually state requirements.

    Benefits and company blurbs are dropped. Lines under a "nice to have"
    heading, or containing cues like "a plus", get a lower weight.
    """
    sections = split_sections(jd_text, kind="jd")
    picked: list[Requirement] = []
    useful = [s for s in ("responsibilities", "requirements", "preferred") if s in sections]
    if not useful:
        # No recognisable headings - use everything except obvious benefit lines
        useful = list(sections)
    for name in useful:
        if name == "benefits":
            continue
        for unit in units(sections[name]):
            weight = preferred_weight if (name == "preferred" or PREFERRED_CUES.search(unit)) else 1.0
            picked.append(Requirement(unit, weight))
    return picked


# --- dates and experience length -----------------------------------------

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTHS["sept"] = 9

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_POINT = rf"(?:{_MONTH}\s*'?\d{{2,4}}|\d{{1,2}}/\d{{4}}|(?:19|20)\d{{2}})"
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_POINT}|{_MONTH})\s*(?:-|–|—|to|until)\s*(?P<end>{_POINT}|present|current|now|today|ongoing)",
    re.I,
)


def _parse_point(s: str, default_year: int | None = None, today: date | None = None) -> tuple[int, int] | None:
    s = s.strip().lower().rstrip(".")
    today = today or date.today()
    if s in {"present", "current", "now", "today", "ongoing"}:
        return today.year, today.month
    if m := re.match(r"(\d{1,2})/(\d{4})", s):
        return int(m.group(2)), int(m.group(1))
    if m := re.match(rf"({_MONTH})\s*'?(\d{{2,4}})?", s):
        month = MONTHS.get(m.group(1)[:4].rstrip("."), MONTHS.get(m.group(1)[:3]))
        year = m.group(2)
        if year is None:
            if default_year is None:
                return None
            year = default_year
        else:
            year = int(year)
            if year < 100:
                year += 2000 if year < 50 else 1900
        return year, month
    if m := re.match(r"(19|20)\d{2}", s):
        return int(m.group()), 1
    return None


def date_ranges(text: str, today: date | None = None) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    out = []
    for m in DATE_RANGE_RE.finditer(text):
        end = _parse_point(m.group("end"), today=today)
        if end is None:
            continue
        start = _parse_point(m.group("start"), default_year=end[0], today=today)
        if start is None or start > end:
            continue
        out.append((start, end))
    return out


def experience_months(text: str, today: date | None = None) -> int:
    """Total months covered by date ranges, with overlaps merged so two
    concurrent jobs aren't double-counted."""
    spans = sorted((s[0] * 12 + s[1], e[0] * 12 + e[1]) for s, e in date_ranges(text, today))
    total, cur_start, cur_end = 0, None, None
    for start, end in spans:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start + 1
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start + 1
    return total
