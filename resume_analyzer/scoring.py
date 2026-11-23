"""Resume quality rubric (no job description needed).

Every sub-score is computed from things you can point at in the resume, and
each one comes with the findings that produced it. The weights and target
ranges below are my own judgement calls, collected from recruiter guides
and career-centre checklists - they are *not* learned from data, and the
README says so. Change them in RUBRIC_WEIGHTS if you disagree.

The previous version asked an LLM for the overall score, which meant the
same resume could get 72 one run and 81 the next. This is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bullets import BulletReport
from .parsing import ContactInfo

RUBRIC_WEIGHTS = {
    "impact": 0.25,          # strong vs weak bullet openers
    "quantification": 0.20,  # bullets with a concrete number
    "structure": 0.20,       # expected sections present
    "ats": 0.15,             # parse-ability red flags
    "concision": 0.10,       # overall length and bullet length
    "readability": 0.10,     # Flesch reading ease, within a resume-ish band
}

CORE_SECTIONS = ("experience", "education", "skills")
NICE_SECTIONS = ("summary", "projects")


@dataclass
class Component:
    score: float  # 0-100
    findings: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    overall: float
    components: dict[str, Component]

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 1),
            "components": {k: {"score": round(c.score, 1), "findings": c.findings}
                           for k, c in self.components.items()},
        }


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below lo, 100 at or above hi, linear in between."""
    if hi == lo:
        return 100.0 if x >= hi else 0.0
    return max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))


def _band(x: float, lo_ok: float, hi_ok: float, lo_zero: float, hi_zero: float) -> float:
    """100 inside [lo_ok, hi_ok], falling to 0 at lo_zero / hi_zero."""
    if lo_ok <= x <= hi_ok:
        return 100.0
    if x < lo_ok:
        return _ramp(x, lo_zero, lo_ok)
    return 100.0 - _ramp(x, hi_ok, hi_zero)


def score_impact(b: BulletReport) -> Component:
    if b.total == 0:
        return Component(0, ["No bullet points found - recruiters skim bullets, so use them for experience and projects."])
    # 70% strong openers is already very good; weak openers cost extra
    s = _ramp(b.strong_ratio, 0.2, 0.7) - 60 * b.weak_ratio
    findings = [f"{b.strong}/{b.total} bullets open with a strong action verb."]
    if b.weak:
        findings.append(f"{b.weak} bullet(s) open with passive phrasing such as 'responsible for' or 'helped'.")
    return Component(max(0.0, s), findings)


def score_quantification(b: BulletReport) -> Component:
    if b.total == 0:
        return Component(0, ["No bullets to check for metrics."])
    s = _ramp(b.quantified_ratio, 0.1, 0.6)
    findings = [f"{b.quantified}/{b.total} bullets contain a number (%, $, users, latency, ...)."]
    if b.quantified_ratio < 0.4:
        findings.append("Aim for roughly half of your bullets to show scale or outcome with a number.")
    return Component(s, findings)


def score_structure(sections: dict[str, str], contact: ContactInfo) -> Component:
    findings = []
    s = 0.0
    missing_core = [x for x in CORE_SECTIONS if x not in sections]
    # projects can stand in for experience on a student resume
    if "experience" in missing_core and "projects" in sections:
        missing_core.remove("experience")
    s += 60 * (1 - len(missing_core) / len(CORE_SECTIONS))
    if missing_core:
        findings.append("Missing or unrecognised section(s): " + ", ".join(missing_core) + ".")
    present_nice = [x for x in NICE_SECTIONS if x in sections]
    s += 10 * len(present_nice)
    absent_nice = [x for x in NICE_SECTIONS if x not in sections]
    if absent_nice:
        findings.append("Optional but useful: " + " and ".join(absent_nice) + " section.")
    contact_pts = sum(bool(getattr(contact, f)) for f in ("email", "phone")) * 7.5
    contact_pts += 5 if (contact.linkedin or contact.github or contact.website) else 0
    s += contact_pts
    if not contact.email:
        findings.append("No email address detected.")
    if not contact.phone:
        findings.append("No phone number detected.")
    if not (contact.linkedin or contact.github or contact.website):
        findings.append("No LinkedIn / GitHub / portfolio link.")
    if not findings:
        findings.append("All the usual sections and contact details are present.")
    return Component(min(100.0, s), findings)


def score_ats(text: str, sections: dict[str, str], pages: int) -> Component:
    s, findings = 100.0, []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pipes = sum(ln.count("|") for ln in lines)
    if pipes > 8:
        s -= 10
        findings.append("Lots of '|' separators - often a sign of tables or columns, which some ATS parsers scramble.")
    odd = sum(1 for ch in text if ord(ch) > 0x2000 and ch not in "•–—‘’“”…€£₹·●▪")
    if odd > 15:
        s -= 10
        findings.append(f"{odd} unusual symbols/icons found; icon fonts often come out as garbage in ATS.")
    if lines:
        fragment_ratio = sum(1 for ln in lines if len(ln.split()) <= 2) / len(lines)
        if fragment_ratio > 0.45:
            s -= 15
            findings.append("Many very short lines - could be a multi-column layout that parses out of order.")
    if len([k for k in sections if k != "header"]) < 2:
        s -= 25
        findings.append("Couldn't find standard section headings (Experience, Education, Skills).")
    if pages > 2:
        s -= 10
        findings.append(f"{pages} pages; keep it to 1-2.")
    if not findings:
        findings.append("No obvious parsing problems.")
    return Component(max(0.0, s), findings)


def score_concision(word_count: int, b: BulletReport, months_experience: int) -> Component:
    # more experience earns more room
    hi = 750 if months_experience < 60 else 1000
    s = 0.6 * _band(word_count, 300, hi, 120, hi + 600)
    if b.total:
        s += 0.4 * 100 * (1 - (b.too_long + b.too_short) / b.total)
    else:
        s += 0.4 * 50
    findings = [f"{word_count} words (target ~300-{hi} for your experience level)."]
    if b.too_long:
        findings.append(f"{b.too_long} bullet(s) run past 35 words; split or trim them.")
    return Component(s, findings)


def _syllables(word: str) -> int:
    # Vowel-group heuristic. Off by one on plenty of words, but that washes
    # out over a whole resume and avoids depending on a pronunciation dict.
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def flesch_reading_ease(text: str) -> float | None:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    if len(words) < 30:
        return None
    # Resume lines rarely end with a full stop, so treat line breaks and
    # bullet boundaries as sentence ends too.
    sentences = [s for s in re.split(r"[.!?]+\s|\n+", text) if re.search(r"[A-Za-z]", s)]
    syllables = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / max(1, len(sentences))) - 84.6 * (syllables / len(words))


def score_readability(text: str) -> Component:
    ease = flesch_reading_ease(text)
    if ease is None:
        return Component(50.0, ["Too little text to judge readability."])
    # Resumes are dense with jargon; 30-65 is normal, below ~10 is a wall of text.
    s = _band(ease, 30, 65, 0, 95)
    return Component(s, [f"Flesch reading ease {ease:.0f} (30-65 is typical for technical resumes)."])


def quality_report(text: str, sections: dict[str, str], contact: ContactInfo, bullets: BulletReport,
                   pages: int, months_experience: int) -> QualityReport:
    comps = {
        "impact": score_impact(bullets),
        "quantification": score_quantification(bullets),
        "structure": score_structure(sections, contact),
        "ats": score_ats(text, sections, pages),
        "concision": score_concision(len(text.split()), bullets, months_experience),
        "readability": score_readability(text),
    }
    overall = sum(RUBRIC_WEIGHTS[k] * c.score for k, c in comps.items())
    return QualityReport(overall, comps)
