"""The analyzer: parse -> sections -> skills/bullets -> quality score -> (JD match)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .bullets import BulletReport, analyse_bullets
from .encoders import Encoder, cached_encoder
from .matching import POOL_WEIGHTS, MatchConfig, MatchResult, combine, match, resume_match_units, tfidf_scores
from .parsing import ContactInfo, ParsedDocument, extract_contact, load_document
from .scoring import QualityReport, quality_report
from .sections import bullets as find_bullets
from .sections import experience_months, split_sections
from .skills import SkillHit, SkillTaxonomy, default_taxonomy


@dataclass
class Resume:
    text: str
    pages: int
    sections: dict[str, str]
    contact: ContactInfo
    skills: dict[str, SkillHit]
    bullets: list[str]
    months_experience: int
    source: str = ""
    _units: list[str] | None = field(default=None, repr=False)

    @property
    def units(self) -> list[str]:
        if self._units is None:
            self._units = resume_match_units(self.sections)
        return self._units


@dataclass
class Analysis:
    resume: Resume
    quality: QualityReport
    bullets: BulletReport
    match: MatchResult | None = None
    taxonomy: SkillTaxonomy | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        tax = self.taxonomy or default_taxonomy()
        r = self.resume
        out = {
            "source": r.source,
            "contact": r.contact.to_dict(),
            "stats": {
                "words": len(r.text.split()),
                "pages": r.pages,
                "bullets": len(r.bullets),
                "months_experience": r.months_experience,
                "sections_found": [s for s in r.sections if s != "header"],
            },
            "skills": tax.by_group(r.skills),
            "quality": self.quality.to_dict(),
            "bullets": self.bullets.to_dict(),
        }
        if self.match is not None:
            out["match"] = self.match.to_dict()
        return out

    def summary_text(self) -> str:
        """Short plain-text summary, handed to the LLM so its feedback builds
        on the measured numbers."""
        q = self.quality
        lines = [f"Overall quality {q.overall:.0f}/100."]
        for name, comp in q.components.items():
            lines.append(f"- {name}: {comp.score:.0f} ({'; '.join(comp.findings)})")
        if self.match:
            m = self.match
            lines.append(f"Job match {m.score * 100:.0f}/100. Missing skills: {', '.join(m.skills.missing[:10]) or 'none'}.")
        return "\n".join(lines)


class ResumeAnalyzer:
    def __init__(self, encoder: Encoder | str | None = "auto", taxonomy: SkillTaxonomy | None = None,
                 match_config: MatchConfig | None = None, today: date | None = None):
        self.taxonomy = taxonomy or default_taxonomy()
        self.encoder = cached_encoder(encoder) if isinstance(encoder, str) else encoder
        self.match_config = match_config or MatchConfig()
        self.today = today

    def parse(self, source: str | Path | bytes | ParsedDocument, filename: str | None = None) -> Resume:
        if isinstance(source, ParsedDocument):
            doc = source
        elif isinstance(source, str) and filename is None and ("\n" in source or not Path(source).suffix):
            doc = ParsedDocument(text=source.strip(), pages=max(1, round(len(source.split()) / 450)))
        else:
            doc = load_document(source, filename)
        sections = split_sections(doc.text, "resume")
        dated_text = sections.get("experience", "") or doc.text
        return Resume(
            text=doc.text,
            pages=doc.pages,
            sections=sections,
            contact=extract_contact(doc.text),
            skills=self.taxonomy.extract(sections),
            bullets=find_bullets(doc.text),
            months_experience=experience_months(dated_text, self.today),
            source=doc.source,
        )

    def analyze(self, source, job_description: str | None = None, filename: str | None = None) -> Analysis:
        resume = source if isinstance(source, Resume) else self.parse(source, filename)
        bullet_report = analyse_bullets(resume.bullets)
        quality = quality_report(resume.text, resume.sections, resume.contact, bullet_report,
                                 resume.pages, resume.months_experience)
        result = Analysis(resume, quality, bullet_report, taxonomy=self.taxonomy)
        if job_description and job_description.strip():
            result.match = self.match(resume, job_description)
        return result

    def match(self, resume: Resume, job_description: str) -> MatchResult:
        return match(resume.text, resume.sections, resume.skills, job_description, self.taxonomy,
                     self.encoder, self.match_config, resume_units=resume.units)

    def rank(self, resumes: list[Resume], job_description: str,
             weights: dict[str, float] | None = None) -> list[RankedResume]:
        """Rank a pool of resumes for one JD ("pool mode").

        With three or more resumes, TF-IDF similarity over the pool is added
        as a fourth signal (min-max scaled within the pool). It needs corpus
        statistics, which is why it isn't part of the single-resume score.
        """
        weights = weights or POOL_WEIGHTS
        matches = [self.match(r, job_description) for r in resumes]
        tfidf = None
        if len(resumes) >= 3:
            raw = tfidf_scores([r.text for r in resumes], job_description)
            span = raw.max() - raw.min()
            tfidf = (raw - raw.min()) / span if span else np.zeros_like(raw)
        out = []
        for i, (r, m) in enumerate(zip(resumes, matches)):
            comps = dict(m.components)
            if tfidf is not None:
                comps["tfidf"] = float(tfidf[i])
            out.append(RankedResume(r, m, combine(comps, weights), comps))
        return sorted(out, key=lambda x: -x.score)


@dataclass
class RankedResume:
    resume: Resume
    match: MatchResult
    score: float
    components: dict[str, float]
