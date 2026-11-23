"""Resume <-> job description matching.

The score is a weighted mix of three signals that fail in different ways:

1. **Skill coverage** - which skills the JD asks for are present in the
   resume, with partial credit for a sibling skill (TensorFlow for PyTorch)
   and reduced credit when a skill is only *listed* and never shown in an
   experience or project bullet. Fully explainable; blind to anything outside
   the taxonomy.
2. **Requirement coverage** (dense) - every JD requirement line is embedded
   and matched to its most similar resume line; the score is the weighted
   mean of those best matches. Catches paraphrases ("ran containerised
   workloads on EKS") that the taxonomy misses. This replaces the old
   approach of embedding the first 512 characters of each document, which
   mostly compared contact headers with company blurbs.
3. **Keyword recall** - share of the JD's content words that appear in the
   resume. Crude, but it's what a lot of ATS filters actually do.

The baseline functions at the bottom exist for the benchmark only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from .encoders import Encoder
from .sections import Requirement, jd_requirements, units
from .skills import SkillHit, SkillTaxonomy

# Sections where a skill mention counts as demonstrated rather than claimed.
EVIDENCE_SECTIONS = {"experience", "projects", "publications", "certifications", "volunteering", "awards"}

# Single-resume mode: no other resumes to compare against, so only absolute
# signals. Set by hand before running the benchmark; kept as-is so the
# benchmark can report how a "reasonable guess" does.
DEFAULT_WEIGHTS = {"skills": 0.5, "dense": 0.3, "keywords": 0.2}

# Pool mode (ranking many resumes at once) adds TF-IDF similarity computed over
# the pool. These are the weights picked on the benchmark's dev split
# (setting A) - see benchmark/results/.
POOL_WEIGHTS = {"skills": 0.2, "dense": 0.2, "keywords": 0.3, "tfidf": 0.3}


@dataclass
class MatchConfig:
    related_credit: float = 0.5      # credit for a same-family skill
    listed_only_credit: float = 0.6  # credit for a skill that only appears in the skills list
    preferred_weight: float = 0.5    # weight of "nice to have" requirements
    use_families: bool = True
    use_evidence: bool = True
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class SkillMatchItem:
    skill: str
    weight: float
    credit: float
    via: str  # "exact", "listed-only", "related:<name>", "missing"


@dataclass
class SkillMatch:
    score: float
    items: list[SkillMatchItem]

    @property
    def matched(self) -> list[str]:
        return [i.skill for i in self.items if i.via in ("exact", "listed-only")]

    @property
    def partial(self) -> list[str]:
        return [f"{i.skill} (have {i.via.split(':', 1)[1]})" for i in self.items if i.via.startswith("related:")]

    @property
    def missing(self) -> list[str]:
        return [i.skill for i in sorted(self.items, key=lambda i: -i.weight) if i.via == "missing"]


@dataclass
class RequirementMatch:
    requirement: str
    weight: float
    best_line: str
    similarity: float


@dataclass
class MatchResult:
    score: float
    components: dict[str, float]
    skills: SkillMatch
    requirements: list[RequirementMatch]
    keyword_hits: list[str]
    keyword_misses: list[str]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score * 100, 1),
            "components": {k: round(v * 100, 1) for k, v in self.components.items()},
            "skills": {
                "score": round(self.skills.score * 100, 1),
                "matched": self.skills.matched,
                "partial": self.skills.partial,
                "missing": self.skills.missing,
                "items": [asdict(i) for i in self.skills.items],
            },
            "requirements": [asdict(r) for r in self.requirements],
            "keywords": {"hits": self.keyword_hits, "misses": self.keyword_misses},
        }


# --- skill coverage -----------------------------------------------------------

def jd_skill_weights(jd_text: str, taxonomy: SkillTaxonomy, preferred_weight: float = 0.5) -> dict[str, float]:
    weights: dict[str, float] = {}
    reqs = jd_requirements(jd_text, preferred_weight) or [Requirement(jd_text, 1.0)]
    for req in reqs:
        for name in taxonomy.find(req.text):
            weights[name] = max(weights.get(name, 0.0), req.weight)
    return weights


def skill_coverage(resume_hits: dict[str, SkillHit], jd_weights: dict[str, float],
                   taxonomy: SkillTaxonomy, cfg: MatchConfig | None = None) -> SkillMatch:
    cfg = cfg or MatchConfig()
    if not jd_weights:
        return SkillMatch(0.0, [])

    def evidence(hit: SkillHit) -> tuple[float, str]:
        if not cfg.use_evidence or hit.sections & EVIDENCE_SECTIONS:
            return 1.0, "exact"
        return cfg.listed_only_credit, "listed-only"

    items = []
    for name, w in jd_weights.items():
        hit = resume_hits.get(name)
        if hit:
            credit, via = evidence(hit)
        else:
            credit, via = 0.0, "missing"
            family = taxonomy.get(name).family
            if cfg.use_families and taxonomy.substitutable(family):
                siblings = [h for h in resume_hits.values() if h.skill.family == family]
                if siblings:
                    best = max(siblings, key=lambda h: evidence(h)[0])
                    credit = cfg.related_credit * evidence(best)[0]
                    via = f"related:{best.skill.name}"
        items.append(SkillMatchItem(name, w, credit, via))

    total = sum(i.weight for i in items)
    score = sum(i.weight * i.credit for i in items) / total
    items.sort(key=lambda i: (-i.weight, -i.credit, i.skill))
    return SkillMatch(score, items)


# --- dense requirement coverage -------------------------------------------

def requirement_coverage(resume_units: list[str], reqs: list[Requirement],
                         encoder: Encoder, resume_vecs: np.ndarray | None = None,
                         req_vecs: np.ndarray | None = None) -> tuple[float, list[RequirementMatch]]:
    if not resume_units or not reqs:
        return 0.0, []
    R = resume_vecs if resume_vecs is not None else encoder.encode(resume_units)
    Q = req_vecs if req_vecs is not None else encoder.encode([r.text for r in reqs])
    sims = Q @ R.T
    best_idx = sims.argmax(axis=1)
    best = encoder.rescale(sims.max(axis=1))
    weights = np.array([r.weight for r in reqs])
    score = float((weights * best).sum() / weights.sum())
    details = [
        RequirementMatch(r.text, r.weight, resume_units[j], round(float(s), 3))
        for r, j, s in zip(reqs, best_idx, best)
    ]
    return score, details


# --- keyword recall -------------------------------------------------------------

# Words that show up in nearly every JD and say nothing about fit.
JD_FILLER = {
    "experience", "years", "year", "team", "teams", "work", "working", "strong", "ability",
    "able", "skills", "skill", "knowledge", "understanding", "including", "plus", "preferred",
    "required", "requirements", "responsibilities", "role", "join", "looking", "candidate",
    "excellent", "good", "great", "proven", "track", "record", "solid", "hands", "using",
    "build", "building", "help", "across", "within", "related", "field", "degree", "similar",
    "equivalent", "least", "minimum", "deep", "expert", "level", "proficiency", "proficient",
    "familiarity", "familiar", "environment", "opportunity", "company", "nice", "bonus",
    "will", "you", "your", "our", "we", "us", "etc", "e.g", "i.e",
}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#.\-]*[a-z0-9+#]|[a-z]")


def _stem(tok: str) -> str:
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
            return tok[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return tok


def content_tokens(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    toks = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in toks if len(t) > 2 and t not in ENGLISH_STOP_WORDS and t not in JD_FILLER]


def keyword_recall(resume_text: str, reqs: list[Requirement]) -> tuple[float, list[str], list[str]]:
    weights: dict[str, float] = {}
    for r in reqs:
        for tok in set(content_tokens(r.text)):
            weights[tok] = max(weights.get(tok, 0.0), r.weight)
    if not weights:
        return 0.0, [], []
    have = set(content_tokens(resume_text))
    hits = sorted((t for t in weights if t in have), key=lambda t: (-weights[t], t))
    misses = sorted((t for t in weights if t not in have), key=lambda t: (-weights[t], t))
    score = sum(weights[t] for t in hits) / sum(weights.values())
    return score, hits, misses


# --- putting it together ----------------------------------------------------

def combine(components: dict[str, float | None], weights: dict[str, float]) -> float:
    """Weighted mean over the components that are actually available."""
    usable = {k: v for k, v in components.items() if v is not None and weights.get(k, 0) > 0}
    if not usable:
        return 0.0
    total_w = sum(weights[k] for k in usable)
    return sum(weights[k] * v for k, v in usable.items()) / total_w


def match(resume_text: str, resume_sections: dict[str, str], resume_hits: dict[str, SkillHit],
          jd_text: str, taxonomy: SkillTaxonomy, encoder: Encoder | None = None,
          cfg: MatchConfig | None = None, resume_units: list[str] | None = None,
          resume_vecs: np.ndarray | None = None) -> MatchResult:
    cfg = cfg or MatchConfig()
    reqs = jd_requirements(jd_text, cfg.preferred_weight) or [Requirement(u, 1.0) for u in units(jd_text)]

    skills = skill_coverage(resume_hits, jd_skill_weights(jd_text, taxonomy, cfg.preferred_weight), taxonomy, cfg)

    dense_score, req_details = None, []
    if encoder is not None:
        r_units = resume_units if resume_units is not None else resume_match_units(resume_sections)
        dense_score, req_details = requirement_coverage(r_units, reqs, encoder, resume_vecs)

    kw_score, kw_hits, kw_misses = keyword_recall(resume_text, reqs)

    components = {"skills": skills.score if skills.items else None, "dense": dense_score, "keywords": kw_score}
    score = combine(components, cfg.weights)
    return MatchResult(
        score=score,
        components={k: v for k, v in components.items() if v is not None},
        skills=skills,
        requirements=req_details,
        keyword_hits=kw_hits[:40],
        keyword_misses=kw_misses[:40],
    )


def resume_match_units(sections: dict[str, str]) -> list[str]:
    """Lines worth matching JD requirements against. The contact header is
    left out; skills lists are kept since "Languages: Python, Go" is a
    perfectly good answer to "proficiency in Python"."""
    out = []
    for name, text in sections.items():
        if name == "header":
            continue
        out.extend(units(text, min_words=2))
    return out


# --- baselines (used by benchmark/evaluate.py) ----------------------------

def v1_keyword_overlap(resume_text: str, jd_text: str) -> float:
    """The original repo's keyword score, reproduced as-is for comparison:
    |JD words ∩ resume words| / |JD words| over words with 4+ letters."""
    jd_words = set(re.findall(r"\b\w{4,}\b", jd_text.lower()))
    resume_words = set(re.findall(r"\b\w{4,}\b", resume_text.lower()))
    stop = {"this", "that", "with", "have", "from", "they", "will", "your", "been", "were", "more",
            "also", "some", "than", "such", "when", "then", "each", "their", "other", "into",
            "about", "over", "after"}
    matched = {w for w in jd_words & resume_words if w not in stop}
    return len(matched) / (len(jd_words) or 1)


class BM25:
    """Okapi BM25 over a fixed corpus of resumes. Needs the whole pool, so it
    can rank candidates but can't score one resume in isolation."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(content_tokens(d)) for d in docs]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = self.lengths.mean() if len(self.docs) else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = content_tokens(query)
        out = np.zeros(len(self.docs))
        for i, doc in enumerate(self.docs):
            denom_base = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
            s = 0.0
            for t in q:
                tf = doc.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + denom_base)
            out[i] = s
        return out


def tfidf_scores(resumes: list[str], query: str) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(tokenizer=content_tokens, lowercase=False, token_pattern=None, sublinear_tf=True)
    m = vec.fit_transform(resumes + [query])
    sims = (m[:-1] @ m[-1].T).toarray().ravel()
    return sims


def dense_truncated(resume_vec: np.ndarray, jd_vec: np.ndarray) -> float:
    """Whole-document cosine, the way v1 did it (after truncating both texts
    to 512 characters)."""
    return float(resume_vec @ jd_vec)
