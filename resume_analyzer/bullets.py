"""Bullet-level writing checks: opening verbs, metrics, length.

Two bugs from the first version are fixed here:

* weak phrases were matched as substrings anywhere in the bullet, so "was"
  matched "Kubernetes was migrated" and "had" matched "shadow traffic";
  now only the *opening* of the bullet is checked.
* the "has a number" regex accepted a bare comma, so almost every bullet
  counted as quantified. Years and date ranges are also no longer metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STRONG_VERBS = {
    "accelerated", "achieved", "analysed", "analyzed", "architected", "automated", "boosted",
    "built", "championed", "co-authored", "collaborated", "configured", "consolidated",
    "created", "cut", "debugged", "decreased", "defined", "delivered", "deployed", "designed",
    "developed", "devised", "directed", "doubled", "drove", "eliminated", "engineered",
    "established", "evaluated", "expanded", "fine-tuned", "founded", "generated", "grew",
    "halved", "identified", "implemented", "improved", "increased", "integrated", "introduced",
    "investigated", "launched", "led", "lowered", "maintained", "managed", "mentored",
    "migrated", "modelled", "modeled", "negotiated", "optimised", "optimized", "orchestrated",
    "organised", "organized", "owned", "pioneered", "presented", "prototyped", "published",
    "rebuilt", "redesigned", "reduced", "refactored", "replaced", "resolved", "restructured",
    "revamped", "saved", "scaled", "secured", "shipped", "simplified", "spearheaded",
    "standardised", "standardized", "streamlined", "taught", "tested", "trained",
    "transformed", "tripled", "tuned", "unified", "upgraded", "wrote", "authored", "benchmarked",
}

# Openers that describe presence rather than outcome.
WEAK_OPENERS = (
    "responsible for", "helped", "helping", "assisted", "assisting", "worked on", "working on",
    "involved in", "participated in", "tasked with", "in charge of", "duties included",
    "did", "was", "were", "had", "used", "made", "handled", "dealt with", "exposure to",
    "familiar with", "contributed to", "supported",
)

_YEAR = r"(?:19|20)\d{2}"
DATE_NOISE_RE = re.compile(
    rf"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*'?\d{{2,4}}\b|\b{_YEAR}\s*[-–]\s*(?:{_YEAR}|present)\b|\b{_YEAR}\b",
    re.I,
)
METRIC_RE = re.compile(
    r"""(
        [$€£₹]\s?\d[\d,.]*\s?(?:k|m|mn|b|bn|million|billion|thousand|lakh|crore)?   # money
      | \d[\d,.]*\s?%                                         # percentages
      | \d[\d,.]*\s?(?:x|×)\b                                 # 3x
      | \b\d[\d,.]*\+?\s?(?:k|m|b|ms|s|sec|min|hrs?|hours|days|weeks|months|users|customers|
           requests|qps|rps|tps|gb|tb|pb|records|rows|engineers|people|students|teams|
           services|countries|clients|models|papers|stars|downloads)\b
      | \b\d{2,}[\d,]*\+?                                     # any 2+ digit number
      | \b\d+(?:\.\d+)?\s?(?:/|out\s+of\s+)\s?\d+              # 4.6/5
      | \bp9\d\b                                              # p95/p99 latency
    )""",
    re.I | re.X,
)


def opening_words(bullet: str, n: int = 3) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", bullet)
    return " ".join(words[:n]).lower()


def classify_opening(bullet: str) -> str:
    """'strong', 'weak' or 'neutral'."""
    opening = opening_words(bullet)
    if not opening:
        return "neutral"
    first = opening.split()[0]
    if any(opening == w or opening.startswith(w + " ") for w in WEAK_OPENERS):
        return "weak"
    if first in STRONG_VERBS:
        return "strong"
    # past-tense verbs we don't know yet are usually fine ("Instrumented", "Containerised")
    if first.endswith("ed") and len(first) > 4:
        return "strong"
    return "neutral"


def has_metric(bullet: str) -> bool:
    cleaned = DATE_NOISE_RE.sub(" ", bullet)
    return bool(METRIC_RE.search(cleaned))


@dataclass
class BulletReport:
    total: int = 0
    strong: int = 0
    weak: int = 0
    quantified: int = 0
    too_long: int = 0
    too_short: int = 0
    weak_examples: list[str] = field(default_factory=list)
    unquantified_examples: list[str] = field(default_factory=list)
    long_examples: list[str] = field(default_factory=list)

    @property
    def strong_ratio(self) -> float:
        return self.strong / self.total if self.total else 0.0

    @property
    def weak_ratio(self) -> float:
        return self.weak / self.total if self.total else 0.0

    @property
    def quantified_ratio(self) -> float:
        return self.quantified / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "strong_openers": self.strong,
            "weak_openers": self.weak,
            "quantified": self.quantified,
            "too_long": self.too_long,
            "too_short": self.too_short,
            "strong_ratio": round(self.strong_ratio, 3),
            "quantified_ratio": round(self.quantified_ratio, 3),
            "weak_examples": self.weak_examples,
            "unquantified_examples": self.unquantified_examples,
            "long_examples": self.long_examples,
        }


def analyse_bullets(bullets: list[str], max_examples: int = 5,
                    long_words: int = 35, short_words: int = 5) -> BulletReport:
    report = BulletReport(total=len(bullets))
    for b in bullets:
        kind = classify_opening(b)
        if kind == "strong":
            report.strong += 1
        elif kind == "weak":
            report.weak += 1
            if len(report.weak_examples) < max_examples:
                report.weak_examples.append(b)
        if has_metric(b):
            report.quantified += 1
        elif len(report.unquantified_examples) < max_examples:
            report.unquantified_examples.append(b)
        n = len(b.split())
        if n > long_words:
            report.too_long += 1
            if len(report.long_examples) < max_examples:
                report.long_examples.append(b)
        elif n < short_words:
            report.too_short += 1
    return report
