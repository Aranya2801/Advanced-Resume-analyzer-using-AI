"""Skill extraction against a small hand-built taxonomy.

Each skill has a canonical name, aliases ("k8s" -> Kubernetes) and a family
(PyTorch and TensorFlow are both ``dl_frameworks``). Families are what let the
matcher give partial credit instead of treating "knows TensorFlow, JD asks for
PyTorch" the same as "knows nothing relevant".
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "skills.json"


@dataclass(frozen=True)
class Skill:
    name: str
    group: str
    family: str


@dataclass
class SkillHit:
    skill: Skill
    count: int = 0
    sections: set[str] = field(default_factory=set)


def _alias_pattern(alias: str) -> re.Pattern:
    case_sensitive = alias.startswith("=")
    alias = alias.lstrip("=")
    body = re.escape(alias).replace(r"\ ", r"[\s\-]+")
    # \b doesn't work for things like "C++" or ".NET", so spell out what
    # may not touch the match on either side.
    after = r"(?![\w+#&])" if len(alias) > 1 else r"(?![\w+#&.])"
    pattern = rf"(?<![\w+#.]){body}{after}"
    return re.compile(pattern, 0 if case_sensitive else re.I)


class SkillTaxonomy:
    def __init__(self, path: Path | str = DATA_PATH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.skills: dict[str, Skill] = {}
        self._patterns: list[tuple[re.Pattern, Skill]] = []
        self.families: dict[str, list[str]] = defaultdict(list)
        self.no_partial: set[str] = set(raw.get("_no_partial", []))
        for group, families in raw.items():
            if group.startswith("_"):
                continue
            for family, entries in families.items():
                for entry in entries:
                    canonical, *aliases = entry
                    skill = Skill(canonical, group, family)
                    self.skills[canonical.lower()] = skill
                    self.families[family].append(canonical)
                    # if the canonical name itself is case-sensitive it is listed as an alias with '='
                    forms = aliases if any(a.startswith("=") and a[1:] == canonical for a in aliases) else [canonical, *aliases]
                    for form in forms:
                        self._patterns.append((_alias_pattern(form), skill))
        # longest aliases first so "Spring Boot" is consumed before "Boot"-like fragments
        self._patterns.sort(key=lambda p: -len(p[0].pattern))

    def __len__(self) -> int:
        return len(self.skills)

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name.lower())

    def find(self, text: str) -> dict[str, int]:
        """Canonical skill name -> number of mentions."""
        counts: dict[str, int] = defaultdict(int)
        taken: list[tuple[int, int]] = []
        for pattern, skill in self._patterns:
            for m in pattern.finditer(text):
                span = m.span()
                if any(s < span[1] and span[0] < e for s, e in taken):
                    continue  # already covered by a longer alias
                taken.append(span)
                counts[skill.name] += 1
        return dict(counts)

    def extract(self, sections: dict[str, str]) -> dict[str, SkillHit]:
        """Find skills per resume section, so we know *where* a skill shows up.

        A skill only listed under "Skills" is weaker evidence than one used in
        an experience or project bullet; the matcher uses that distinction.
        """
        hits: dict[str, SkillHit] = {}
        for section, text in sections.items():
            for name, n in self.find(text).items():
                hit = hits.setdefault(name, SkillHit(self.skills[name.lower()]))
                hit.count += n
                hit.sections.add(section)
        return hits

    def related(self, a: str, b: str) -> bool:
        sa, sb = self.get(a), self.get(b)
        return bool(sa and sb and sa.name != sb.name and sa.family == sb.family
                    and sa.family not in self.no_partial)

    def substitutable(self, family: str) -> bool:
        return family not in self.no_partial

    def by_group(self, names) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for n in sorted(names, key=str.lower):
            s = self.get(n)
            if s:
                out[s.group].append(s.name)
        return dict(out)


@lru_cache(maxsize=1)
def default_taxonomy() -> SkillTaxonomy:
    return SkillTaxonomy()
