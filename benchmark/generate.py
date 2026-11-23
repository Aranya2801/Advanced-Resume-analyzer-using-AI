"""Generate the synthetic resume/JD benchmark.

    python -m benchmark.generate            # writes benchmark/data/
    python -m benchmark.generate --seed 7   # a different draw

Why synthetic: public resume datasets either have no job descriptions, no
relevance labels, or licences that don't allow redistribution. Generating
the data lets the labels be exact and lets me plant specific failure cases:

* ``normal``     - ordinary resume for its role.
* ``paraphrase`` - same role, but skills are mostly described in words the
                   taxonomy doesn't know, and the skills section is tiny.
                   Tests whether matching survives vocabulary mismatch.
* ``stuffed``    - experience is from an unrelated role, but the skills
                   section is padded with a target role's keywords. Tests
                   whether matching can be gamed.

The obvious weakness - the same person wrote the generator and the
matcher - is discussed in docs/report.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from resume_analyzer.skills import DATA_PATH, default_taxonomy

from .roles import COMPANIES, DEGREES, FIRST_NAMES, LAST_NAMES, PARAPHRASES, ROLES, SCHOOLS, TECH_ROLES, relevance

OUT = Path(__file__).parent / "data"

DUTIES = {
    "ml_engineer": ["Train, evaluate and deploy machine learning models to production",
                    "Build and own the ML platform: feature pipelines, training infrastructure and model serving",
                    "Work with data scientists to productionise research models",
                    "Monitor model quality and latency in production"],
    "data_scientist": ["Design experiments and analyse their results for product teams",
                       "Build predictive models for churn, pricing and demand",
                       "Turn ambiguous business questions into analyses and metrics",
                       "Communicate findings to product and leadership"],
    "data_engineer": ["Build and maintain batch and streaming data pipelines",
                      "Model warehouse tables for analytics and reporting",
                      "Own data quality, lineage and pipeline reliability",
                      "Optimise storage and compute costs of the data platform"],
    "backend_engineer": ["Design and build scalable backend services and APIs",
                         "Own service reliability, performance and on-call",
                         "Work on data models, caching and database performance",
                         "Review code and contribute to technical design"],
    "frontend_engineer": ["Build responsive, accessible web interfaces",
                          "Work closely with designers to turn specs into polished UI",
                          "Improve web performance and Core Web Vitals",
                          "Write unit and end-to-end tests for the web app"],
    "fullstack_engineer": ["Ship features end to end across frontend, API and database",
                           "Build internal tools and customer-facing pages",
                           "Design APIs and database schemas for new products",
                           "Own deployment and monitoring of the features you build"],
    "devops_engineer": ["Run and scale our production infrastructure and clusters",
                        "Manage infrastructure as code and CI/CD pipelines",
                        "Improve observability, alerting and incident response",
                        "Reduce cloud costs and improve reliability"],
    "nlp_scientist": ["Research and develop new methods for language understanding and generation",
                      "Train and fine-tune large language models",
                      "Publish at top venues and collaborate with academic partners",
                      "Build evaluation benchmarks for model quality and safety"],
    "data_analyst": ["Build dashboards and reports for business teams",
                     "Analyse sales, marketing and customer data",
                     "Present insights and recommendations to stakeholders",
                     "Maintain the reporting data model and metric definitions"],
    "qa_engineer": ["Design and automate regression and end-to-end test suites",
                    "Integrate automated tests into the CI pipeline",
                    "Triage and track defects with engineering teams",
                    "Improve release quality and test reliability"],
}

REQ_TEMPLATES = [
    "{y}+ years of experience with {s}",
    "Strong hands-on experience with {s} and {s2}",
    "Proficiency in {s}",
    "Production experience with {s}",
    "Solid understanding of {s}",
    "Experience using {s} in a team setting",
]
PREF_TEMPLATES = ["Experience with {s} is a plus", "Familiarity with {s}", "Exposure to {s} and {s2}"]


def _fill(template: str, rng: random.Random, skill: str | None = None) -> str:
    vals = {
        "pct": rng.randint(8, 45), "pct2": rng.choice([95, 98, 99, 99.9]), "n": rng.randint(3, 60),
        "n2": rng.randint(2, 40), "ms": rng.randint(120, 600), "ms2": rng.randint(20, 110),
        "d": rng.randint(5, 21), "d2": rng.randint(1, 4), "k": rng.randint(40, 900),
        "m": rng.randint(5, 30), "x": rng.randint(2, 6), "s": round(rng.uniform(2.5, 5.0), 1),
        "s2": round(rng.uniform(0.8, 2.2), 1), "auc": round(rng.uniform(0.81, 0.95), 2),
        "r": round(rng.uniform(1.0, 4.5), 1), "skill": skill or "", "skill2": "{skill2}",
    }
    return template.format(**vals)


class Generator:
    def __init__(self, seed: int = 13):
        self.rng = random.Random(seed)
        self.tax = default_taxonomy()
        self.paraphrases = self._checked_paraphrases()
        self.aliases = self._aliases()

    def _checked_paraphrases(self) -> dict[str, list[str]]:
        ok = {}
        for skill, phrases in PARAPHRASES.items():
            kept = [p for p in phrases if skill not in self.tax.find(p)]
            if kept:
                ok[skill] = kept
        return ok

    def _aliases(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        for group, fams in raw.items():
            if group.startswith("_"):
                continue
            for entries in fams.values():
                for canonical, *aliases in entries:
                    forms = [a.lstrip("=") for a in aliases if len(a.lstrip("=")) > 2]
                    out[canonical] = forms
        return out

    def surface(self, skill: str, mode: str) -> str:
        """How a resume refers to a skill. mode: 'normal' or 'paraphrase'."""
        r = self.rng.random()
        if mode == "paraphrase" and skill in self.paraphrases and r < 0.85:
            return self.rng.choice(self.paraphrases[skill])
        if mode == "normal" and skill in self.paraphrases and r < 0.15:
            return self.rng.choice(self.paraphrases[skill])
        if self.aliases.get(skill) and self.rng.random() < 0.3:
            return self.rng.choice(self.aliases[skill])
        return skill

    def _bullet(self, template: tuple[str, list[str]], mode: str, weak_rate: float) -> str:
        text, options = template
        skill = self.surface(self.rng.choice(options), mode) if options else ""
        b = _fill(text, self.rng, skill)
        if "{skill2}" in text:
            b = b.replace("{skill2}", "CSS")
        if self.rng.random() < weak_rate:
            # "Built a churn model..." -> "Worked on a churn model..."
            opener = self.rng.choice(["Responsible for", "Helped with", "Worked on", "Involved in"])
            words = b.split(" ")
            cut = 1
            if len(words) > 3 and words[1] == "and":  # "Trained and deployed ..."
                cut = 3
            elif words[1] == "up":  # "Set up ..."
                cut = 2
            b = f"{opener} {' '.join(words[cut:])}"
        return b

    def resume(self, idx: int, role: str, kind: str, stuff_role: str | None = None) -> dict:
        rng = self.rng
        spec = ROLES[role]
        mode = "paraphrase" if kind == "paraphrase" else "normal"
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        years = rng.randint(1, 12)
        title = rng.choice(spec["titles"])
        weak_rate = rng.choice([0.0, 0.1, 0.3])

        lines = [name.upper(), f"{name.split()[0].lower()}.{name.split()[1].lower()}@mail.com | +1 555 {rng.randint(100,999)} {rng.randint(1000,9999)}", ""]
        lines += ["SUMMARY", f"{title} with {years} year{'s' if years > 1 else ''} of experience. " +
                  rng.choice(["Enjoys working on hard problems with a small team.",
                              "Looking for a role with more ownership.",
                              "Comfortable working across teams and time zones."]), ""]

        lines.append("EXPERIENCE")
        n_jobs = 1 if years < 3 else rng.randint(2, 3)
        end_year, remaining = 2026, years
        for j in range(n_jobs):
            span = max(1, remaining // (n_jobs - j))
            start_year = end_year - span
            end = "Present" if j == 0 else f"Jun {end_year}"
            lines.append(f"{rng.choice(spec['titles'])} | {rng.choice(COMPANIES)} | Jan {start_year} - {end}")
            for template in rng.sample(spec["bullets"], min(len(spec["bullets"]), rng.randint(3, 5))):
                lines.append("• " + self._bullet(template, mode, weak_rate))
            lines.append("")
            end_year, remaining = start_year, remaining - span

        lines.append("EDUCATION")
        degree_kind = "tech" if role in TECH_ROLES else "nontech"
        lines.append(f"{rng.choice(DEGREES[degree_kind])}, {rng.choice(SCHOOLS)}, {2026 - years - rng.randint(0, 2)}")
        lines.append("")

        if kind == "paraphrase":
            listed = rng.sample(spec["core"], 2)
        else:
            listed = rng.sample(spec["core"], max(3, int(len(spec["core"]) * 0.7)))
            listed += rng.sample(spec["optional"], min(len(spec["optional"]), rng.randint(2, 4)))
            noise_role = rng.choice([r for r in TECH_ROLES if r != role])
            listed += rng.sample(ROLES[noise_role]["core"], 1)
        if kind == "stuffed":
            target = ROLES[stuff_role]
            listed += target["core"] + rng.sample(target["optional"], 4)
        listed = list(dict.fromkeys(listed))
        rng.shuffle(listed)
        lines += ["SKILLS", ", ".join(listed)]

        return {"id": f"r{idx:03d}", "role": role, "kind": kind, "stuffed_for": stuff_role,
                "text": "\n".join(lines).strip() + "\n"}

    def jd(self, idx: int, role: str) -> dict:
        rng = self.rng
        spec = ROLES[role]
        company = rng.choice(COMPANIES)
        title = rng.choice(spec["titles"])
        lines = [f"{title} - {company}", "", "About the role",
                 f"{company} is growing and we're hiring a {title} to join a small, senior team.", "",
                 "Responsibilities"]
        lines += [f"- {d}" for d in rng.sample(DUTIES[role], 3)]
        lines += ["", "Requirements"]
        core = rng.sample(spec["core"], len(spec["core"]))
        i = 0
        while i < len(core):
            t = rng.choice(REQ_TEMPLATES)
            if "{s2}" in t and i + 1 < len(core):
                lines.append("- " + t.format(y=rng.randint(2, 6), s=core[i], s2=core[i + 1]))
                i += 2
            else:
                lines.append("- " + t.replace(" and {s2}", "").format(y=rng.randint(2, 6), s=core[i]))
                i += 1
        lines += ["", "Nice to have"]
        opt = rng.sample(spec["optional"], 4)
        lines.append("- " + rng.choice(PREF_TEMPLATES).format(s=opt[0], s2=opt[1]))
        lines.append("- " + rng.choice(PREF_TEMPLATES).format(s=opt[2], s2=opt[3]))
        lines += ["", "We offer", "- Competitive salary and equity", "- Remote-friendly, flexible hours"]
        return {"id": f"j{idx:02d}", "role": role, "text": "\n".join(lines) + "\n"}

    def build(self, per_role: int = 24, jds_per_role: int = 3):
        resumes, idx = [], 0
        for role in TECH_ROLES:
            for k in range(per_role):
                kind = "paraphrase" if k % 4 == 3 else "normal"
                resumes.append(self.resume(idx, role, kind))
                idx += 1
        # Stuffed resumes: for every target role, two non-technical people and
        # one engineer from an unrelated tech role, all padding their skills
        # section with the target role's keywords.
        for target in TECH_ROLES:
            unrelated = [r for r in TECH_ROLES if relevance(target, r) == 0]
            for bg in ("marketing_manager", "operations_manager", self.rng.choice(unrelated)):
                resumes.append(self.resume(idx, bg, "stuffed", stuff_role=target))
                idx += 1

        jds = []
        for role in TECH_ROLES:
            for k in range(jds_per_role):
                j = self.jd(len(jds), role)
                j["split"] = "dev" if k == 0 else "test"
                jds.append(j)
        return resumes, jds


def write(resumes, jds, out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "resumes.jsonl", "w", encoding="utf-8") as f:
        for r in resumes:
            f.write(json.dumps(r) + "\n")
    with open(out / "jobs.jsonl", "w", encoding="utf-8") as f:
        for j in jds:
            f.write(json.dumps(j) + "\n")
    with open(out / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("jd_id\tresume_id\trelevance\n")
        for j in jds:
            for r in resumes:
                rel = relevance(j["role"], r["role"])
                if rel:
                    f.write(f"{j['id']}\t{r['id']}\t{rel}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--per-role", type=int, default=24)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    g = Generator(args.seed)
    resumes, jds = g.build(args.per_role)
    write(resumes, jds, args.out)
    kinds = {k: sum(r["kind"] == k for r in resumes) for k in ("normal", "paraphrase", "stuffed")}
    dropped = sum(len(v) for v in PARAPHRASES.values()) - sum(len(v) for v in g.paraphrases.values())
    print(f"{len(resumes)} resumes {kinds}, {len(jds)} job descriptions -> {args.out}")
    print(f"{dropped} paraphrase(s) dropped because the taxonomy already recognises them")


if __name__ == "__main__":
    main()
