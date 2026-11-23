"""Evaluate matching methods on the benchmark.

    python -m benchmark.generate
    python -m benchmark.evaluate                    # spaCy vectors if installed
    python -m benchmark.evaluate --encoder sbert    # needs sentence-transformers

Each job description is a query; all 270 resumes are ranked for it. Hybrid
weights are picked by grid search on the 10 *dev* JDs and reported on the 20
*test* JDs, so the headline number isn't tuned on itself.

Writes benchmark/results/results_<encoder>.md and .json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from resume_analyzer.encoders import load_encoder
from resume_analyzer.matching import (
    BM25,
    POOL_WEIGHTS,
    MatchConfig,
    jd_skill_weights,
    keyword_recall,
    requirement_coverage,
    skill_coverage,
    tfidf_scores,
    v1_keyword_overlap,
)
from resume_analyzer.pipeline import ResumeAnalyzer
from resume_analyzer.sections import Requirement, jd_requirements, units

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"


# --- metrics -----------------------------------------------------------------

def dcg(rels: np.ndarray) -> float:
    return float(((2.0 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))).sum())


def ndcg_at(ranked_rels: np.ndarray, all_rels: np.ndarray, k: int) -> float:
    ideal = dcg(np.sort(all_rels)[::-1][:k])
    return dcg(ranked_rels[:k]) / ideal if ideal else 0.0


def average_precision(ranked_rels: np.ndarray, threshold: int = 2) -> float:
    hits = ranked_rels >= threshold
    if not hits.any():
        return 0.0
    precisions = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precisions[hits].mean())


def query_metrics(scores: np.ndarray, rels: np.ndarray, kinds: np.ndarray) -> dict[str, float]:
    # stable sort on -score with a fixed tie order (resume id) keeps runs reproducible
    order = np.argsort(-scores, kind="stable")
    r = rels[order]
    R = int((rels == 2).sum())
    top_r_kinds = kinds[order][:R]
    top_r_rels = r[:R]
    first = np.flatnonzero(r == 2)
    n_para = int(((rels == 2) & (kinds == "paraphrase")).sum())
    return {
        "nDCG@10": ndcg_at(r, rels, 10),
        "nDCG@50": ndcg_at(r, rels, 50),
        "MAP": average_precision(r),
        "MRR": 1.0 / (first[0] + 1) if len(first) else 0.0,
        "R-Prec": float((top_r_rels == 2).mean()) if R else 0.0,
        # how many keyword-stuffed resumes sneak into the top 10
        "Stuffed@10": float((kinds[order][:10] == "stuffed").mean()),
        # share of same-role paraphrase resumes that make it into the top R
        "ParaRecall@R": float(((top_r_kinds == "paraphrase") & (top_r_rels == 2)).sum() / n_para) if n_para else 0.0,
    }


def mean_metrics(per_query: list[dict]) -> dict[str, float]:
    return {k: float(np.mean([q[k] for q in per_query])) for k in per_query[0]}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """95% CI for mean(a - b) by resampling queries. Returns (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --- benchmark state -----------------------------------------------------------

class Bench:
    def __init__(self, encoder_kind: str | None = None, data: Path = DATA, encoder=None):
        self.resumes = [json.loads(line) for line in open(data / "resumes.jsonl", encoding="utf-8")]
        self.jds = [json.loads(line) for line in open(data / "jobs.jsonl", encoding="utf-8")]
        qrels: dict[tuple[str, str], int] = {}
        with open(data / "qrels.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                j, r, rel = line.split("\t")
                qrels[(j, r)] = int(rel)
        self.rels = {j["id"]: np.array([qrels.get((j["id"], r["id"]), 0) for r in self.resumes]) for j in self.jds}
        self.kinds = np.array([r["kind"] for r in self.resumes])

        self.encoder = encoder if encoder is not None else load_encoder(encoder_kind)
        self.analyzer = ResumeAnalyzer(encoder=self.encoder)
        self.tax = self.analyzer.taxonomy
        t = time.time()
        self.parsed = [self.analyzer.parse(r["text"]) for r in self.resumes]
        self.texts = [r["text"] for r in self.resumes]
        self.bm25 = BM25(self.texts)
        self.cfg = MatchConfig()
        self.reqs = {j["id"]: (jd_requirements(j["text"]) or [Requirement(u, 1.0) for u in units(j["text"])]) for j in self.jds}
        if self.encoder:
            self.unit_vecs = [self.encoder.encode(p.units) for p in self.parsed]
            self.req_vecs = {jid: self.encoder.encode([q.text for q in reqs]) for jid, reqs in self.reqs.items()}
            self.doc_vecs_trunc = self.encoder.encode([t[:512] for t in self.texts])
            self.doc_vecs_full = self.encoder.encode(self.texts)
            self.jd_vecs_trunc = {j["id"]: self.encoder.encode([j["text"][:512]])[0] for j in self.jds}
            self.jd_vecs_full = {j["id"]: self.encoder.encode([j["text"]])[0] for j in self.jds}
        self.prep_seconds = time.time() - t
        self._component_cache: dict = {}

    def components(self, jd: dict, cfg: MatchConfig) -> dict[str, np.ndarray]:
        key = (jd["id"], cfg.related_credit, cfg.listed_only_credit, cfg.use_families, cfg.use_evidence)
        if key in self._component_cache:
            return self._component_cache[key]
        jw = jd_skill_weights(jd["text"], self.tax, cfg.preferred_weight)
        reqs = self.reqs[jd["id"]]
        skills = np.array([skill_coverage(p.skills, jw, self.tax, cfg).score for p in self.parsed])
        kw = np.array([keyword_recall(t, reqs)[0] for t in self.texts])
        out = {"skills": skills, "keywords": kw}
        if self.encoder:
            out["dense"] = np.array([
                requirement_coverage(p.units, reqs, self.encoder, v, self.req_vecs[jd["id"]])[0]
                for p, v in zip(self.parsed, self.unit_vecs)
            ])
        self._component_cache[key] = out
        return out

    def hybrid(self, jd: dict, weights: dict[str, float], cfg: MatchConfig | None = None) -> np.ndarray:
        """Same formula as matching.combine(), vectorised over all resumes.
        A "tfidf" weight switches on pool mode (TF-IDF min-max scaled per query)."""
        comps = dict(self.components(jd, cfg or self.cfg))
        if weights.get("tfidf"):
            comps["tfidf"] = self.tfidf(jd)
        used = {k: w for k, w in weights.items() if w > 0 and k in comps}
        total = sum(used.values())
        return sum(w * comps[k] for k, w in used.items()) / total

    def tfidf(self, jd: dict) -> np.ndarray:
        key = ("tfidf", jd["id"])
        if key not in self._component_cache:
            s = tfidf_scores(self.texts, jd["text"])
            self._component_cache[key] = (s - s.min()) / ((s.max() - s.min()) or 1.0)
        return self._component_cache[key]

    def methods(self) -> dict:
        m = {
            "v1 keyword overlap (original repo)": lambda jd: np.array([v1_keyword_overlap(t, jd["text"]) for t in self.texts]),
            "TF-IDF cosine": lambda jd: tfidf_scores(self.texts, jd["text"]),
            "BM25": lambda jd: self.bm25.scores(jd["text"]),
            "Keyword recall": lambda jd: self.components(jd, self.cfg)["keywords"],
            "Skills: exact only": lambda jd: self.components(jd, replace(self.cfg, use_families=False, use_evidence=False))["skills"],
            "Skills: + families": lambda jd: self.components(jd, replace(self.cfg, use_evidence=False))["skills"],
            "Skills: + families + evidence": lambda jd: self.components(jd, self.cfg)["skills"],
        }
        if self.encoder:
            m["Dense: whole doc, 512 chars (v1)"] = lambda jd: self.doc_vecs_trunc @ self.jd_vecs_trunc[jd["id"]]
            m["Dense: whole doc, full text"] = lambda jd: self.doc_vecs_full @ self.jd_vecs_full[jd["id"]]
            m["Dense: requirement coverage"] = lambda jd: self.components(jd, self.cfg)["dense"]
        return m

    def hard_mask(self, jd: dict) -> np.ndarray:
        """The hard setting. Candidates are:
        - same-role resumes written in paraphrase (relevance 2),
        - ordinary adjacent-role resumes, which use the JD's exact skill names (1),
        - resumes stuffed with this JD's keywords (0).
        Ordinary same-role resumes are removed because every method finds them."""
        rels = self.rels[jd["id"]]
        stuffed_here = np.array([r["stuffed_for"] == jd["role"] for r in self.resumes])
        return ((rels == 2) & (self.kinds == "paraphrase")) | ((rels == 1) & (self.kinds == "normal")) | stuffed_here

    def evaluate(self, scorer, split: str, setting: str = "full") -> tuple[dict, list[dict]]:
        per_q = []
        for jd in self.jds:
            if jd["split"] != split:
                continue
            scores, rels, kinds = scorer(jd), self.rels[jd["id"]], self.kinds
            if setting == "hard":
                mask = self.hard_mask(jd)
                scores, rels, kinds = scores[mask], rels[mask], kinds[mask]
            per_q.append(query_metrics(scores, rels, kinds))
        return mean_metrics(per_q), per_q


def weight_grid(names: list[str], step: float = 0.1):
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 2)
    for combo in itertools.product(ticks, repeat=len(names)):
        if abs(sum(combo) - 1) < 1e-6:
            yield {n: float(c) for n, c in zip(names, combo)}


def tune(bench: Bench, names: list[str], metric: str = "nDCG@10", setting: str = "full") -> tuple[dict, float]:
    best, best_val = None, -1.0
    for w in weight_grid(names):
        val, _ = bench.evaluate(lambda jd, w=w: bench.hybrid(jd, w), "dev", setting)
        # ties -> prefer the more even split (less overfit to dev)
        if val[metric] > best_val + 1e-9 or (abs(val[metric] - best_val) < 1e-9 and np.std(list(w.values())) < np.std(list(best.values()))):
            best, best_val = w, val[metric]
    return best, best_val


def fmt_table(rows: list[tuple[str, dict]], cols: list[str]) -> str:
    head = "| Method | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    best = {c: (min if c == "Stuffed@10" else max)(r[1][c] for r in rows) for c in cols}
    body = ""
    for name, vals in rows:
        cells = []
        for c in cols:
            s = f"{vals[c]:.3f}"
            cells.append(f"**{s}**" if abs(vals[c] - best[c]) < 1e-9 else s)
        body += f"| {name} | " + " | ".join(cells) + " |\n"
    return head + body


def run_setting(bench: Bench, setting: str, cols: list[str]) -> tuple[list, list, dict, str]:
    rows, per_query = [], {}
    for name, fn in bench.methods().items():
        mean, pq = bench.evaluate(fn, "test", setting)
        rows.append((name, mean))
        per_query[name] = pq

    comp_names = ["skills", "dense", "keywords"] if bench.encoder else ["skills", "keywords"]
    hybrids = {
        "Hybrid (default weights)": MatchConfig().weights,
        "Hybrid (weights tuned on dev)": tune(bench, comp_names, setting=setting)[0],
        "Hybrid, pool mode (+ TF-IDF, tuned on dev)": tune(bench, comp_names + ["tfidf"], setting=setting)[0],
    }
    if bench.encoder:
        hybrids["Hybrid, pool mode (library default weights)"] = dict(POOL_WEIGHTS)
    for name, w in hybrids.items():
        mean, pq = bench.evaluate(lambda jd, w=w: bench.hybrid(jd, w), "test", setting)
        rows.append((name, mean))
        per_query[name] = pq

    ablations = []
    for drop in comp_names:
        rest = [c for c in comp_names if c != drop]
        w = tune(bench, rest, setting=setting)[0] if len(rest) > 1 else {rest[0]: 1.0}
        m, _ = bench.evaluate(lambda jd, w=w: bench.hybrid(jd, w), "test", setting)
        ablations.append((f"Hybrid without {drop}", m))
    # evidence weighting, measured on the skill matcher alone (the hybrid may
    # give skills zero weight, which would make this ablation a no-op)
    for credit in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        cfg = replace(bench.cfg, listed_only_credit=credit)
        m, _ = bench.evaluate(lambda jd, cfg=cfg: bench.components(jd, cfg)["skills"], "test", setting)
        ablations.append((f"Skills only, listed-only credit = {credit}", m))
    for rc in (0.0, 0.25, 0.5, 0.75):
        cfg = replace(bench.cfg, related_credit=rc)
        m, _ = bench.evaluate(lambda jd, cfg=cfg: bench.components(jd, cfg)["skills"], "test", setting)
        ablations.append((f"Skills only, related-skill credit = {rc}", m))

    lines = [fmt_table(rows, cols), "Weights picked on dev:", ""]
    lines += [f"- {k}: `{ {n: round(v, 2) for n, v in w.items()} }`" for k, w in hybrids.items()]
    lines.append("")
    # paired bootstrap on nDCG@10 for the comparisons that matter
    def ci(a_name, b_name):
        a = np.array([q["nDCG@10"] for q in per_query[a_name]])
        b = np.array([q["nDCG@10"] for q in per_query[b_name]])
        d, lo, hi = paired_bootstrap(a, b)
        return f"- {a_name} vs {b_name}: ΔnDCG@10 = {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]"
    lines.append("Paired bootstrap over test queries (10k resamples):")
    lines.append("")
    for a, b in [("Skills: + families + evidence", "Skills: exact only"),
                 ("Hybrid (weights tuned on dev)", "v1 keyword overlap (original repo)"),
                 ("Hybrid, pool mode (+ TF-IDF, tuned on dev)", "TF-IDF cosine"),
                 ("Hybrid, pool mode (+ TF-IDF, tuned on dev)", "BM25")]:
        lines.append(ci(a, b))
    lines.append("")
    return rows, ablations, hybrids, "\n".join(lines)


def seed_study(encoder_kind: str, seeds: list[int]) -> str:
    """Regenerate the benchmark with several seeds and report mean ± std of
    the key numbers, re-tuning hybrid weights on each seed's dev split."""
    import tempfile

    from .generate import Generator, write

    encoder = load_encoder(encoder_kind)
    keep = ["v1 keyword overlap (original repo)", "TF-IDF cosine", "BM25", "Skills: exact only",
            "Skills: + families + evidence", "Dense: whole doc, 512 chars (v1)", "Dense: requirement coverage"]
    collected: dict[tuple[str, str], list[dict]] = {}
    for seed in seeds:
        with tempfile.TemporaryDirectory() as tmp:
            g = Generator(seed)
            write(*g.build(), Path(tmp))
            bench = Bench("none", data=Path(tmp), encoder=encoder)
            comp_names = ["skills", "dense", "keywords"] if bench.encoder else ["skills", "keywords"]
            for setting in ("full", "hard"):
                scorers = {k: v for k, v in bench.methods().items() if k in keep}
                tuned = tune(bench, comp_names + ["tfidf"], setting=setting)[0]
                scorers["Hybrid, pool mode (+ TF-IDF, tuned on dev)"] = lambda jd, b=bench, w=tuned: b.hybrid(jd, w)
                if bench.encoder:
                    scorers["Hybrid, pool mode (library default weights)"] = lambda jd, b=bench: b.hybrid(jd, POOL_WEIGHTS)
                for name, fn in scorers.items():
                    collected.setdefault((setting, name), []).append(bench.evaluate(fn, "test", setting)[0])
        print(f"seed {seed} done")
    out = [f"# Results across generator seeds {seeds}", "",
           "Mean ± std over seeds, test split. Hybrid weights re-tuned on each seed's dev JDs.", ""]
    for setting, title in [("full", "Setting A (full pool)"), ("hard", "Setting B (hard pool)")]:
        out += [f"## {title}", "", "| Method | nDCG@10 | MAP | Stuffed@10 |", "|---|---:|---:|---:|"]
        for (st, name), runs in collected.items():
            if st != setting:
                continue
            cells = []
            for metric in ("nDCG@10", "MAP", "Stuffed@10"):
                vals = np.array([r[metric] for r in runs])
                cells.append(f"{vals.mean():.3f} ± {vals.std():.3f}")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="spacy", help="sbert | spacy | none")
    ap.add_argument("--seeds", type=int, nargs="*", help="run the multi-seed study instead, e.g. --seeds 13 7 42 101 2024")
    args = ap.parse_args()

    if args.seeds:
        text = seed_study(args.encoder, args.seeds)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"seeds_{args.encoder}.md").write_text(text, encoding="utf-8")
        print(text)
        return

    bench = Bench(args.encoder)
    enc_name = bench.encoder.name if bench.encoder else "none"
    n_test = sum(j["split"] == "test" for j in bench.jds)
    n_dev = len(bench.jds) - n_test
    print(f"{len(bench.resumes)} resumes, {len(bench.jds)} JDs, encoder={enc_name} (prep {bench.prep_seconds:.1f}s)")
    cols = ["nDCG@10", "nDCG@50", "MAP", "MRR", "R-Prec", "ParaRecall@R", "Stuffed@10"]

    md = [f"# Benchmark results ({enc_name})", "",
          f"{len(bench.resumes)} resumes, {n_test} test JDs (+{n_dev} dev JDs used only for picking weights). "
          "Generated by `python -m benchmark.evaluate`; don't edit by hand.", ""]
    dump = {"encoder": enc_name}
    for setting, title in [("full", "Setting A: rank the full pool (270 resumes)"),
                           ("hard", "Setting B: hard pool (paraphrased same-role vs. adjacent-role vs. keyword-stuffed)")]:
        rows, ablations, hybrids, text = run_setting(bench, setting, cols)
        md += [f"## {title}", "", text, "### Ablations", "", fmt_table(ablations, cols), ""]
        dump[setting] = {"main": dict(rows), "ablations": dict(ablations), "weights": hybrids}
        print(f"done: {setting}")

    RESULTS.mkdir(exist_ok=True)
    tag = enc_name.split(":")[0]
    out_md = RESULTS / f"results_{tag}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    (RESULTS / f"results_{tag}.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"wrote {out_md}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
