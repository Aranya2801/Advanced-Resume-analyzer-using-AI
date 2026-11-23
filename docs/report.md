# Explainable resume–job matching: what actually helps?

*Technical notes for v0.3 of this repo. All numbers come from `benchmark/results/` and can be reproduced with two commands (see the end).*

## 1. Motivation

The first version of this project did what most resume-analyzer demos do: extract text, count keyword overlap with the
job description, embed both documents with a sentence model, and ask an LLM for an "overall score". Looking at it
again, three things bothered me:

1. **The headline score wasn't reproducible.** The overall score came straight from the LLM, so the same resume could
   get different scores on different runs, and nothing explained *why* it got that number.
2. **The semantic similarity was mostly comparing headers.** Both documents were truncated to 512 characters before
   embedding, which for a resume is the name, contact line and summary, and for a JD is usually the company blurb.
3. **There was no way to tell whether any of it worked.** No labelled data, no baseline, no metric.

This version tries to fix all three: every score is deterministic and comes with the evidence behind it, matching
works at the level of individual requirements, and there's a benchmark with baselines, a dev/test split and
significance tests.

The main questions:

- **Q1.** Does a taxonomy-based skill matcher (which is easy to explain to a user) rank candidates as well as
  standard lexical retrieval (TF-IDF, BM25)?
- **Q2.** How easy is each method to game by keyword stuffing, and does weighting skills by *where* they appear help?
- **Q3.** Does requirement-level dense matching beat whole-document embedding?
- **Q4.** Does combining the signals beat the best single one?

## 2. Method

### 2.1 Parsing and sections

PDF (pdfplumber) and DOCX (python-docx, including tables, because many templates put the header in a table) are
converted to text. Sections are found by matching short lines against a heading vocabulary (`sections.py`); the old
code's `re.fullmatch` approach missed common headings like "WORK EXPERIENCE" and "Technical Skills:". Job
descriptions are split the same way into responsibilities / requirements / nice-to-have / benefits; benefits are
dropped, and nice-to-have lines (or lines containing cues like "a plus") get weight 0.5.

### 2.2 Skill coverage

A hand-built taxonomy (`resume_analyzer/data/skills.json`, ~200 skills) maps aliases to canonical names (`k8s` →
Kubernetes, `golang` → Go) and groups skills into families. Short or ambiguous names (`Go`, `R`, `C`) only match
case-sensitively and with custom boundaries so "go to", "R&D" and "D.C." don't count.

For a JD with skill weights *w_s* (1.0 required, 0.5 preferred), the coverage score is

    coverage = Σ_s w_s · credit(s) / Σ_s w_s

where `credit(s)` is

| situation | credit |
|---|---|
| skill appears in an experience / project / publication bullet | 1.0 |
| skill appears **only** in the skills list ("listed-only") | 0.6 |
| skill missing, but a sibling from the same family is present (PyTorch ↔ TensorFlow) | 0.5 × sibling's credit |
| otherwise | 0 |

The *evidence* rule is the part I care most about. Anyone can type "Kubernetes" into a skills list; a bullet that
says what was done with it is stronger evidence. Families where members aren't really substitutes (programming
languages, soft skills, ...) are excluded from partial credit.

### 2.3 Requirement coverage (dense)

Each JD requirement line *q* is embedded and compared with every resume line *r* (bullets, sentences, skills lines;
the contact header is skipped). The score is the weighted mean, over requirements, of the best match:

    dense = Σ_q w_q · rescale(max_r cos(q, r)) / Σ_q w_q

`rescale` subtracts an encoder-specific floor (cosine similarity between unrelated texts is well above 0 for
averaged word vectors). The best-matching resume line is shown to the user next to each requirement, which turned
out to be the most useful part of the UI.

### 2.4 Keyword recall and the hybrid

Keyword recall is the weighted share of the JD's content words (stop words and JD filler like "experience",
"strong", "team" removed, light suffix stripping) that appear in the resume.

The **single-resume score** is a weighted mean of skills, dense and keyword recall. Default weights (0.5 / 0.3 / 0.2)
were chosen by hand *before* running the benchmark, and are reported as-is.

**Pool mode** (ranking many resumes for one JD) adds TF-IDF cosine computed over the pool, min-max scaled within
the pool. It isn't available for a single resume because it needs corpus statistics.

### 2.5 Resume quality rubric

Independent of any JD: impact (strong vs weak bullet openers), quantification (bullets with a number; years and
date ranges don't count), structure, ATS red flags, concision and readability (Flesch, computed locally). The
weights are judgement calls, not learned, and the rubric isn't evaluated against human ratings; it's there to give
concrete, checkable feedback, and it's deterministic.

Two bugs from v1 are worth mentioning because they inflated everything: weak phrases were matched as substrings
anywhere in a bullet (so "Kubernetes **was** migrated" and "s**had**ow traffic" were "weak"), and the "contains a
number" regex accepted a bare comma, so nearly every bullet counted as quantified.

## 3. Benchmark

### 3.1 Why synthetic

I couldn't find a public dataset that has resumes, job descriptions, relevance labels *and* a licence that allows
redistribution. So `benchmark/generate.py` creates one from role profiles (`benchmark/roles.py`): 10 technical
roles, plus two non-technical backgrounds used only for stuffed resumes.

- **270 resumes**: 180 *normal*, 60 *paraphrase* (same role, but skills mostly described in words the taxonomy
  doesn't know, e.g. "container orchestration on GKE" instead of Kubernetes, and only two skills listed), 30
  *stuffed* (experience from an unrelated role, skills section padded with a target role's keywords).
- **30 JDs**: 3 per role, each with responsibilities, required skills, nice-to-have skills and benefits. 10 are
  **dev** (used only to pick hybrid weights), 20 are **test**.
- **Labels**: 2 = same role, 1 = adjacent role (e.g. ML engineer ↔ data scientist), 0 = otherwise. Stuffed resumes
  are labelled by their *real* background.

The generator checks every paraphrase against the taxonomy and drops the ones it recognises, so paraphrase resumes
really are out-of-vocabulary for the skill matcher.

### 3.2 Settings and metrics

- **Setting A, full pool:** rank all 270 resumes for each test JD.
- **Setting B, hard pool:** only the cases that are hard to separate: same-role *paraphrase* resumes (rel 2),
  *normal* adjacent-role resumes (rel 1, and these use the JD's exact skill names), and resumes stuffed for this
  role (rel 0). Normal same-role resumes are removed because every method finds them.

Metrics: nDCG@10 (graded), MAP (rel 2), MRR, R-precision, **ParaRecall@R** (share of same-role paraphrase resumes in
the top R), and **Stuffed@10** (share of the top 10 that are stuffed resumes; lower is better). Differences are
tested with a paired bootstrap over test queries (10k resamples).

## 4. Results

Encoder: spaCy `en_core_web_md` (averaged word vectors). Sentence-BERT couldn't be downloaded in the environment
where these were produced; `python -m benchmark.evaluate --encoder sbert` runs the same thing with it.

### 4.1 Main table (test split, seed 13)

| Method | A: nDCG@10 | A: MAP | A: Stuffed@10 ↓ | B: nDCG@10 | B: MAP | B: ParaRecall@R | B: Stuffed@10 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 keyword overlap (original) | 0.992 | 0.943 | 0.035 | 0.860 | 0.762 | 0.717 | 0.085 |
| TF-IDF cosine | 0.997 | **0.990** | 0.025 | 0.942 | 0.937 | **0.908** | 0.110 |
| BM25 | **0.998** | 0.980 | 0.030 | 0.917 | 0.900 | 0.850 | 0.130 |
| Skills, exact match only | 0.744 | 0.750 | 0.280 | 0.406 | 0.303 | 0.225 | 0.300 |
| Skills + families | 0.723 | 0.751 | 0.285 | 0.434 | 0.332 | 0.300 | 0.300 |
| Skills + families + evidence | 0.972 | 0.878 | 0.055 | 0.583 | 0.461 | 0.350 | 0.270 |
| Dense, whole doc, 512 chars (v1) | 0.549 | 0.413 | **0.015** | 0.766 | 0.566 | 0.467 | **0.010** |
| Dense, requirement coverage | 0.687 | 0.553 | 0.170 | 0.713 | 0.567 | 0.492 | 0.095 |
| Hybrid, single-resume (default weights) | 0.981 | 0.909 | 0.045 | 0.644 | 0.538 | 0.400 | 0.265 |
| Hybrid, single-resume (tuned on dev) | 0.988 | 0.931 | 0.065 | 0.799 | 0.722 | 0.683 | 0.170 |
| **Hybrid, pool mode (tuned on dev)** | 0.995 | 0.978 | 0.040 | **0.947** | **0.938** | 0.892 | 0.110 |

Full tables, including ablations, are in `benchmark/results/results_spacy.md`. Across five generator seeds the
picture is the same (std of nDCG@10 ≤ 0.04 for every method; `benchmark/results/seeds_spacy.md`).

### 4.2 Answers

**Q1: skill matching vs lexical retrieval.** On its own, the taxonomy matcher is clearly worse than TF-IDF/BM25,
and badly so on paraphrased resumes (B: nDCG@10 0.58 vs 0.94). That's expected (it can't see skills it doesn't
know) but it's worth stating plainly, because "skill extraction + overlap" is what most resume tools show users.
Its value is explanation (which skills are missing, which are only claimed), not ranking.

**Q2: keyword stuffing.** Exact skill matching puts stuffed resumes in 28% of top-10 slots. Evidence weighting cuts
that to 5.5% and raises nDCG@10 by **+0.227** (95% CI [+0.188, +0.270]) in setting A and **+0.177** ([+0.138,
+0.215]) in setting B. The ablation over listed-only credit shows the trade-off directly: at credit 1.0 (no evidence
rule) Stuffed@10 is 0.285, at 0.0 it is 0.040, but recall of honest candidates who only list some skills drops.
Values between 0.4 and 0.6 work best in setting A; the default of 0.6 was set beforehand and wasn't changed after
seeing test results.

Family-level partial credit hurts slightly in setting A (0.744 → 0.723) and helps in setting B (0.406 → 0.434). It
matters most for paraphrase resumes, where a sibling skill is often the only thing the matcher can see.

**Q3: requirement-level vs whole-document embeddings.** With averaged word vectors, requirement coverage beats
whole-document embedding in setting A (0.687 vs 0.549) but not in B (0.713 vs 0.766). Looking at the failures: the
first 512 characters of a resume contain the summary line with the person's job title, which on this data is
almost a label. That explains why the truncated version is the best method on Stuffed@10: stuffed resumes have
honest job titles. It's a real signal (recruiters read titles first too), but it's also a property of how the
generator writes summaries, so I don't read too much into it. A job-title matching component is an obvious next
step.

Averaged GloVe-style vectors are a weak encoder; I expect the dense numbers to move most with Sentence-BERT.

**Q4: does the hybrid help?** In single-resume mode, no. The hybrid with hand-set weights is worse than plain
keyword overlap, and even with dev-tuned weights it's statistically indistinguishable from v1 in setting A and
significantly *worse* in setting B (ΔnDCG@10 = −0.062, CI [−0.124, −0.006]), because the skill component drags it
down on paraphrase resumes. In pool mode, the tuned hybrid is the best method in setting B and beats BM25
(+0.030, CI [+0.004, +0.063]), but it is **not** significantly better than TF-IDF alone in either setting.

So the honest summary: for *ranking*, TF-IDF over the pool is a very strong baseline, and the hybrid matches it
while adding explanations. The weights tuned in setting A (used as the library default for pool mode) don't
transfer well to setting B (0.880 vs 0.947), which is a reminder that weights tuned on one candidate distribution
may not hold on another.

## 5. Threats to validity

- **Same author for generator and matcher.** I tried to limit this (paraphrases are checked to be outside the
  taxonomy, stuffed resumes target the skill matcher's weak spot, the strongest baselines are standard IR methods
  I didn't design), but the benchmark is surely easier for my methods in ways I can't see. Results on real hiring
  data could look different.
- **Templates.** Each role has 4–8 bullet templates. Resumes of the same role share a lot of vocabulary, which helps
  lexical methods; real resumes vary much more.
- **Labels are role-based.** Seniority, location and the "can this person do the job" judgement a recruiter makes are
  not modelled. A strong junior and a strong senior get the same label.
- **Small query set.** 20 test JDs. Confidence intervals are reported for that reason.
- **The quality rubric is not validated** against human judgements.

## 6. Next steps

- Run with Sentence-BERT and a cross-encoder re-ranker on the hard setting.
- A job-title component (normalised titles, title families).
- A small human-labelled set: ~50 real (anonymised, consented) resume–JD pairs rated by recruiters, to check
  whether the synthetic conclusions survive.
- Learn the taxonomy aliases from data (e.g. ESCO / O*NET skill lists) instead of maintaining them by hand.

## Reproduce

```bash
python -m benchmark.generate                       # writes benchmark/data/
python -m benchmark.evaluate --encoder spacy       # ~25 s on a laptop CPU
python -m benchmark.evaluate --encoder spacy --seeds 13 7 42 101 2024
```
