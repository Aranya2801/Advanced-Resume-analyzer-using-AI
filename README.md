# Resume Analyzer

Explainable resume scoring and resume-to-job matching, plus a benchmark to check whether the matching actually works.

[![CI](https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Give it a resume (PDF, DOCX or TXT) and, optionally, a job description. You get:

- a **quality score** broken into six parts (impact, quantification, structure, ATS issues, length, readability),
  each with the specific findings behind it;
- a **job match score** with matched / partially matched / missing skills, and each requirement in the JD shown
  next to the resume line that best supports it;
- a flag for skills that are **only listed** in the skills section and never shown in an actual bullet;
- **ranking** of many resumes against one JD;
- optional **LLM feedback** (bullet rewrites, written review, cover letter draft) if you add an API key. The scores
  don't depend on it.

Everything except the optional LLM part runs locally on CPU.

## Why this version exists

The first version of this repo was a typical "AI resume analyzer": keyword overlap, a sentence embedding of the
first 512 characters of each document, and an LLM asked to produce the overall score. When I went back to it I
found that the score changed between runs, the embedding mostly compared contact headers with company blurbs, a
regex bug counted almost every bullet as "quantified", and there was no way to tell if any of it was right.

So I rebuilt it around two ideas: every number should be **deterministic and explainable**, and every design
choice should be **measured against baselines** on a labelled benchmark. The full write-up is in
[`docs/report.md`](docs/report.md).

## Results

Ranking resumes for job descriptions on the synthetic benchmark (270 resumes, 20 test JDs, weights tuned on 10
separate dev JDs). **Setting A** ranks the full pool. **Setting B** is the hard part: same-role candidates who
describe their skills in different words vs. adjacent-role candidates who use the JD's exact keywords vs.
keyword-stuffed resumes.

| Method | A: nDCG@10 | A: stuffed in top 10 ↓ | B: nDCG@10 | B: stuffed in top 10 ↓ |
|---|---:|---:|---:|---:|
| v1 keyword overlap (original code) | 0.992 | 3.5% | 0.860 | 8.5% |
| BM25 | 0.998 | 3.0% | 0.917 | 13.0% |
| TF-IDF cosine | 0.997 | 2.5% | 0.942 | 11.0% |
| Skill match, exact | 0.744 | 28.0% | 0.406 | 30.0% |
| Skill match + evidence weighting | 0.972 | 5.5% | 0.583 | 27.0% |
| Hybrid, pool mode | 0.995 | 4.0% | **0.947** | 11.0% |

What I took from this:

- **Plain skill matching is easy to game.** Weighting skills by where they appear (a bullet vs. just the skills
  list) cuts stuffed resumes in the top 10 from 28% to 5.5% and improves nDCG@10 by +0.23 (95% CI +0.19 to +0.27).
- **TF-IDF over the candidate pool is a very strong baseline.** The pool-mode hybrid beats BM25 on the hard setting
  (+0.030, CI +0.004 to +0.063) but is *not* significantly better than TF-IDF. Its real advantage is that it can
  tell you *why* a candidate ranked where they did.
- **The skill taxonomy can't see what it doesn't know**, which shows up clearly on paraphrased resumes. That's
  the main argument for keeping a lexical or dense signal next to it.

These are numbers on synthetic data I generated myself. The report has a section on why that matters
(and what I did to make it less circular). Dense results use spaCy word vectors; Sentence-BERT is supported but I
haven't published numbers for it yet.

## Quick start

Python 3.10+.

**Windows (PowerShell)**

```powershell
git clone https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI.git
cd Advanced-Resume-analyzer-using-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_md
streamlit run app.py
```

**macOS / Linux**

```bash
git clone https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI.git
cd Advanced-Resume-analyzer-using-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
streamlit run app.py
```

For better semantic matching, also `pip install sentence-transformers` (pulls in PyTorch, ~2 GB). The app picks it
up automatically. To turn on LLM feedback, copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (or
`OPENAI_API_KEY`; any OpenAI-compatible server works, including a local Ollama).

### Command line

```bash
python -m resume_analyzer analyze data/samples/resumes/data_scientist_mid.txt --jd data/samples/jobs/senior_ml_engineer.txt
python -m resume_analyzer analyze my_resume.pdf --json > report.json
python -m resume_analyzer rank "resumes/*.pdf" --jd job.txt
```

Example output:

```
data_scientist_mid.txt - quality 79/100
  impact             78   8/13 bullets open with a strong action verb.
                         1 bullet(s) open with passive phrasing such as 'responsible for' or 'helped'.
  quantification     42   4/13 bullets contain a number (%, $, users, latency, ...).
                         Aim for roughly half of your bullets to show scale or outcome with a number.
  ...
  Job match 37/100 (skills 38, dense 38, keywords 31)
  matched: Machine Learning, Mentoring, Research, MLflow, PyTorch, TensorFlow
  partial: Fine-tuning (have BERT), LLM (have BERT), RLHF (have BERT), Transformers (have BERT), Kubeflow (have MLflow), MLOps (have MLflow)
  missing: Distributed Systems, Kubernetes, Model Serving, System Design, Go, ONNX, Python, Rust, Statistics, TensorRT, vLLM
```

### REST API

```bash
uvicorn api:app --reload          # http://localhost:8000/docs
```

| Endpoint | What it does |
|---|---|
| `POST /analyze` | upload a resume (+ optional `job_description` form field) → full report |
| `POST /match` | JSON `{resume_text, job_description}` → match report |
| `POST /rank` | several files + `job_description` → ranked list |
| `POST /llm/rewrite-bullet`, `POST /llm/cover-letter` | only if an LLM key is set on the server |
| `GET /health` | status, which encoder is loaded, whether LLM is on |

API keys live in the server's environment and are never sent in requests.

### Docker

```bash
docker compose up --build         # app on :8501, API on :8000
```

## How the matching works

The match score combines three signals that fail in different ways:

1. **Skill coverage.** ~200 skills with aliases (`k8s` → Kubernetes) and families. A required skill found in an
   experience/project bullet gets full credit, one that's only in the skills list gets 0.6, and a missing skill
   with a sibling present (TensorFlow for PyTorch) gets half credit. Nice-to-have skills count half.
2. **Requirement coverage.** Each JD requirement line is embedded and matched to the most similar resume line.
   This replaces embedding the first 512 characters of each document.
3. **Keyword recall.** The share of the JD's content words found in the resume, after removing filler like
   "experience" and "strong".

When ranking a pool, TF-IDF similarity across the pool is added as a fourth signal. Weights and the reasoning
behind each choice are in [`docs/report.md`](docs/report.md).

The quality score is a fixed rubric. The weights are my own judgement calls, not learned from data, and they're
all in `resume_analyzer/scoring.py` if you want to change them.

## Reproducing the benchmark

```bash
python -m benchmark.generate                    # writes benchmark/data/ (seeded, deterministic)
python -m benchmark.evaluate --encoder spacy    # ~25 s on CPU -> benchmark/results/results_spacy.md
python -m benchmark.evaluate --encoder sbert    # if sentence-transformers is installed
python -m benchmark.evaluate --encoder spacy --seeds 13 7 42 101 2024   # stability across seeds
```

## Project layout

```
resume_analyzer/
  parsing.py      PDF/DOCX/TXT -> text, contact details
  sections.py     section splitting, JD requirement extraction, date ranges -> months of experience
  skills.py       taxonomy loading and skill extraction (data/skills.json)
  bullets.py      opening-verb and metric checks
  matching.py     skill coverage, requirement coverage, keyword recall, baselines
  encoders.py     sentence-transformers / spaCy vectors / none
  scoring.py      quality rubric
  llm.py          optional feedback via Anthropic or OpenAI-compatible APIs
  pipeline.py     ResumeAnalyzer: analyze(), match(), rank()
benchmark/        data generator, evaluation, results
app.py            Streamlit UI
api.py            FastAPI
tests/            pytest (no network or model downloads needed)
docs/report.md    method, benchmark design, results, limitations
```

## Limitations

- The benchmark is synthetic and the labels are role-based (no seniority, no recruiter judgement). A small
  human-labelled set is the next thing I want to add.
- Skill extraction is dictionary-based. Anything outside the taxonomy is invisible to the skill score (the dense and
  keyword signals partly cover for it).
- Scanned PDFs aren't supported (no OCR). The app says so instead of returning an empty report.
- English only.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## License

MIT, see [LICENSE](LICENSE).
