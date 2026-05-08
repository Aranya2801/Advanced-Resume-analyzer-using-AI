<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Advanced%20Resume%20Analyzer&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Powered%20by%20Claude%20AI%20%C2%B7%20NLP%20%C2%B7%20Semantic%20Search%20%C2%B7%20Vector%20Embeddings&descAlignY=55&descFontSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Claude AI](https://img.shields.io/badge/Claude_AI-claude--sonnet--4-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/Aranya2801/Advanced-Resume-analyzer-using-AI/ci.yml?style=for-the-badge&label=CI%2FCD)](https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI/actions)

<br/>

> **🏆 MIT-Grade AI system for comprehensive resume analysis, ATS optimisation,**
> **career path intelligence, and automated cover letter generation.**

<br/>

[🚀 Live Demo](#-quick-start) · [📖 Documentation](#-documentation) · [🐛 Report Bug](https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI/issues) · [✨ Request Feature](https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI/issues)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [⚡ Quick Start](#-quick-start)
- [🐳 Docker Deployment](#-docker-deployment)
- [🔌 REST API](#-rest-api)
- [🧠 AI Analysis Pipeline](#-ai-analysis-pipeline)
- [📊 Scoring System](#-scoring-system)
- [📁 Project Structure](#-project-structure)
- [📦 Dataset](#-dataset)
- [🧪 Testing](#-testing)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 AI-Powered Analysis
- **Claude Sonnet 4** deep resume review with executive-level feedback
- Semantic similarity scoring using **Sentence-BERT** embeddings
- Named Entity Recognition via **spaCy** for structured extraction
- Career trajectory prediction with personalised path recommendations

### 📊 Scoring Engine
- **Overall Score** (0–100) composite metric
- **ATS Compatibility** — tests against real ATS heuristics
- **Content Quality** — depth, clarity, structure
- **Impact Score** — achievements vs responsibilities ratio
- **Quantification Score** — metrics & numbers density
- **Action Verb Score** — power vs weak verb ratio

</td>
<td width="50%">

### 🔑 Keyword Intelligence
- Token-level keyword matching against job descriptions
- **Semantic similarity** beyond exact matches
- Gap analysis: matched vs missing keywords with priority ranking
- Industry-specific keyword database (5,000+ terms)

### 🛠️ Generation Tools
- **AI Bullet Rewriter** — transforms weak bullets into impact statements
- **Cover Letter Generator** — personalised, company-aware letters
- **Summary Rewriter** — executive-level professional summaries
- **Full JSON Report** — machine-readable analysis export

</td>
</tr>
</table>

### 🎯 Additional Capabilities

| Capability | Details |
|---|---|
| 📄 **Multi-format Parsing** | PDF (pdfplumber), DOCX (python-docx), TXT with OCR fallback |
| 📈 **Readability Analysis** | Flesch-Kincaid grade level, reading ease, sentence complexity |
| 🗺️ **Career Path Intel** | Salary estimates, market demand, skill investment recommendations |
| 🏆 **Benchmark Comparison** | Industry percentile scoring (entry/mid/senior/executive) |
| ⚡ **REST API** | FastAPI backend with `/analyse`, `/cover-letter`, `/improve-bullet` endpoints |
| 🐳 **Docker Ready** | Single command deployment with `docker compose up` |
| 🔒 **Privacy First** | Files processed in-memory, never stored or logged |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Advanced Resume Analyzer                        │
│                                                                     │
│   ┌──────────────┐    ┌──────────────────────────────────────────┐ │
│   │   Frontend   │    │              Core Pipeline                │ │
│   │  Streamlit   │───▶│                                          │ │
│   │  Dark UI     │    │  ┌────────────┐   ┌──────────────────┐  │ │
│   └──────────────┘    │  │  Document  │──▶│  Skill Extractor │  │ │
│                        │  │  Parser    │   │  (5,000+ terms)  │  │ │
│   ┌──────────────┐    │  │  PDF/DOCX  │   └──────────────────┘  │ │
│   │  FastAPI     │    │  │  TXT/OCR   │                         │ │
│   │  REST API    │───▶│  └────────────┘   ┌──────────────────┐  │ │
│   │  /analyse    │    │        │           │  ATS Scorer      │  │ │
│   │  /cover-     │    │        ▼           │  + Keyword Match │  │ │
│   │  letter      │    │  ┌────────────┐   └──────────────────┘  │ │
│   └──────────────┘    │  │ spaCy NER  │                         │ │
│                        │  │ Extraction │   ┌──────────────────┐  │ │
│                        │  └────────────┘   │ Sentence-BERT    │  │ │
│                        │        │           │ Semantic Sim.    │  │ │
│                        │        ▼           └──────────────────┘  │ │
│                        │  ┌──────────────────────────────────┐    │ │
│                        │  │        Claude AI Analysis         │    │ │
│                        │  │  • Deep review & scoring         │    │ │
│                        │  │  • Bullet rewriting              │    │ │
│                        │  │  • Career path prediction        │    │ │
│                        │  │  • Cover letter generation       │    │ │
│                        │  └──────────────────────────────────┘    │ │
│                        └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites

```bash
Python 3.11+
An Anthropic API key → https://console.anthropic.com
```

### 1. Clone the Repository

```bash
git clone https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI.git
cd Advanced-Resume-analyzer-using-AI
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 3. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Launch the App

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. 🎉

### 5. Generate Sample Dataset

```bash
python data/generate_dataset.py
# Creates sample resumes and job descriptions in data/
```

---

## 🐳 Docker Deployment

```bash
# Single command — build and run
docker compose up --build

# App:  http://localhost:8501
# API:  http://localhost:8000/docs
```

Or with Docker directly:

```bash
docker build -t resume-analyzer .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your-key resume-analyzer
```

---

## 🔌 REST API

The FastAPI backend exposes a full REST API. Interactive docs at `http://localhost:8000/docs`.

### Analyse Resume

```bash
curl -X POST http://localhost:8000/analyse \
  -F "file=@resume.pdf" \
  -F "api_key=sk-ant-..." \
  -F "job_description=We need a Python expert..." \
  -F "target_role=Senior ML Engineer"
```

**Response:**
```json
{
  "overall_score": 84,
  "ats_score": 91,
  "content_score": 78,
  "impact_score": 82,
  "strengths": ["Strong quantified achievements", "Relevant tech stack"],
  "weaknesses": ["Summary could be stronger"],
  "recommendations": [
    {
      "priority": "HIGH",
      "section": "Summary",
      "issue": "Too generic",
      "fix": "Lead with your most impressive metric",
      "example": "Engineer who reduced infra costs by $2.3M at Stripe..."
    }
  ],
  "rewritten_summary": "Staff Software Engineer with 8+ years...",
  "matched_keywords": ["Python", "Kubernetes", "distributed systems"],
  "missing_keywords": ["RLHF", "vLLM", "Kubeflow"],
  "career_insights": { ... }
}
```

### Generate Cover Letter

```bash
curl -X POST http://localhost:8000/cover-letter \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "...",
    "job_description": "...",
    "company": "Google",
    "api_key": "sk-ant-..."
  }'
```

### Improve a Bullet Point

```bash
curl -X POST http://localhost:8000/improve-bullet \
  -H "Content-Type: application/json" \
  -d '{
    "bullet": "Responsible for backend API development",
    "context": "Senior Software Engineer at fintech startup",
    "api_key": "sk-ant-..."
  }'
```

**Response:**
```json
{
  "original": "Responsible for backend API development",
  "improved": "Architected 12 RESTful microservices handling 50K req/sec, reducing P99 latency by 40%",
  "explanation": "Added ownership verb, scale metrics, and measurable performance impact",
  "impact_words": ["Architected", "reducing"]
}
```

---

## 🧠 AI Analysis Pipeline

```
Resume File (PDF/DOCX/TXT)
        │
        ▼
┌───────────────────┐
│  Document Parser  │  → Extract raw text, page count
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  NLP Extraction   │  → spaCy NER: names, orgs, dates, locations
│                   │  → Regex: email, phone, LinkedIn, GitHub
│                   │  → Section classifier: Experience/Education/Skills
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Skill Extraction │  → Match against 350+ skill taxonomy
│                   │  → Categorise: Languages/Cloud/ML/Databases
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  ATS Scoring      │  → Heuristic ATS simulation
│                   │  → Keyword token matching
│                   │  → Sentence-BERT semantic similarity
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Readability      │  → Flesch-Kincaid grade level
│                   │  → Action verb analysis (power vs weak)
│                   │  → Quantification density score
└───────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Claude AI Deep Analysis      │
│  • Overall/ATS/Content scores │
│  • Strengths & weaknesses     │
│  • Prioritised recommendations│
│  • Rewritten summary          │
│  • Career level estimation    │
│  • Keyword gap suggestions    │
└───────────────────────────────┘
        │
        ▼
┌───────────────────┐
│  Career Path AI   │  → Trajectory analysis
│                   │  → Salary range estimation
│                   │  → Skill investment advice
│                   │  → Market demand assessment
└───────────────────┘
        │
        ▼
  AnalysisResult (JSON)
```

---

## 📊 Scoring System

| Score | Range | Meaning |
|---|---|---|
| 🏆 Excellent | 80–100 | Ready to apply; minor polish only |
| ✅ Good | 65–79 | Strong but needs targeted improvements |
| ⚠️ Fair | 45–64 | Significant sections need rework |
| ❌ Poor | 0–44 | Major structural and content issues |

### Score Breakdown

```
Overall Score = weighted average of:
  ├── ATS Compatibility    (20%)  — parseable by automated systems
  ├── Content Quality      (25%)  — depth, relevance, structure
  ├── Impact Score         (25%)  — achievements vs. responsibilities
  ├── Quantification       (15%)  — numbers, percentages, $ amounts
  └── Action Verb Quality  (15%)  — power verbs, specificity
```

---

## 📁 Project Structure

```
Advanced-Resume-analyzer-using-AI/
│
├── 📂 src/
│   ├── analyzer.py          # Core analysis engine (DocumentParser, SkillExtractor, ClaudeAnalyser)
│   ├── app.py               # Streamlit web application
│   └── api.py               # FastAPI REST backend
│
├── 📂 data/
│   ├── generate_dataset.py  # Sample resume & JD generator
│   ├── sample_resumes/      # 3 sample resumes (PDF/TXT)
│   └── job_descriptions/    # 2 sample job descriptions
│
├── 📂 tests/
│   └── test_analyzer.py     # Pytest test suite (20+ tests)
│
├── 📂 .github/
│   └── workflows/ci.yml     # CI/CD: lint → test → security → docker
│
├── 📂 .streamlit/
│   └── config.toml          # Dark theme configuration
│
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── 📋 requirements.txt
├── 🔐 .env.example
└── 📖 README.md
```

---

## 📦 Dataset

The project includes a dataset generator for testing and development. Run:

```bash
python data/generate_dataset.py
```

This generates:

| File | Description |
|---|---|
| `sample_resumes/software_engineer_senior.txt` | Staff SWE resume (Stripe, Airbnb, Stanford) |
| `sample_resumes/data_scientist_mid.txt` | DS II resume (Spotify, AmEx, NeurIPS papers) |
| `sample_resumes/frontend_engineer_entry.txt` | Entry-level FE resume (BU grad, React/Next.js) |
| `job_descriptions/senior_ml_engineer_jd.txt` | Senior MLE JD ($250K–$320K) |
| `job_descriptions/frontend_engineer_jd.txt` | Frontend Engineer JD ($130K–$165K) |

Use these to test keyword matching, ATS scoring, and cover letter generation without needing your own documents.

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html

# Run specific test class
pytest tests/test_analyzer.py::TestDocumentParser -v
```

**Test Coverage:**

| Module | Tests |
|---|---|
| `DocumentParser` | email, phone, LinkedIn, GitHub extraction; section parsing; file type validation |
| `SkillExtractor` | language detection, cloud tools, databases, return types |
| `analyse_action_verbs` | power/weak detection, score range, empty input |
| `score_quantification` | number detection, edge cases, range validation |
| `ATSScorer` | score range, keyword matching |
| Integration | Full pipeline with mocked Claude AI |

---

## 🗺️ Roadmap

- [x] Core parsing engine (PDF, DOCX, TXT)
- [x] Claude AI deep analysis
- [x] ATS scoring & keyword matching
- [x] Semantic similarity with Sentence-BERT
- [x] Bullet point rewriter
- [x] Cover letter generator
- [x] Career path analysis
- [x] FastAPI REST backend
- [x] Docker deployment
- [x] CI/CD pipeline
- [ ] **LinkedIn profile importer** — scrape & analyse LinkedIn URLs
- [ ] **Multi-language support** — French, German, Spanish, Hindi
- [ ] **Resume template generator** — AI-designed ATS-optimised templates
- [ ] **Interview Q&A generator** — personalised interview prep from resume
- [ ] **Batch analysis dashboard** — HR teams can analyse 100+ resumes
- [ ] **Browser extension** — analyse resumes directly from LinkedIn/Indeed
- [ ] **Chrome Extension** — one-click analysis while browsing job boards
- [ ] **VS Code Extension** — analyse and edit resumes in your editor

---

## 🤝 Contributing

Contributions are what make open source amazing. Any contribution is **greatly appreciated**.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

Please follow the code style (black + isort) and add tests for new features.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

## 👤 Author

**Aranya2801**

[![GitHub](https://img.shields.io/badge/GitHub-Aranya2801-181717?style=for-the-badge&logo=github)](https://github.com/Aranya2801)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**⭐ Star this repo if it helped you land your dream job!**

</div>
