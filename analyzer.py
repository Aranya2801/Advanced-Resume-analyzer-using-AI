"""
╔══════════════════════════════════════════════════════════════════╗
║          ADVANCED RESUME ANALYZER - Core Engine                  ║
║          Powered by Claude AI + NLP + Vector Embeddings          ║
╚══════════════════════════════════════════════════════════════════╝

Author: Aranya2801
License: MIT
"""

from __future__ import annotations

import re
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

import anthropic
import spacy
import pdfplumber
import docx
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_kincaid_grade, flesch_reading_ease


# ─────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────

@dataclass
class PersonalInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    location: str = ""
    website: str = ""


@dataclass
class WorkExperience:
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_months: int = 0
    description: str = ""
    achievements: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)


@dataclass
class Education:
    institution: str = ""
    degree: str = ""
    field: str = ""
    gpa: float = 0.0
    graduation_year: int = 0
    honors: list[str] = field(default_factory=list)
    relevant_courses: list[str] = field(default_factory=list)


@dataclass
class ResumeData:
    raw_text: str = ""
    personal_info: PersonalInfo = field(default_factory=PersonalInfo)
    summary: str = ""
    work_experience: list[WorkExperience] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    skills: dict[str, list[str]] = field(default_factory=dict)  # category → skills
    certifications: list[str] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    publications: list[str] = field(default_factory=list)
    awards: list[str] = field(default_factory=list)
    volunteer: list[str] = field(default_factory=list)
    word_count: int = 0
    page_count: int = 0
    parsed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AnalysisResult:
    resume_data: ResumeData
    overall_score: float = 0.0
    ats_score: float = 0.0
    content_score: float = 0.0
    format_score: float = 0.0
    impact_score: float = 0.0
    keyword_match_score: float = 0.0
    skills_gap: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    rewritten_summary: str = ""
    bullet_improvements: list[dict] = field(default_factory=list)
    career_insights: dict = field(default_factory=dict)
    industry_benchmarks: dict = field(default_factory=dict)
    readability_stats: dict = field(default_factory=dict)
    action_verb_analysis: dict = field(default_factory=dict)
    quantification_score: float = 0.0
    ai_feedback: str = ""
    job_match_percentage: float = 0.0


# ─────────────────────────────────────────────
#  Document Parser
# ─────────────────────────────────────────────

class DocumentParser:
    """Parse PDF, DOCX, and TXT resume files with high fidelity."""

    SECTION_PATTERNS = {
        "summary": r"(?i)(summary|profile|objective|about\s*me|professional\s*summary)",
        "experience": r"(?i)(experience|work\s*history|employment|career|professional\s*background)",
        "education": r"(?i)(education|academic|qualification|degree|university|college)",
        "skills": r"(?i)(skills|technical\s*skills|competencies|technologies|tools|expertise)",
        "projects": r"(?i)(projects|portfolio|work\s*samples|key\s*projects)",
        "certifications": r"(?i)(certifications?|licenses?|credentials|training)",
        "awards": r"(?i)(awards?|honors?|achievements?|recognition|accomplishments?)",
        "publications": r"(?i)(publications?|papers?|research|articles?)",
        "languages": r"(?i)(languages?|linguistic)",
        "volunteer": r"(?i)(volunteer|community|social|extracurricular)",
    }

    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
    PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
    LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.I)
    GITHUB_RE = re.compile(r"github\.com/[\w\-]+", re.I)
    URL_RE = re.compile(r"https?://[^\s]+")
    GPA_RE = re.compile(r"GPA[:\s]*(\d\.\d+)", re.I)
    YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
    DATE_RE = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|"
        r"March|April|June|July|August|September|October|November|December)"
        r"[\s,]*\d{2,4}",
        re.I,
    )

    def parse_pdf(self, path: Path) -> tuple[str, int]:
        """Extract text and page count from PDF."""
        text_parts = []
        page_count = 0
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                text_parts.append(page_text)
        return "\n".join(text_parts), page_count

    def parse_docx(self, path: Path) -> tuple[str, int]:
        """Extract text from DOCX."""
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        estimated_pages = max(1, len(" ".join(paragraphs).split()) // 500)
        return "\n".join(paragraphs), estimated_pages

    def parse_txt(self, path: Path) -> tuple[str, int]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        estimated_pages = max(1, len(text.split()) // 500)
        return text, estimated_pages

    def parse(self, path: Path) -> tuple[str, int]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.parse_pdf(path)
        elif suffix in (".docx", ".doc"):
            return self.parse_docx(path)
        elif suffix == ".txt":
            return self.parse_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def extract_personal_info(self, text: str) -> PersonalInfo:
        lines = text.split("\n")
        info = PersonalInfo()
        # Name: assume first non-empty line with title case and no special tokens
        for line in lines[:8]:
            stripped = line.strip()
            if stripped and len(stripped.split()) <= 5 and not self.EMAIL_RE.search(stripped):
                if stripped.istitle() or stripped.isupper():
                    info.name = stripped
                    break

        emails = self.EMAIL_RE.findall(text)
        if emails:
            info.email = emails[0]

        phones = self.PHONE_RE.findall(text)
        if phones:
            info.phone = re.sub(r"\s+", " ", phones[0]).strip()

        linkedin = self.LINKEDIN_RE.search(text)
        if linkedin:
            info.linkedin = "https://" + linkedin.group()

        github = self.GITHUB_RE.search(text)
        if github:
            info.github = "https://" + github.group()

        return info

    def extract_sections(self, text: str) -> dict[str, str]:
        """Split resume text into labelled sections."""
        lines = text.split("\n")
        sections: dict[str, list[str]] = {"header": []}
        current_section = "header"

        for line in lines:
            matched = False
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.fullmatch(pattern, line.strip()):
                    current_section = section_name
                    sections.setdefault(current_section, [])
                    matched = True
                    break
            if not matched:
                sections.setdefault(current_section, []).append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items()}


# ─────────────────────────────────────────────
#  Skill Extractor
# ─────────────────────────────────────────────

class SkillExtractor:
    """Extract and categorise technical and soft skills."""

    SKILL_DB: dict[str, list[str]] = {
        "Programming Languages": [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
            "Kotlin", "Swift", "R", "MATLAB", "Scala", "Ruby", "PHP", "Perl",
            "Haskell", "Lua", "Julia", "Dart", "Elixir",
        ],
        "Web Technologies": [
            "React", "Vue", "Angular", "Next.js", "Nuxt", "Svelte", "HTML5", "CSS3",
            "Tailwind", "Bootstrap", "Webpack", "Vite", "Node.js", "Express", "Django",
            "Flask", "FastAPI", "Spring Boot", "Laravel", "GraphQL", "REST API",
        ],
        "AI & Machine Learning": [
            "TensorFlow", "PyTorch", "Keras", "scikit-learn", "XGBoost", "LightGBM",
            "Hugging Face", "LangChain", "LlamaIndex", "OpenAI", "Claude", "GPT-4",
            "BERT", "Transformer", "NLP", "Computer Vision", "Reinforcement Learning",
            "MLflow", "Weights & Biases", "Ray", "ONNX", "TensorRT",
        ],
        "Data Engineering": [
            "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Flink", "Databricks",
            "Snowflake", "BigQuery", "Redshift", "duckDB", "Pandas", "Polars",
            "NumPy", "ETL", "Data Pipeline", "Data Warehouse", "Data Lake",
        ],
        "Cloud & DevOps": [
            "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
            "Jenkins", "GitHub Actions", "CircleCI", "ArgoCD", "Helm", "Prometheus",
            "Grafana", "ELK", "Datadog", "New Relic", "Istio",
        ],
        "Databases": [
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
            "DynamoDB", "Neo4j", "InfluxDB", "CockroachDB", "SQLite", "Oracle",
            "ChromaDB", "Pinecone", "Weaviate", "Qdrant",
        ],
        "Soft Skills": [
            "Leadership", "Communication", "Problem Solving", "Teamwork", "Agile",
            "Scrum", "Mentoring", "Project Management", "Critical Thinking",
            "Adaptability", "Creativity", "Time Management",
        ],
    }

    def extract(self, text: str) -> dict[str, list[str]]:
        text_lower = text.lower()
        found: dict[str, list[str]] = {}
        for category, skills in self.SKILL_DB.items():
            matched = [s for s in skills if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower)]
            if matched:
                found[category] = matched
        return found


# ─────────────────────────────────────────────
#  Action Verb Analyser
# ─────────────────────────────────────────────

POWER_VERBS = {
    "led", "built", "designed", "architected", "developed", "launched", "deployed",
    "optimised", "optimized", "reduced", "increased", "improved", "scaled",
    "delivered", "created", "established", "pioneered", "spearheaded", "drove",
    "managed", "mentored", "collaborated", "negotiated", "streamlined",
    "automated", "implemented", "engineered", "transformed", "accelerated",
    "achieved", "generated", "saved", "cut", "boosted", "surpassed",
}

WEAK_VERBS = {
    "responsible for", "helped", "assisted", "worked on", "involved in",
    "participated in", "did", "made", "was", "had", "used",
}


def analyse_action_verbs(text: str) -> dict:
    bullets = re.findall(r"[•\-\*]\s*(.+)", text)
    power_count = 0
    weak_count = 0
    weak_bullets = []
    strong_bullets = []

    for bullet in bullets:
        first_word = bullet.strip().split()[0].lower() if bullet.strip() else ""
        if any(first_word.startswith(v) for v in POWER_VERBS):
            power_count += 1
            strong_bullets.append(bullet.strip())
        elif any(phrase in bullet.lower() for phrase in WEAK_VERBS):
            weak_count += 1
            weak_bullets.append(bullet.strip())

    total = power_count + weak_count or 1
    return {
        "power_verb_ratio": round(power_count / total, 2),
        "power_count": power_count,
        "weak_count": weak_count,
        "total_bullets": len(bullets),
        "weak_bullets": weak_bullets[:5],
        "strong_bullets": strong_bullets[:5],
        "score": min(100, int((power_count / total) * 100)),
    }


def score_quantification(text: str) -> float:
    """Reward bullets that contain numbers/percentages/$ amounts."""
    bullets = re.findall(r"[•\-\*]\s*(.+)", text)
    if not bullets:
        return 0.0
    quantified = sum(
        1 for b in bullets
        if re.search(r"\b\d+[%$KMBx]?\b|\$[\d,]+|[\d,]+\+?", b)
    )
    return round(quantified / len(bullets) * 100, 1)


# ─────────────────────────────────────────────
#  ATS Scorer
# ─────────────────────────────────────────────

class ATSScorer:
    """Score resume against ATS systems and job descriptions."""

    ATS_RED_FLAGS = [
        "tables", "columns", "headers", "footers", "text boxes",
        "graphics", "images", "special characters",
    ]

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def keyword_match(self, resume_text: str, jd_text: str) -> dict:
        # Simple token-level matching
        jd_words = set(re.findall(r"\b\w{4,}\b", jd_text.lower()))
        resume_words = set(re.findall(r"\b\w{4,}\b", resume_text.lower()))

        matched = jd_words & resume_words
        missing = jd_words - resume_words

        # Filter to likely meaningful words (skip stop words)
        stopwords = {
            "this", "that", "with", "have", "from", "they", "will", "your",
            "been", "were", "more", "also", "some", "than", "such", "when",
            "then", "each", "their", "other", "into", "about", "over", "after",
        }
        matched = {w for w in matched if w not in stopwords}
        missing = {w for w in missing if w not in stopwords}

        score = len(matched) / (len(jd_words) or 1) * 100
        return {
            "matched": sorted(matched)[:30],
            "missing": sorted(missing)[:30],
            "score": round(score, 1),
        }

    def semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        emb_resume = self.embedder.encode([resume_text[:512]])
        emb_jd = self.embedder.encode([jd_text[:512]])
        sim = cosine_similarity(emb_resume, emb_jd)[0][0]
        return round(float(sim) * 100, 1)

    def ats_score(self, resume_text: str) -> float:
        score = 100.0
        # Penalise common ATS-unfriendly patterns
        if re.search(r"\|", resume_text):
            score -= 5
        if len(re.findall(r"[^\x00-\x7F]", resume_text)) > 10:
            score -= 8
        lines = resume_text.split("\n")
        short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 20)
        if short_lines / (len(lines) or 1) > 0.4:
            score -= 10
        return max(0, round(score, 1))


# ─────────────────────────────────────────────
#  Claude AI Analyser
# ─────────────────────────────────────────────

class ClaudeAnalyser:
    """Use Claude AI for deep resume intelligence."""

    SYSTEM_PROMPT = """You are an elite resume coach and talent acquisition expert with 20+ years of experience 
at top-tier companies like Google, McKinsey, and Goldman Sachs. You have reviewed over 50,000 resumes.

Your analysis is:
- Brutally honest but constructive
- Quantitative with specific scores (0-100)
- Rich with actionable, specific suggestions
- Industry-aware and role-specific
- ATS-optimised

Always return valid JSON matching the requested schema exactly. No markdown, no commentary outside JSON."""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def deep_analyse(self, resume_text: str, job_description: str = "", target_role: str = "") -> dict:
        context = ""
        if job_description:
            context += f"\n\nJOB DESCRIPTION:\n{job_description[:2000]}"
        if target_role:
            context += f"\n\nTARGET ROLE: {target_role}"

        prompt = f"""Analyse this resume thoroughly and return a JSON object with EXACTLY this structure:

{{
  "overall_score": <int 0-100>,
  "ats_compatibility": <int 0-100>,
  "content_quality": <int 0-100>,
  "impact_score": <int 0-100>,
  "strengths": ["<specific strength 1>", "<specific strength 2>", "<specific strength 3>"],
  "critical_weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
  "recommendations": [
    {{"priority": "HIGH", "section": "<section>", "issue": "<issue>", "fix": "<specific fix>", "example": "<example>"}},
    {{"priority": "HIGH", "section": "<section>", "issue": "<issue>", "fix": "<specific fix>", "example": "<example>"}},
    {{"priority": "MEDIUM", "section": "<section>", "issue": "<issue>", "fix": "<specific fix>", "example": "<example>"}}
  ],
  "rewritten_summary": "<A powerful 3-4 sentence professional summary rewritten for impact>",
  "career_level": "<Entry/Mid/Senior/Executive>",
  "estimated_yoe": <int years of experience>,
  "top_industries": ["<industry1>", "<industry2>", "<industry3>"],
  "missing_sections": ["<section1>", "<section2>"],
  "keyword_suggestions": ["<kw1>", "<kw2>", "<kw3>", "<kw4>", "<kw5>"],
  "quantification_feedback": "<specific feedback on numbers/metrics usage>",
  "executive_summary": "<2-sentence summary of this candidate for a hiring manager>"
}}

RESUME:
{resume_text[:3500]}{context}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            system=self.SYSTEM_PROMPT,
        )
        raw = response.content[0].text.strip()
        # Strip potential markdown fences
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    def improve_bullet(self, bullet: str, context: str = "") -> dict:
        prompt = f"""Rewrite this resume bullet point to be more impactful.
Return JSON: {{"original": "<original>", "improved": "<improved version with metrics>", "explanation": "<why improved>", "impact_words": ["<word1>", "<word2>"]}}

BULLET: {bullet}
CONTEXT: {context or "Software/Technology role"}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            system=self.SYSTEM_PROMPT,
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    def generate_cover_letter(self, resume_text: str, job_description: str, company: str) -> str:
        prompt = f"""Write a compelling, personalised cover letter for this candidate applying to {company}.
Use insights from both the resume and job description.
Make it specific, authentic, and impactful. 3 paragraphs max.

RESUME (excerpt):
{resume_text[:1500]}

JOB DESCRIPTION:
{job_description[:1000]}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def career_path_analysis(self, resume_text: str) -> dict:
        prompt = f"""Analyse this resume and suggest career paths. Return JSON:
{{
  "current_trajectory": "<current career path>",
  "recommended_paths": [
    {{"path": "<career path>", "fit_score": <int 0-100>, "steps": ["<step1>", "<step2>"], "timeline": "<timeline>"}},
    {{"path": "<career path>", "fit_score": <int 0-100>, "steps": ["<step1>", "<step2>"], "timeline": "<timeline>"}}
  ],
  "skill_investments": ["<skill to learn 1>", "<skill to learn 2>", "<skill to learn 3>"],
  "salary_range": {{"min": <int>, "max": <int>, "currency": "USD"}},
  "market_demand": "<High/Medium/Low>",
  "differentiation_advice": "<advice on standing out>"
}}

RESUME:
{resume_text[:2000]}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)


# ─────────────────────────────────────────────
#  Main Analyser Orchestrator
# ─────────────────────────────────────────────

class ResumeAnalyzer:
    """Orchestrates the full analysis pipeline."""

    def __init__(self, anthropic_api_key: str | None = None):
        self.parser = DocumentParser()
        self.skill_extractor = SkillExtractor()
        self.ats_scorer = ATSScorer()
        self.claude = ClaudeAnalyser(anthropic_api_key)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        logger.info("ResumeAnalyzer initialised successfully")

    def parse_resume(self, file_path: Path | str) -> ResumeData:
        path = Path(file_path)
        logger.info(f"Parsing resume: {path.name}")
        text, pages = self.parser.parse(path)

        data = ResumeData(
            raw_text=text,
            page_count=pages,
            word_count=len(text.split()),
            personal_info=self.parser.extract_personal_info(text),
            skills=self.skill_extractor.extract(text),
        )
        return data

    def full_analysis(
        self,
        file_path: Path | str,
        job_description: str = "",
        target_role: str = "",
    ) -> AnalysisResult:
        """Run the complete analysis pipeline."""
        resume_data = self.parse_resume(file_path)
        text = resume_data.raw_text

        # Readability
        readability = {
            "flesch_reading_ease": round(flesch_reading_ease(text), 1),
            "grade_level": round(flesch_kincaid_grade(text), 1),
            "word_count": resume_data.word_count,
            "page_count": resume_data.page_count,
            "avg_words_per_sentence": round(resume_data.word_count / max(1, text.count(".")), 1),
        }

        # Action verbs
        verb_analysis = analyse_action_verbs(text)

        # Quantification
        quant_score = score_quantification(text)

        # ATS
        ats_raw = self.ats_scorer.ats_score(text)

        # Keyword match (if JD provided)
        kw_match = {}
        semantic_sim = 0.0
        if job_description:
            kw_match = self.ats_scorer.keyword_match(text, job_description)
            semantic_sim = self.ats_scorer.semantic_similarity(text, job_description)

        # Claude AI deep analysis
        logger.info("Running Claude AI deep analysis...")
        ai_data = self.claude.deep_analyse(text, job_description, target_role)

        # Career path
        logger.info("Running career path analysis...")
        career = self.claude.career_path_analysis(text)

        # Assemble result
        result = AnalysisResult(
            resume_data=resume_data,
            overall_score=ai_data.get("overall_score", 0),
            ats_score=min(ats_raw, ai_data.get("ats_compatibility", ats_raw)),
            content_score=ai_data.get("content_quality", 0),
            impact_score=ai_data.get("impact_score", 0),
            keyword_match_score=kw_match.get("score", 0),
            matched_keywords=kw_match.get("matched", []),
            missing_keywords=kw_match.get("missing", []),
            skills_gap=[],
            strengths=ai_data.get("strengths", []),
            weaknesses=ai_data.get("critical_weaknesses", []),
            recommendations=ai_data.get("recommendations", []),
            rewritten_summary=ai_data.get("rewritten_summary", ""),
            career_insights=career,
            readability_stats=readability,
            action_verb_analysis=verb_analysis,
            quantification_score=quant_score,
            ai_feedback=ai_data.get("executive_summary", ""),
            job_match_percentage=semantic_sim,
        )

        logger.success(f"Analysis complete. Overall score: {result.overall_score}/100")
        return result

    def improve_bullets(self, result: AnalysisResult, n: int = 5) -> list[dict]:
        """Get AI-improved versions of weak bullet points."""
        weak = result.action_verb_analysis.get("weak_bullets", [])[:n]
        improved = []
        for bullet in weak:
            try:
                imp = self.claude.improve_bullet(bullet)
                improved.append(imp)
            except Exception as e:
                logger.warning(f"Bullet improvement failed: {e}")
        result.bullet_improvements = improved
        return improved

    def generate_cover_letter(
        self, result: AnalysisResult, job_description: str, company: str
    ) -> str:
        return self.claude.generate_cover_letter(
            result.resume_data.raw_text, job_description, company
        )

    def to_dict(self, result: AnalysisResult) -> dict:
        """Serialise AnalysisResult to JSON-compatible dict."""
        from dataclasses import asdict
        return asdict(result)
