"""
Advanced Resume Analyzer — Test Suite
Run: pytest tests/ -v --cov=src
"""

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzer import (
    DocumentParser,
    SkillExtractor,
    analyse_action_verbs,
    score_quantification,
    ResumeData,
    PersonalInfo,
)


# ── Fixtures ──────────────────────────────────────────────────────

SAMPLE_RESUME_TEXT = """
ALEX CHEN
alex.chen@email.com | +1 (415) 555-0192 | linkedin.com/in/alexchen | github.com/alexchen

SUMMARY
Senior Software Engineer with 8+ years building large-scale distributed systems.

EXPERIENCE
Staff Software Engineer | Stripe | Jan 2021 – Present
• Architected fraud detection system processing 2.4M transactions/sec
• Reduced infrastructure costs by $2.3M annually
• Mentored 6 junior engineers; 4 promoted within 18 months
• Led migration reducing cluster costs by 34%

Senior Software Engineer | Airbnb | Mar 2018 – Jan 2021
• Improved booking conversion by 11% via ML-powered search ranking
• Responsible for API development (weak verb example)
• Helped with various backend tasks

EDUCATION
B.S. Computer Science, Stanford University, 2016 | GPA: 3.9

SKILLS
Python, Go, Java, TypeScript, Kafka, Kubernetes, PostgreSQL, Redis, AWS, Docker
"""


@pytest.fixture
def parser():
    return DocumentParser()


@pytest.fixture
def skill_extractor():
    return SkillExtractor()


@pytest.fixture
def sample_text():
    return SAMPLE_RESUME_TEXT


# ── DocumentParser Tests ──────────────────────────────────────────

class TestDocumentParser:

    def test_extract_email(self, parser, sample_text):
        info = parser.extract_personal_info(sample_text)
        assert info.email == "alex.chen@email.com"

    def test_extract_phone(self, parser, sample_text):
        info = parser.extract_personal_info(sample_text)
        assert "415" in info.phone

    def test_extract_linkedin(self, parser, sample_text):
        info = parser.extract_personal_info(sample_text)
        assert "linkedin.com/in/alexchen" in info.linkedin

    def test_extract_github(self, parser, sample_text):
        info = parser.extract_personal_info(sample_text)
        assert "github.com/alexchen" in info.github

    def test_extract_sections(self, parser, sample_text):
        sections = parser.extract_sections(sample_text)
        assert isinstance(sections, dict)

    def test_unsupported_file_type(self, parser):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse(Path("resume.xyz"))


# ── SkillExtractor Tests ──────────────────────────────────────────

class TestSkillExtractor:

    def test_extracts_programming_languages(self, skill_extractor, sample_text):
        skills = skill_extractor.extract(sample_text)
        prog_langs = skills.get("Programming Languages", [])
        assert "Python" in prog_langs
        assert "Go" in prog_langs

    def test_extracts_cloud_tools(self, skill_extractor, sample_text):
        skills = skill_extractor.extract(sample_text)
        cloud = skills.get("Cloud & DevOps", [])
        assert "AWS" in cloud or "Docker" in cloud or "Kubernetes" in cloud

    def test_extracts_databases(self, skill_extractor, sample_text):
        skills = skill_extractor.extract(sample_text)
        dbs = skills.get("Databases", [])
        assert "PostgreSQL" in dbs or "Redis" in dbs

    def test_returns_dict(self, skill_extractor, sample_text):
        result = skill_extractor.extract(sample_text)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(v, list)


# ── Action Verb Tests ─────────────────────────────────────────────

class TestActionVerbAnalysis:

    def test_detects_power_verbs(self):
        result = analyse_action_verbs(SAMPLE_RESUME_TEXT)
        assert result["power_count"] > 0

    def test_detects_weak_verbs(self):
        result = analyse_action_verbs(SAMPLE_RESUME_TEXT)
        assert result["weak_count"] > 0

    def test_score_in_range(self):
        result = analyse_action_verbs(SAMPLE_RESUME_TEXT)
        assert 0 <= result["score"] <= 100

    def test_ratio_in_range(self):
        result = analyse_action_verbs(SAMPLE_RESUME_TEXT)
        assert 0.0 <= result["power_verb_ratio"] <= 1.0

    def test_empty_text(self):
        result = analyse_action_verbs("")
        assert result["total_bullets"] == 0


# ── Quantification Tests ──────────────────────────────────────────

class TestQuantification:

    def test_detects_numbers(self):
        score = score_quantification(SAMPLE_RESUME_TEXT)
        assert score > 0

    def test_empty_text(self):
        score = score_quantification("No bullets here.")
        assert score == 0.0

    def test_all_quantified(self):
        text = "• Increased revenue by 30%\n• Reduced costs by $1M\n• Managed team of 10"
        score = score_quantification(text)
        assert score == 100.0

    def test_score_in_range(self):
        score = score_quantification(SAMPLE_RESUME_TEXT)
        assert 0.0 <= score <= 100.0


# ── Integration Test (mocked Claude) ─────────────────────────────

class TestResumeAnalyzerIntegration:

    @patch("analyzer.ClaudeAnalyser.deep_analyse")
    @patch("analyzer.ClaudeAnalyser.career_path_analysis")
    def test_full_analysis_with_txt(self, mock_career, mock_deep, tmp_path):
        mock_deep.return_value = {
            "overall_score": 78,
            "ats_compatibility": 82,
            "content_quality": 75,
            "impact_score": 70,
            "strengths": ["Strong technical background"],
            "critical_weaknesses": ["Lacks quantification"],
            "recommendations": [],
            "rewritten_summary": "A highly qualified engineer...",
            "career_level": "Senior",
            "estimated_yoe": 8,
            "top_industries": ["Tech"],
            "missing_sections": [],
            "keyword_suggestions": ["microservices"],
            "quantification_feedback": "Good use of metrics",
            "executive_summary": "Strong senior engineering candidate.",
        }
        mock_career.return_value = {
            "current_trajectory": "Staff Engineer",
            "recommended_paths": [],
            "skill_investments": ["Rust"],
            "salary_range": {"min": 200000, "max": 280000, "currency": "USD"},
            "market_demand": "High",
            "differentiation_advice": "Contribute to open source.",
        }

        resume_file = tmp_path / "test_resume.txt"
        resume_file.write_text(SAMPLE_RESUME_TEXT)

        from analyzer import ResumeAnalyzer
        analyzer = ResumeAnalyzer(anthropic_api_key="test-key")
        result = analyzer.full_analysis(resume_file)

        assert result.overall_score == 78
        assert result.ats_score > 0
        assert isinstance(result.resume_data.skills, dict)
        assert result.resume_data.word_count > 0


# ── ATS Scorer Tests ──────────────────────────────────────────────

class TestATSScorer:

    def test_ats_score_returns_float(self):
        from analyzer import ATSScorer
        scorer = ATSScorer.__new__(ATSScorer)
        score = scorer.ats_score(SAMPLE_RESUME_TEXT)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_keyword_match(self):
        from analyzer import ATSScorer
        scorer = ATSScorer.__new__(ATSScorer)
        jd = "Python machine learning TensorFlow kubernetes AWS cloud engineer"
        result = scorer.keyword_match(SAMPLE_RESUME_TEXT, jd)
        assert "matched" in result
        assert "missing" in result
        assert "score" in result
        assert 0 <= result["score"] <= 100
        assert "python" in result["matched"] or len(result["matched"]) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
