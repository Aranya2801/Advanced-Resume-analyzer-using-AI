"""Tests for the analysis pipeline. They run without network access and
without any embedding model (encoder="none")."""

from datetime import date

import pytest

from resume_analyzer import MatchConfig, ResumeAnalyzer
from resume_analyzer.bullets import analyse_bullets, classify_opening, has_metric
from resume_analyzer.matching import BM25, combine, keyword_recall, skill_coverage, v1_keyword_overlap
from resume_analyzer.parsing import UnsupportedFormatError, extract_contact, load_document
from resume_analyzer.scoring import flesch_reading_ease
from resume_analyzer.sections import (
    _RESUME_LOOKUP,
    Requirement,
    bullets,
    experience_months,
    jd_requirements,
    match_heading,
    split_sections,
)
from resume_analyzer.skills import default_taxonomy

RESUME = """ALEX CHEN
alex.chen@email.com | +1 (415) 555-0192 | linkedin.com/in/alexchen | github.com/alexchen

PROFESSIONAL SUMMARY
Backend engineer with 6 years building payment systems.

WORK EXPERIENCE
Senior Software Engineer | Stripe | Jan 2021 – Present
• Architected a fraud detection service in Go processing 2.4M transactions/sec
• Reduced infrastructure costs by $2.3M annually by moving batch jobs to Spark
• Mentored 6 junior engineers; 4 were promoted within 18 months

Software Engineer | Airbnb | Mar 2018 – Jan 2021
• Responsible for API development, including a new GraphQL gateway
• Helped with various backend tasks

EDUCATION
B.S. Computer Science, Stanford University, 2016 | GPA: 3.9

Technical Skills:
Python, Java, TypeScript, Kafka, k8s, PostgreSQL, Redis, AWS, Docker
"""

JD = """Senior Backend Engineer

Responsibilities
- Design and run high-throughput services in Go
- Own our Kafka-based event pipeline

Requirements
- 5+ years of backend experience with Go or Java
- Production experience with Kubernetes and PostgreSQL
- Experience with Kafka

Nice to have
- Familiarity with Rust
- gRPC is a plus

We offer
- Remote-friendly, competitive salary
"""


@pytest.fixture(scope="module")
def analyzer():
    return ResumeAnalyzer(encoder="none", today=date(2026, 6, 1))


@pytest.fixture(scope="module")
def tax():
    return default_taxonomy()


# --- parsing -----------------------------------------------------------------

class TestContact:
    def test_fields(self):
        c = extract_contact(RESUME)
        assert c.name == "Alex Chen"
        assert c.email == "alex.chen@email.com"
        assert "415" in c.phone
        assert c.linkedin.endswith("linkedin.com/in/alexchen")
        assert c.github.endswith("github.com/alexchen")

    def test_year_range_is_not_a_phone_number(self):
        c = extract_contact("JANE DOE\njane@x.org\nAcme Corp 2018 - 2021")
        assert c.phone == ""

    def test_name_from_pipe_header(self):
        assert extract_contact("Priya Sharma | priya@ml.dev | +1 628 555 0847").name == "Priya Sharma"


class TestLoading:
    def test_txt(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text(RESUME, encoding="utf-8")
        doc = load_document(p)
        assert "Stripe" in doc.text and doc.pages >= 1

    def test_docx_includes_tables(self, tmp_path):
        docx = pytest.importorskip("docx")
        d = docx.Document()
        t = d.add_table(rows=1, cols=2)
        t.cell(0, 0).text = "Sam Lee"
        t.cell(0, 1).text = "sam@lee.dev"
        d.add_paragraph("Experience")
        d.add_paragraph("Built a thing that served 10k users")
        p = tmp_path / "r.docx"
        d.save(p)
        text = load_document(p).text
        assert "sam@lee.dev" in text and "10k users" in text

    def test_bytes_need_filename(self):
        with pytest.raises(ValueError):
            load_document(b"hello")

    @pytest.mark.parametrize("name", ["resume.doc", "resume.xyz", "resume"])
    def test_unsupported(self, name):
        with pytest.raises(UnsupportedFormatError):
            load_document(b"abc", filename=name)


# --- sections and dates ------------------------------------------------------

class TestSections:
    @pytest.mark.parametrize("line,expected", [
        ("WORK EXPERIENCE", "experience"),
        ("Technical Skills:", "skills"),
        ("Professional Summary", "summary"),
        ("Projects & Publications", "projects"),
        ("Education", "education"),
        ("• Built the experience layer", None),
        ("I have a lot of experience with Python and Go", None),
    ])
    def test_headings(self, line, expected):
        assert match_heading(line, _RESUME_LOOKUP) == expected

    def test_split(self):
        s = split_sections(RESUME)
        assert {"header", "summary", "experience", "education", "skills"} <= set(s)
        assert "Stripe" in s["experience"] and "Kafka" in s["skills"]

    def test_bullets(self):
        b = bullets(RESUME)
        assert len(b) == 5
        assert b[0].startswith("Architected")

    def test_experience_merges_overlaps(self):
        text = "Jan 2020 - Dec 2020\nJun 2020 - Jun 2021\n2023 - Present"
        # Jan 2020..Jun 2021 = 18 months, Jan 2023..Jun 2026 = 42 months
        assert experience_months(text, today=date(2026, 6, 1)) == 18 + 42

    def test_month_without_year_takes_end_year(self):
        assert experience_months("Jun – Aug 2023", today=date(2026, 1, 1)) == 3

    def test_jd_requirements_weights(self):
        reqs = jd_requirements(JD)
        texts = {r.text: r.weight for r in reqs}
        assert texts["Experience with Kafka"] == 1.0
        assert texts["gRPC is a plus"] == 0.5
        assert not any("salary" in t for t in texts)  # benefits dropped


# --- skills ----------------------------------------------------------------------

class TestSkills:
    def test_aliases(self, tax):
        found = tax.find("Deployed on k8s with Postgres; wrote golang and sklearn code")
        assert {"Kubernetes", "PostgreSQL", "Go", "scikit-learn"} <= set(found)

    def test_ambiguous_short_names(self, tax):
        found = tax.find("I will go to the office. R&D budget. Washington D.C.")
        assert "Go" not in found and "R" not in found and "C" not in found

    def test_symbols(self, tax):
        found = tax.find("C++, C#, .NET and Node.js")
        assert {"C++", "C#", ".NET", "Node.js"} <= set(found)
        assert "C" not in found and "JavaScript" not in found

    def test_longest_alias_wins(self, tax):
        found = tax.find("Tests written with React Testing Library")
        assert "React Testing Library" in found and "React" not in found

    def test_sections_recorded(self, analyzer):
        r = analyzer.parse(RESUME)
        assert "experience" in r.skills["Go"].sections
        assert r.skills["Kubernetes"].sections == {"skills"}


# --- bullets ---------------------------------------------------------------------

class TestBullets:
    @pytest.mark.parametrize("bullet,kind", [
        ("Led a team of 5 engineers", "strong"),
        ("Responsible for API development", "weak"),
        ("Helped with backend tasks", "weak"),
        ("Instrumented the checkout flow", "strong"),   # unknown -ed verb
        ("Kubernetes was migrated to 1.28", "neutral"),  # old code called this weak because of "was"
        ("Shadow traffic testing for the new API", "neutral"),  # ...and this, because of "had" in "shadow"
    ])
    def test_openers(self, bullet, kind):
        assert classify_opening(bullet) == kind

    @pytest.mark.parametrize("bullet,expected", [
        ("Increased revenue by 30%", True),
        ("Saved $1.2M a year", True),
        ("Served 50k users", True),
        ("Cut p99 latency", True),
        ("Worked on billing, invoicing, and payments", False),  # the old regex matched the commas
        ("Joined the team in 2021", False),
        ("Jan 2020 - Present", False),
    ])
    def test_metrics(self, bullet, expected):
        assert has_metric(bullet) is expected

    def test_report(self):
        rep = analyse_bullets(bullets(RESUME))
        assert rep.total == 5 and rep.weak == 2 and rep.strong == 3
        assert rep.quantified == 3


# --- matching --------------------------------------------------------------------

class TestMatching:
    def test_evidence_and_families(self, analyzer, tax):
        r = analyzer.parse(RESUME)
        weights = {"Go": 1.0, "Kubernetes": 1.0, "gRPC": 0.5, "Rust": 0.5}
        m = skill_coverage(r.skills, weights, tax)
        via = {i.skill: i.via for i in m.items}
        assert via["Go"] == "exact"              # used in an experience bullet
        assert via["Kubernetes"] == "listed-only"  # only in the skills list
        assert via["gRPC"] == "related:GraphQL"
        assert via["Rust"] == "missing"          # languages aren't treated as substitutes
        # Go 1.0 + K8s 0.6 + gRPC 0.5*0.5 (related, GraphQL is in experience) + Rust 0
        assert m.score == pytest.approx((1.0 + 0.6 + 0.25 + 0) / 3.0)

    def test_ablation_switches(self, analyzer, tax):
        r = analyzer.parse(RESUME)
        w = {"Kubernetes": 1.0, "gRPC": 1.0}
        plain = skill_coverage(r.skills, w, tax, MatchConfig(use_families=False, use_evidence=False))
        assert plain.score == pytest.approx(0.5)

    def test_full_match(self, analyzer):
        m = analyzer.analyze(RESUME, job_description=JD).match
        assert m is not None
        assert set(m.components) == {"skills", "keywords"}  # no encoder in tests
        assert 0 < m.score < 1
        assert "Kafka" in m.skills.matched
        assert "Rust" in m.skills.missing

    def test_unrelated_jd_scores_lower(self, analyzer):
        r = analyzer.parse(RESUME)
        good = analyzer.match(r, JD).score
        bad = analyzer.match(r, "Requirements\n- Figma and UX research\n- Tableau dashboards").score
        assert good > bad

    def test_keyword_recall(self):
        score, hits, misses = keyword_recall("python services on kubernetes", [Requirement("Python and Kubernetes and Terraform", 1.0)])
        assert score == pytest.approx(2 / 3)
        assert "terraform" in misses

    def test_combine_renormalises(self):
        assert combine({"a": 1.0, "b": None}, {"a": 0.25, "b": 0.75}) == 1.0
        assert combine({}, {"a": 1}) == 0.0

    def test_bm25_ranks_relevant_first(self):
        docs = ["kafka streaming pipelines in go", "figma mockups and user research", "go services and kafka consumers"]
        scores = BM25(docs).scores("go kafka")
        assert scores[1] == 0 and min(scores[0], scores[2]) > 0

    def test_v1_baseline_runs(self):
        assert 0 <= v1_keyword_overlap(RESUME, JD) <= 1

    def test_rank(self, analyzer):
        other = RESUME.replace("Go", "Figma").replace("Kafka", "Sketch").replace("Spark", "Photoshop")
        third = "SAM\nsam@x.com\n\nEXPERIENCE\n• Managed social media campaigns for 3 brands"
        ranked = analyzer.rank([analyzer.parse(t) for t in (third, other, RESUME)], JD)
        assert ranked[0].resume.text == RESUME.strip()
        assert "tfidf" in ranked[0].components


# --- scoring -------------------------------------------------------------------

class TestQuality:
    def test_report_shape(self, analyzer):
        a = analyzer.analyze(RESUME)
        d = a.to_dict()
        assert 0 <= d["quality"]["overall"] <= 100
        assert set(d["quality"]["components"]) == {"impact", "quantification", "structure", "ats", "concision", "readability"}
        assert d["stats"]["months_experience"] > 60

    def test_deterministic(self, analyzer):
        assert analyzer.analyze(RESUME).quality.overall == analyzer.analyze(RESUME).quality.overall

    def test_better_bullets_score_higher(self, analyzer):
        weak = RESUME.replace("Architected", "Responsible for").replace("Reduced", "Helped with").replace("Mentored", "Worked on")
        assert analyzer.analyze(weak).quality.components["impact"].score < analyzer.analyze(RESUME).quality.components["impact"].score

    def test_flesch(self):
        assert flesch_reading_ease("too short") is None
        assert -50 < flesch_reading_ease(RESUME) < 100


# --- misc ------------------------------------------------------------------------

def test_cli_analyze_json(tmp_path, capsys):
    import json

    from resume_analyzer.__main__ import main

    r = tmp_path / "r.txt"
    r.write_text(RESUME, encoding="utf-8")
    j = tmp_path / "jd.txt"
    j.write_text(JD, encoding="utf-8")
    assert main(["analyze", str(r), "--jd", str(j), "--json", "--encoder", "none"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["match"]["skills"]["missing"]


def test_cli_rank(tmp_path, capsys):
    from resume_analyzer.__main__ import main

    for i, text in enumerate([RESUME, RESUME.replace("Kafka", "Sketch"), "Nobody\n\nEXPERIENCE\n• Sold shoes"]):
        (tmp_path / f"r{i}.txt").write_text(text, encoding="utf-8")
    (tmp_path / "jd.txt").write_text(JD, encoding="utf-8")
    assert main(["rank", str(tmp_path / "r*.txt"), "--jd", str(tmp_path / "jd.txt"), "--encoder", "none"]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert "r0.txt" in first


@pytest.mark.parametrize("raw", [
    '{"rewrite": "x", "why": "y"}',
    '```json\n{"rewrite": "x", "why": "y"}\n```',
    'Sure! Here it is:\n{"rewrite": "x", "why": "y"}\nHope that helps.',
])
def test_llm_json_extraction(raw):
    from resume_analyzer.llm import _extract_json

    assert _extract_json(raw) == {"rewrite": "x", "why": "y"}


def test_llm_off_without_key(monkeypatch):
    from resume_analyzer.llm import get_llm

    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    assert get_llm() is None


def test_pdf_roundtrip(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    p = tmp_path / "r.pdf"
    c = canvas.Canvas(str(p), pagesize=letter)
    y = 750
    for line in RESUME.splitlines():
        c.drawString(50, y, line.replace("•", "-").replace("–", "-"))
        y -= 14
    c.save()
    doc = load_document(p)
    assert doc.pages == 1 and "Stripe" in doc.text
    assert extract_contact(doc.text).email == "alex.chen@email.com"


def test_scanned_pdf_is_reported(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    p = tmp_path / "blank.pdf"
    canvas.Canvas(str(p)).save()
    with pytest.raises(UnsupportedFormatError, match="scanned"):
        load_document(p)
