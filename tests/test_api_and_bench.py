import numpy as np
import pytest

from benchmark.evaluate import average_precision, ndcg_at, paired_bootstrap, query_metrics
from benchmark.roles import relevance


class TestMetrics:
    def test_ndcg_perfect_and_reversed(self):
        rels = np.array([2, 1, 0, 0])
        assert ndcg_at(rels, rels, 4) == pytest.approx(1.0)
        assert ndcg_at(rels[::-1], rels, 4) < 0.6

    def test_average_precision(self):
        # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
        assert average_precision(np.array([2, 0, 2, 0])) == pytest.approx((1 + 2 / 3) / 2)

    def test_query_metrics(self):
        scores = np.array([0.9, 0.8, 0.1, 0.5])
        rels = np.array([2, 0, 2, 1])
        kinds = np.array(["normal", "stuffed", "paraphrase", "normal"])
        m = query_metrics(scores, rels, kinds)
        assert m["MRR"] == 1.0
        assert m["Stuffed@10"] == pytest.approx(0.25)
        assert m["ParaRecall@R"] == 0.0  # the paraphrase resume is ranked last

    def test_bootstrap_ci_contains_mean(self):
        a, b = np.array([0.9, 0.8, 0.7, 0.95]), np.array([0.5, 0.6, 0.7, 0.4])
        d, lo, hi = paired_bootstrap(a, b, n=2000)
        assert lo <= d <= hi and lo > 0

    def test_relevance_is_symmetric(self):
        assert relevance("ml_engineer", "data_scientist") == relevance("data_scientist", "ml_engineer") == 1
        assert relevance("ml_engineer", "ml_engineer") == 2
        assert relevance("ml_engineer", "marketing_manager") == 0


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    import os

    os.environ["RESUME_ANALYZER_ENCODER"] = "none"
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        os.environ.pop(k, None)
    from fastapi.testclient import TestClient

    import api

    api.analyzer.cache_clear()
    return TestClient(api.app)


class TestApi:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["llm"] is False

    def test_analyze_upload(self, client):
        text = open("data/samples/resumes/software_engineer_senior.txt", "rb").read()
        jd = open("data/samples/jobs/senior_ml_engineer.txt", encoding="utf-8").read()
        r = client.post("/analyze", files={"file": ("r.txt", text, "text/plain")}, data={"job_description": jd})
        assert r.status_code == 200
        body = r.json()
        assert body["contact"]["email"] == "alex.chen@email.com"
        assert "match" in body and 0 <= body["match"]["score"] <= 100

    def test_rejects_bad_extension(self, client):
        r = client.post("/analyze", files={"file": ("r.exe", b"MZ...", "application/octet-stream")})
        assert r.status_code == 400

    def test_rank(self, client):
        files = [("files", (f"{n}.txt", open(f"data/samples/resumes/{n}.txt", "rb").read(), "text/plain"))
                 for n in ("software_engineer_senior", "data_scientist_mid", "frontend_engineer_entry")]
        jd = open("data/samples/jobs/frontend_engineer.txt", encoding="utf-8").read()
        r = client.post("/rank", files=files, data={"job_description": jd})
        assert r.status_code == 200
        assert r.json()[0]["file"] == "frontend_engineer_entry.txt"

    def test_llm_endpoints_off_without_key(self, client):
        r = client.post("/llm/rewrite-bullet", json={"bullet": "Helped with backend tasks"})
        assert r.status_code == 503
