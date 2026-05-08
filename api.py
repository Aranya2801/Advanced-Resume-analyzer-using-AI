"""
Advanced Resume Analyzer — FastAPI REST Backend
Exposes all analysis capabilities as REST endpoints.
"""

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from loguru import logger

from analyzer import ResumeAnalyzer, AnalysisResult

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Advanced Resume Analyzer API",
    description="MIT-grade AI-powered resume analysis API backed by Claude AI + NLP",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyser (lazy init per request with API key)
_analyzers: dict[str, ResumeAnalyzer] = {}


def get_analyzer(api_key: str) -> ResumeAnalyzer:
    if api_key not in _analyzers:
        _analyzers[api_key] = ResumeAnalyzer(anthropic_api_key=api_key)
    return _analyzers[api_key]


# ── Request / Response models ─────────────────────────────────────

class CoverLetterRequest(BaseModel):
    resume_text: str
    job_description: str
    company: str
    api_key: str


class BulletImprovementRequest(BaseModel):
    bullet: str
    context: Optional[str] = ""
    api_key: str


class CareerPathRequest(BaseModel):
    resume_text: str
    api_key: str


# ── Routes ────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "🟢 online",
        "service": "Advanced Resume Analyzer API",
        "version": "2.0.0",
        "endpoints": ["/analyse", "/cover-letter", "/improve-bullet", "/career-path", "/docs"],
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "timestamp": __import__("datetime").datetime.utcnow().isoformat()}


@app.post("/analyse", tags=["Analysis"])
async def analyse_resume(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT resume"),
    api_key: str = Form(..., description="Anthropic API key"),
    job_description: str = Form("", description="Optional job description for matching"),
    target_role: str = Form("", description="Target job role"),
):
    """
    Full resume analysis pipeline:
    - Text extraction
    - ATS scoring
    - Keyword matching
    - Claude AI deep analysis
    - Career path insights
    - Action verb analysis
    - Readability metrics
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".txt"}:
        raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, or TXT.")

    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        analyzer = get_analyzer(api_key)
        result = analyzer.full_analysis(tmp_path, job_description=job_description, target_role=target_role)
        return JSONResponse(content=analyzer.to_dict(result))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/cover-letter", tags=["Generation"])
async def generate_cover_letter(req: CoverLetterRequest):
    """Generate a personalised cover letter from resume + JD."""
    try:
        analyzer = get_analyzer(req.api_key)
        text = analyzer.claude.generate_cover_letter(req.resume_text, req.job_description, req.company)
        return {"cover_letter": text}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/improve-bullet", tags=["Generation"])
async def improve_bullet(req: BulletImprovementRequest):
    """AI-powered bullet point rewrite."""
    try:
        analyzer = get_analyzer(req.api_key)
        result = analyzer.claude.improve_bullet(req.bullet, req.context)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/career-path", tags=["Analysis"])
async def career_path(req: CareerPathRequest):
    """Career trajectory and path recommendations."""
    try:
        analyzer = get_analyzer(req.api_key)
        result = analyzer.claude.career_path_analysis(req.resume_text)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
