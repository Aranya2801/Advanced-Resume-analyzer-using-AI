"""REST API.

    uvicorn api:app --reload        # docs at http://localhost:8000/docs

Changes from the first version: API keys are read from the server's
environment instead of being sent with every request (they were also used as
dict keys to cache one analyzer per key, which grew forever), uploads have a
size limit, and errors no longer echo raw exception text back to the client.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from resume_analyzer import ResumeAnalyzer, __version__
from resume_analyzer.llm import get_llm
from resume_analyzer.parsing import SUPPORTED_EXTENSIONS, UnsupportedFormatError

MAX_UPLOAD_MB = float(os.getenv("MAX_FILE_SIZE_MB", "5"))
MAX_RANK_FILES = 50

app = FastAPI(title="Resume Analyzer API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def analyzer() -> ResumeAnalyzer:
    return ResumeAnalyzer(encoder=os.getenv("RESUME_ANALYZER_ENCODER", "auto"))


async def _read_upload(file: UploadFile) -> bytes:
    name = file.filename or ""
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB:g} MB")
    return data


def _parse(data: bytes, filename: str):
    try:
        return analyzer().parse(data, filename=filename)
    except UnsupportedFormatError as err:
        raise HTTPException(422, str(err)) from err
    except Exception as err:
        logger.exception(f"failed to parse {filename}")
        raise HTTPException(422, "Could not read this file. Is it a valid PDF/DOCX/TXT?") from err


class TextMatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=50)
    job_description: str = Field(..., min_length=30)


class BulletRequest(BaseModel):
    bullet: str = Field(..., min_length=5, max_length=600)
    role: str = ""


class CoverLetterRequest(BaseModel):
    resume_text: str = Field(..., min_length=50)
    job_description: str = Field(..., min_length=30)
    company: str = Field(..., min_length=1, max_length=120)


@app.get("/health")
def health():
    a = analyzer()
    return {
        "status": "ok",
        "version": __version__,
        "encoder": a.encoder.name if a.encoder else None,
        "llm": bool(get_llm()),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), job_description: str = Form("")):
    """Quality report for one resume; adds a job match if a JD is given."""
    data = await _read_upload(file)
    resume = _parse(data, file.filename)
    return analyzer().analyze(resume, job_description=job_description or None).to_dict()


@app.post("/match")
def match_text(req: TextMatchRequest):
    a = analyzer()
    return a.match(a.parse(req.resume_text), req.job_description).to_dict()


@app.post("/rank")
async def rank(files: list[UploadFile] = File(...), job_description: str = Form(...)):
    """Rank several resumes for one JD (pool mode)."""
    if len(files) > MAX_RANK_FILES:
        raise HTTPException(400, f"At most {MAX_RANK_FILES} files per request")
    resumes = [_parse(await _read_upload(f), f.filename) for f in files]
    ranked = analyzer().rank(resumes, job_description)
    return [
        {
            "file": r.resume.source,
            "score": round(r.score * 100, 1),
            "components": {k: round(v * 100, 1) for k, v in r.components.items()},
            "matched_skills": r.match.skills.matched,
            "missing_skills": r.match.skills.missing,
        }
        for r in ranked
    ]


def _llm():
    llm = get_llm()
    if llm is None:
        raise HTTPException(503, "LLM features are not configured on this server")
    return llm


@app.post("/llm/rewrite-bullet")
def rewrite_bullet(req: BulletRequest):
    try:
        return _llm().rewrite_bullet(req.bullet, req.role)
    except HTTPException:
        raise
    except Exception as err:
        logger.exception("bullet rewrite failed")
        raise HTTPException(502, "The language model request failed") from err


@app.post("/llm/cover-letter")
def cover_letter(req: CoverLetterRequest):
    try:
        return {"cover_letter": _llm().cover_letter(req.resume_text, req.job_description, req.company)}
    except HTTPException:
        raise
    except Exception as err:
        logger.exception("cover letter failed")
        raise HTTPException(502, "The language model request failed") from err
