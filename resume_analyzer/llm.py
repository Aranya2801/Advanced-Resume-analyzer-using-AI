"""Optional LLM helpers: written feedback, bullet rewrites, cover letters.

Nothing in the scoring depends on this module. If no API key is configured
the app simply hides these features. Supports Anthropic and any
OpenAI-compatible endpoint (OpenAI itself, Groq, a local Ollama server...).

    LLM_PROVIDER=anthropic | openai
    LLM_MODEL=<model id>            # defaults below
    ANTHROPIC_API_KEY / OPENAI_API_KEY
    OPENAI_BASE_URL=<url>           # optional, for compatible servers
"""

from __future__ import annotations

import json
import os
import re

from loguru import logger

DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini"}


class LLMUnavailable(RuntimeError):
    pass


def _extract_json(raw: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add a sentence before
    it. Grab the outermost {...} and parse that."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(raw[start : end + 1])


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None, api_key: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or self._guess_provider()).lower()
        if self.provider not in DEFAULT_MODELS:
            raise LLMUnavailable(f"unknown LLM provider '{self.provider}'")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        key_var = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        self.api_key = api_key or os.getenv(key_var)
        if not self.api_key and not os.getenv("OPENAI_BASE_URL"):
            raise LLMUnavailable(f"{key_var} is not set")
        self._client = None

    @staticmethod
    def _guess_provider() -> str:
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
            return "openai"
        raise LLMUnavailable("no LLM API key configured")

    def _complete(self, system: str, prompt: str, max_tokens: int = 1200) -> str:
        if self.provider == "anthropic":
            import anthropic

            self._client = self._client or anthropic.Anthropic(api_key=self.api_key)
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        from openai import OpenAI

        self._client = self._client or OpenAI(api_key=self.api_key or "none", base_url=os.getenv("OPENAI_BASE_URL"))
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def _json(self, system: str, prompt: str, max_tokens: int = 1200, retries: int = 1) -> dict:
        last = None
        for attempt in range(retries + 1):
            raw = self._complete(system, prompt, max_tokens)
            try:
                return _extract_json(raw)
            except (ValueError, json.JSONDecodeError) as err:
                last = err
                logger.warning(f"LLM returned invalid JSON (attempt {attempt + 1}): {err}")
                prompt += "\n\nYour last reply was not valid JSON. Reply with the JSON object only."
        raise ValueError(f"LLM did not return valid JSON: {last}")

    # --- tasks -------------------------------------------------------------

    SYSTEM = (
        "You review resumes for technical roles. Be specific and concrete; refer to actual lines "
        "from the resume. Never invent experience, employers or numbers the candidate did not mention. "
        "When a metric would help but isn't given, write a placeholder like [X%] instead of guessing."
    )

    def review(self, resume_text: str, analysis_summary: str, job_description: str = "") -> dict:
        """Written feedback. The deterministic analysis is passed in so the
        model comments on it instead of producing its own competing scores."""
        jd = f"\n\nJOB DESCRIPTION:\n{job_description[:3000]}" if job_description else ""
        prompt = (
            "Here is a resume and an automated analysis of it. Write feedback as JSON with keys:\n"
            '  "strengths": [3 short strings],\n'
            '  "issues": [up to 5 objects {"section", "issue", "fix"}],\n'
            '  "summary_rewrite": "a 2-3 sentence professional summary based only on facts in the resume"\n'
            f"\nANALYSIS:\n{analysis_summary}\n\nRESUME:\n{resume_text[:6000]}{jd}"
        )
        return self._json(self.SYSTEM, prompt, max_tokens=1500)

    def rewrite_bullet(self, bullet: str, role: str = "") -> dict:
        prompt = (
            "Rewrite this resume bullet so it starts with a strong verb and states the outcome. "
            "Keep every fact; use [placeholders] for numbers you don't know. "
            'Reply as JSON: {"rewrite": str, "why": str}.\n\n'
            f"ROLE: {role or 'not specified'}\nBULLET: {bullet}"
        )
        return self._json(self.SYSTEM, prompt, max_tokens=300)

    def cover_letter(self, resume_text: str, job_description: str, company: str) -> str:
        prompt = (
            f"Write a cover letter (max 3 short paragraphs) for this candidate applying to {company}. "
            "Use only facts from the resume, connect them to the job's actual requirements, and avoid "
            "clichés like 'I am writing to express my interest'.\n\n"
            f"RESUME:\n{resume_text[:5000]}\n\nJOB DESCRIPTION:\n{job_description[:3000]}"
        )
        return self._complete(self.SYSTEM, prompt, max_tokens=900).strip()


def get_llm(**kwargs) -> LLMClient | None:
    try:
        return LLMClient(**kwargs)
    except LLMUnavailable as err:
        logger.info(f"LLM features off: {err}")
        return None
