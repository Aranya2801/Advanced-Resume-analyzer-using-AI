"""Explainable resume analysis and resume-JD matching."""

from .matching import MatchConfig, MatchResult
from .pipeline import Analysis, Resume, ResumeAnalyzer

__all__ = ["ResumeAnalyzer", "Analysis", "Resume", "MatchConfig", "MatchResult"]
__version__ = "0.3.0"
