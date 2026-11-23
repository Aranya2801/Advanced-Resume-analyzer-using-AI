"""Command line use.

    python -m resume_analyzer analyze resume.pdf
    python -m resume_analyzer analyze resume.pdf --jd job.txt
    python -m resume_analyzer analyze resume.pdf --jd job.txt --json > report.json
    python -m resume_analyzer rank resumes/*.pdf --jd job.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _read_jd(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8") if path else None


def _print_analysis(a) -> None:
    d = a.to_dict()
    q = d["quality"]
    print(f"\n{d['source'] or 'resume'} - quality {q['overall']:.0f}/100")
    for name, comp in q["components"].items():
        print(f"  {name:15s} {comp['score']:5.0f}   {comp['findings'][0]}")
        for extra in comp["findings"][1:]:
            print(f"  {'':15s}        {extra}")
    if d["bullets"]["weak_examples"]:
        print("\n  Weak openers:")
        for b in d["bullets"]["weak_examples"][:3]:
            print(f"    - {b}")
    if "match" in d:
        m = d["match"]
        comps = ", ".join(f"{k} {v:.0f}" for k, v in m["components"].items())
        print(f"\n  Job match {m['score']:.0f}/100 ({comps})")
        print(f"  matched: {', '.join(m['skills']['matched']) or '-'}")
        if m["skills"]["partial"]:
            print(f"  partial: {', '.join(m['skills']['partial'])}")
        print(f"  missing: {', '.join(m['skills']['missing']) or '-'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="resume_analyzer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="score one resume, optionally against a JD")
    p_an.add_argument("resume")
    p_an.add_argument("--jd", help="job description text file")
    p_an.add_argument("--json", action="store_true", help="print the full report as JSON")

    p_rank = sub.add_parser("rank", help="rank several resumes against one JD")
    p_rank.add_argument("resumes", nargs="+")
    p_rank.add_argument("--jd", required=True)
    p_rank.add_argument("--json", action="store_true")

    for p in (p_an, p_rank):
        p.add_argument("--encoder", default="auto", help="auto | sbert | spacy | none")

    args = ap.parse_args(argv)

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from .pipeline import ResumeAnalyzer

    analyzer = ResumeAnalyzer(encoder=args.encoder)
    jd = _read_jd(args.jd)

    if args.cmd == "analyze":
        result = analyzer.analyze(args.resume, job_description=jd)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_analysis(result)
        return 0

    paths = [p for pattern in args.resumes for p in (glob.glob(pattern) or [pattern])]
    resumes = [analyzer.parse(p) for p in paths]
    ranked = analyzer.rank(resumes, jd)
    if args.json:
        print(json.dumps([{"source": r.resume.source, "score": round(r.score * 100, 1),
                           "components": {k: round(v * 100, 1) for k, v in r.components.items()},
                           "missing_skills": r.match.skills.missing[:8]} for r in ranked], indent=2))
    else:
        for i, r in enumerate(ranked, 1):
            print(f"{i:2d}. {r.score * 100:5.1f}  {r.resume.source}   missing: {', '.join(r.match.skills.missing[:5]) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
