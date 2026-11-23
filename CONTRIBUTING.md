# Contributing

Issues and PRs are welcome. A few notes so things stay easy to review:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m spacy download en_core_web_md
pytest -q
ruff check .
```

- **Adding skills**: edit `resume_analyzer/data/skills.json`. Put aliases after the canonical name; prefix an alias
  with `=` if it should be matched case-sensitively (short names like `Go`). If two skills in a family are *not*
  substitutes for each other, add the family to `_no_partial`. Please add a test in `tests/test_analyzer.py` if the
  alias could plausibly clash with an ordinary English word.
- **Changing scoring weights**: run `python -m benchmark.evaluate` before and after and paste both tables in the PR.
  Don't tune on the test split.
- **New matching methods**: add them to `Bench.methods()` in `benchmark/evaluate.py` so they get compared against
  the baselines.
