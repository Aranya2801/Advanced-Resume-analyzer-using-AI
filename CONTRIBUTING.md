# Contributing to Advanced Resume Analyzer

Thank you for your interest in contributing! 🎉

## Development Setup

```bash
git clone https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI.git
cd Advanced-Resume-analyzer-using-AI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install black isort flake8 pytest pytest-cov
python -m spacy download en_core_web_sm
cp .env.example .env  # add your API key
```

## Code Style

- **Black** for formatting: `black src/ tests/`
- **isort** for imports: `isort src/ tests/`
- **flake8** for linting: `flake8 src/ tests/ --max-line-length=120`
- Type hints everywhere possible
- Docstrings on all public functions/classes

## Tests

- Add tests for every new feature in `tests/test_analyzer.py`
- Run: `pytest tests/ -v --cov=src`
- Target: maintain >80% coverage

## Pull Request Process

1. Branch off `develop` (not `main`)
2. One feature per PR
3. Update README if adding new features
4. All CI checks must pass before merge

## Issues

- Bug reports: include Python version, OS, error traceback
- Feature requests: explain use case and expected behaviour
