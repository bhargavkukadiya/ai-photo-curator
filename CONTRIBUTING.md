# Contributing to Photo Curator

Thank you for your interest in contributing to **Photo Curator**! 🎉

We welcome contributions of all kinds: bug reports, documentation improvements, feature requests, and pull requests. Please take a moment to review these guidelines before getting started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Running Tests](#running-tests)
- [CI Coverage](#ci-coverage)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors and users with respect and courtesy.

---

## Getting Started

1. **Fork the repository** on GitHub: [https://github.com/bhargavkukadiya/ai-photo-curator](https://github.com/bhargavkukadiya/ai-photo-curator)
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/ai-photo-curator.git
   cd ai-photo-curator
   ```
3. **Set the upstream remote**:
   ```bash
   git remote add upstream https://github.com/bhargavkukadiya/ai-photo-curator.git
   ```

---

## Development Setup

Use Python 3.11 or newer. The locked setup matches CI and creates `.venv` automatically:

```bash
python -m pip install --upgrade uv
uv sync --frozen --extra dev
uv run --frozen --extra dev ai-photo-curator --help
```

Alternatively, install from the package constraints with pip in an activated virtual environment:

```bash
# 1. Create a virtual environment with Python 3.11+
python3 -m venv .venv

# 2. Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install in editable development mode with test dependencies
python -m pip install -e ".[dev]"
```

Optional dependencies can also be installed:

```bash
# Optional: Facial emotion scoring (DeepFace)
python -m pip install -e ".[dev,emotion]"

# Optional: HEIC/HEIF photo support
python -m pip install -e ".[dev,heic]"

# Or install everything:
python -m pip install -e ".[all,dev]"
```

For the locked optional-backend environment, use `uv sync --frozen --extra dev --extra emotion --extra heic` and include those same extras in subsequent `uv run` commands.

---

## Coding Guidelines

- **Python Version:** Support Python 3.11, 3.12, and 3.13.
- **Type Annotations:** Use PEP 484 and PEP 604 type annotations (`int | None`, `list[str]`, etc.). Include `from __future__ import annotations` at the top of all new modules.
- **Docstrings:** Document all public functions, classes, and CLI parameters using Google or PEP 257 style docstrings. Explain the *why* and operational constraints, not just the *what*.
- **Error Handling:** Avoid bare `except:` clauses. Catch specific exceptions and log helpful context.
- **Memory Safety:** When processing images, avoid keeping uncompressed full-resolution bitmaps in long-lived variables or batch lists.
- **Filesystem Operations:** Stage and validate writes before publication, then attempt rollback on commit errors. Individual hard links are atomic; an album update and the exclusive-stream fallback are not atomic snapshots. Never overwrite untracked photo destinations, even with `--overwrite`. Record transaction intent before mutations and retain backups when ownership or restoration is uncertain. Stage CSV reports and replace permitted destinations without truncating existing inodes.

---

## Running Tests

Core tests mock neural inference; optional backend smoke tests run when the corresponding extras are installed:

```bash
# Run the complete suite against the locked core and test dependencies
uv run --frozen --extra dev pytest -v

# Include real optional-backend initialization and HEIF roundtrip checks
uv run --frozen --extra dev --extra emotion --extra heic pytest -v

# For the pip development environment, use: python -m pytest -v
# Run a specific test module or method
uv run --frozen --extra dev pytest tests/test_selection.py -k "TestFilterNearDuplicates" -v

# Check for syntax errors across all modules
uv run --frozen --extra dev python -m compileall -q src tests scripts album_selector.py test_album_selector.py setup.py

# Check the legacy test entrypoint
uv run --frozen --extra dev pytest test_album_selector.py -v

# Build the distribution
uv build --wheel

# Audit every locked registry package/version (requires network access)
uv run --no-project --python 3.12 --with pip-audit python scripts/audit_lock.py
```

Ensure all tests pass cleanly with **zero failures** and **zero new warnings** before opening a pull request.

The two optional smoke tests are skipped when their extras are absent. They initialize DeepFace and perform a HEIF encode/decode roundtrip without downloading facial model weights or running real-face emotion inference. The core suite mocks CLIP inference.

---

## CI Coverage

The [CI workflow](.github/workflows/ci.yml) currently runs:

| Job | Platforms / Python | Checks |
| --- | --- | --- |
| Core | Ubuntu: 3.11, 3.12, 3.13; macOS and Windows: 3.12 | Frozen core/dev installation, full suite, wheel build, imports outside the checkout, and both console entrypoints |
| Optional backends | Ubuntu: 3.12 | Frozen dev/emotion/HEIC installation and full suite including the real backend smoke checks |
| Security | Ubuntu: 3.12 | Advisory audit of every distinct registry package/version in `uv.lock`, including alternate versions and extras |

`scripts/check_wheel.py` expects one wheel in `dist/` and replaces the editable package in its calling environment with that wheel. CI runs it after the tests. Use a disposable environment if running that script locally.

---

## Pull Request Process

1. **Create a topic branch** from `main`:
   ```bash
   git checkout -b feature/my-awesome-improvement
   ```
2. **Make your changes** with clear, focused commits following [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat:`, `fix:`, `docs:`, `test:`, `perf:`).
3. **Add or update unit tests** covering your new functionality or bug fix.
4. **Run the test suite** to ensure no regressions:
   ```bash
   uv run --frozen --extra dev pytest -v
   ```
5. **Push your branch** to your fork:
   ```bash
   git push origin feature/my-awesome-improvement
   ```
6. **Open a Pull Request** against the `main` branch of `bhargavkukadiya/ai-photo-curator`.
7. Fill in the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) completely, describing the rationale, changes made, and verification steps.

---

## Reporting Issues

- **Bug Reports:** Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Include your OS, Python version, exact command line arguments, and terminal traceback.
- **Feature Requests:** Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md). Clearly describe the problem you are trying to solve and your proposed solution.
- **Security Vulnerabilities:** Please review our [Security Policy](SECURITY.md) for instructions on confidential disclosure.
