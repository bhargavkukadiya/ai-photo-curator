## Summary of Changes

Provide a brief overview of the changes introduced in this pull request and the problem they solve.

## Motivation & Context

Why is this change required? What issue or discussion does it address?
- Fixes #(issue_number)

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] ⚡ Performance improvement
- [ ] 🛠️ Refactoring / Code quality
- [ ] 📝 Documentation update
- [ ] 🧪 Test suite enhancement
- [ ] Breaking compatibility change (describe migration requirements)

## Key Verification & Testing

Describe the tests you ran to verify your changes:
- [ ] Added or updated unit tests in `tests/`
- [ ] The full test suite passes locally: `uv run --frozen --extra dev pytest -v` (or `python -m pytest -v` after installing `.[dev]`)
- [ ] Relevant optional-backend checks pass with `--extra emotion --extra heic` when applicable
- [ ] Syntax compilation check passes: `uv run --frozen --extra dev python -m compileall -q src tests scripts album_selector.py test_album_selector.py setup.py`
- [ ] Tested CLI execution on `sample_photos/`:
  ```bash
  uv run --frozen --extra dev ai-photo-curator --input ./sample_photos --output ./out --no_deepface --dryrun
  ```

## Security & Safety Checklist

- [ ] No regression to transactional two-phase commit safety
- [ ] Path traversal checks and symlink guards are preserved
- [ ] Post-decode buffer limits and transient decode-memory caveats are preserved
- [ ] No hardcoded paths, credentials, or sensitive files

## Screenshots / CLI Output (if applicable)

```text
Paste sample CLI output or preview CSV snippet here
```
