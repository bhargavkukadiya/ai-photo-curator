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

## Key Verification & Testing
Describe the tests you ran to verify your changes:
- [ ] Added or updated unit tests in `tests/`
- [ ] All 95+ tests pass locally: `pytest -v` (or `uv run pytest -v`)
- [ ] Syntax compilation check passes: `python3 -m py_compile album_selector.py $(find src tests -name "*.py")`
- [ ] Tested CLI execution on `sample_photos/`:
  ```bash
  ai-photo-curator --input ./sample_photos --output ./out --dryrun
  ```

## Security & Safety Checklist
- [ ] No regression to transactional two-phase commit safety
- [ ] Path traversal checks and symlink guards are preserved
- [ ] Bounded memory limits preserved during image ingestion
- [ ] No hardcoded paths, credentials, or sensitive files

## Screenshots / CLI Output (if applicable)
```text
Paste sample CLI output or preview CSV snippet here
```
