# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- Require Python 3.11+ to remove vulnerable legacy dependency branches; refresh the lockfile and enforce security floors for core and optional backends.
- Initialize DeepFace lazily and include its TensorFlow compatibility backend in the emotion extra.
- Use `pyproject.toml` as the single source of package metadata, retaining `setup.py` as a compatibility shim.
- Keep selected deduplication embeddings unit-normalized in one preallocated matrix; selecting 5,000 distinct photos drops from ~3.7s to ~0.15s.
- Skip DeepFace entirely when `--weight_emotion` is 0, since it cannot affect ranking.
- Skip earlier curated albums (directories with a curator manifest) nested inside the input during discovery.
- `emotion_score` no longer reads overrides from the `album_selector` shim module; patch `photo_curator.scoring.DeepFace` instead.
- Dependabot now updates `uv.lock` (uv ecosystem), and CI fails when the lockfile is out of date with `pyproject.toml`.

### Fixed

- Record backup and publication intent before filesystem mutations so interrupts after successful operations can be reconciled safely.
- Publish manifests and restore backups without clobbering concurrent files; retain recovery files if rollback is incomplete or ownership is uncertain.
- Independently reject non-regular JSON and legacy manifest paths.
- Stage CSV reports, enforce overwrite policy during publication, reject photo aliases and symlinks, and replace reports without truncating other hard links.
- Restrict image decoder formats; normalize unsigned 16-bit grayscale before RGB conversion and reject signed or 32-bit integer TIFFs using their encoded sample type.
- Remove exact duplicate embeddings at cosine threshold 1.0 despite floating-point roundoff.
- Normalize large finite scoring weights without overflow and reject non-finite component scores.
- Reject empty image arrays with a clear `ValueError` before technical scoring.
- Reject non-uint8 arrays in `technical_score` instead of silently misscoring them (uint16 mid-gray scored 0.0, floats crashed OpenCV); score gray+alpha arrays from their gray channel.
- Configure logging only when the CLI runs, so importing `photo_curator` no longer installs a root handler.
- Give new preview CSVs umask-default permissions instead of owner-only `0600`, and keep a replaced report's mode.
- Assert specific validation messages in CLI tests; several previously passed on unrelated errors such as a missing relative input path or an unparseable device string.
- Report missing or ambiguous wheel artifacts before attempting the wheel smoke-test installation.
- Document canonical CSV parent paths for prevalidated exports and cover parent symlink redirection with a regression test.
- Correct safety, memory, and DeepFace package-license documentation; clarify EXIF preservation and CSV cutoff semantics.
- Align diagrams, CLI/CSV references, contributor commands, and CI coverage with the current implementation; replace noreply email contacts with private reporting instructions.

### Added

- Regression tests for interrupted operations, concurrent writers, source aliases, high-bit-depth decoding, and scoring boundaries.
- CI jobs for frozen dependency resolution, all-lockfile vulnerability auditing, optional backends, and built-wheel smoke checks.

## [1.0.0] - 2026-09-07

### Added

- **Initial Release of AI Photo Curator**: Production-grade, AI-powered photo selector that curates, ranks, and deduplicates large photo collections for albums with transactional filesystem safety.
- **Multi-Signal AI Quality Engine**:
  - **Technical Quality (40%)**: Canonical Laplacian variance edge sharpness scoring combined with normalized luminance exposure balance. Multichannel-safe for 2D grayscale, 3D single-channel, 3-channel BGR, and 4-channel BGRA arrays.
  - **Aesthetic Quality (40%)**: Batched OpenAI CLIP (`ViT-B/32`) cosine similarity scored against configurable semantic reference text prompts.
  - **Facial Emotion Detection (20%, optional)**: DeepFace facial expression analysis prioritizing genuine smiles and positive emotion.
  - **Dynamic Weight Fusion**: Normalized quality scoring with automatic rebalancing when emotion detection is disabled or unavailable.
- **Vectorized Near-Duplicate Deduplication**:
  - Accelerated burst-shot deduplication utilizing batched PyTorch tensor matrix multiplication (`torch.matmul`); historical benchmark results are documented in the README.
- **Memory-Bounded Image Ingestion**:
  - Post-decode memory downscaling (capped at 1024px max-edge) using Lanczos interpolation, reducing retained dual-buffer memory per 48MP photo by >98% (~288 MB to ~4.72 MB combined) and preventing batch memory accumulation.
- **Robust Two-Phase Transactional Commit**:
  - Isolated temporary directory staging (`.curator_stage_*`) before publication. Exclusive stream copies can be visible as partial files during commit.
  - Non-destructive JSON manifest tracking (`.curator_manifest.json`) with path-traversal safeguards and symlink rejection.
  - Untracked destination collision rejection to prevent clobbering user files.
  - Atomic hard links where supported, with `O_CREAT | O_EXCL` exclusive streaming copy and metadata preservation (`copystat`) on filesystems without hard links.
- **Exclusive Cross-Process Directory Locking**:
  - Atomic directory lock (`.<album>.curator.lock`) placed beside the target album to reject concurrent writers prior to manifest reading or staging.
- **Interrupt-Safe Rollback & Recovery Retention**:
  - Transaction commit phase catches `BaseException` to attempt rollback on `SIGINT` (Ctrl-C) or `SystemExit`.
  - Retention of `.curator_backup_*` recovery files when rollback is interrupted or fails, for manual recovery. Abrupt process termination and power loss require separate recovery handling.
- **Dry-Run & Preview CSV Export**:
  - Actionable CSV reports with detailed quality metrics and explicit status classifications (`Selected`, `Duplicate_Suppressed`, `Rank_Cutoff`), safely exported after photo copying.
- **Hardware Acceleration**:
  - Explicit device routing for Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), and multi-core `cpu`.
- **Packaging & Dual CLI Entrypoints**:
  - Packaged as `ai-photo-curator` with console script entrypoints `ai-photo-curator` and backward-compatible `photo-curator`.
  - Backward-compatibility proxy `album_selector.py` for legacy scripts.
- **Deterministic Dependencies**:
  - Tracked `uv.lock` for exact dependency resolution and fast cross-platform environment synchronization.
- **Comprehensive In-Memory Test Suite**:
  - 95 unit and integration tests executing with mocked neural models in under 8 seconds.
