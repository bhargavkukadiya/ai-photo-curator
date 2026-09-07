# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-09-07

### Added
- **Initial Release of AI Photo Curator**: Production-grade, AI-powered photo selector that curates, ranks, and deduplicates large photo collections for albums with transactional filesystem safety.
- **Multi-Signal AI Quality Engine**:
  - **Technical Quality (40%)**: Canonical Laplacian variance edge sharpness scoring combined with normalized luminance exposure balance. Multichannel-safe for 2D grayscale, 3D single-channel, 3-channel BGR, and 4-channel BGRA arrays.
  - **Aesthetic Quality (40%)**: Batched OpenAI CLIP (`ViT-B/32`) cosine similarity scored against configurable semantic reference text prompts.
  - **Facial Emotion Detection (20%, optional)**: DeepFace facial expression analysis prioritizing genuine smiles and positive emotion.
  - **Dynamic Weight Fusion**: Normalized quality scoring with automatic rebalancing when emotion detection is disabled or unavailable.
- **Vectorized Near-Duplicate Deduplication**:
  - Accelerated burst-shot deduplication utilizing batched PyTorch tensor matrix multiplication (`torch.matmul`), achieving 37×–100× faster filtering over large photo libraries.
- **Memory-Bounded Image Ingestion**:
  - Decode-time memory downscaling (capped at 1024px max-edge) using Lanczos interpolation, reducing retained dual-buffer memory per 48MP photo by >98% (~288 MB to ~4.72 MB combined) and preventing batch memory accumulation.
- **Robust Two-Phase Transactional Commit**:
  - Isolated temporary directory staging (`.curator_stage_*`) ensuring zero partial writes in destination folders.
  - Non-destructive JSON manifest tracking (`.curator_manifest.json`) with path-traversal safeguards and symlink rejection.
  - Untracked destination collision rejection to prevent clobbering user files.
  - Atomic link with `O_CREAT | O_EXCL` exclusive streaming copy and metadata preservation (`copystat`) for FAT32, exFAT, and network shares.
- **Exclusive Cross-Process Directory Locking**:
  - Atomic directory lock (`.<album>.curator.lock`) placed beside the target album to reject concurrent writers prior to manifest reading or staging.
- **Interrupt-Safe Rollback & Recovery Retention**:
  - Transaction commit phase catches `BaseException` to guarantee rollback execution on `SIGINT` (Ctrl-C) or `SystemExit`.
  - Retention of `.curator_backup_*` recovery files if rollback is interrupted or encounters errors, guaranteeing zero data loss.
- **Dry-Run & Preview CSV Export**:
  - Actionable CSV reports with detailed quality metrics and explicit status classifications (`Selected`, `Duplicate_Suppressed`, `Rank_Cutoff`), safely exported after photo copying.
- **Hardware Acceleration**:
  - Native zero-configuration device routing for Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), and multi-core `cpu`.
- **Packaging & Dual CLI Entrypoints**:
  - Packaged as `ai-photo-curator` with console script entrypoints `ai-photo-curator` and backward-compatible `photo-curator`.
  - Backward-compatibility proxy `album_selector.py` for legacy scripts.
- **Deterministic Dependencies**:
  - Tracked `uv.lock` for exact dependency resolution and fast cross-platform environment synchronization.
- **Comprehensive In-Memory Test Suite**:
  - 95 unit and integration tests executing with mocked neural models in under 8 seconds.
