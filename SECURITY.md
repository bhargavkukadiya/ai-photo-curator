# Security Policy

## Supported Versions

Security updates and critical patches are actively provided for the following versions:

| Version | Supported | Notes |
| :--- | :---: | :--- |
| **1.0.x** | :white_check_mark: | Current stable release |
| **< 1.0.0** | :x: | Unsupported legacy development versions |

---

## Security Architecture & Design Principles

Photo Curator handles user filesystem assets and executes deep learning inference. The following describes the current source tree, including the fixes listed under `Unreleased` in [CHANGELOG.md](CHANGELOG.md). Those fixes must not be assumed to exist in older release artifacts.

1. **Transactional Two-Phase Commit (`copy_top_images`)**:
   - New images and manifest entries are prepared in a private staging directory (`.curator_stage_*`). Publication uses no-clobber hard links where available; the fallback creates final files exclusively and streams their data. Readers can see individual files before the manifest commit; the album is not an atomic snapshot.
   - Intent is recorded before filesystem mutations. Commit errors and SIGINT (normally raised as `KeyboardInterrupt`) trigger rollback. Restoration does not overwrite concurrent files. If rollback is interrupted, ownership is uncertain, or restoration fails, backups are retained for manual recovery. SIGKILL and power loss cannot execute Python rollback handlers.

2. **Path Traversal Protection**:
   - Manifest entries must be single filenames within the output directory. Absolute paths, path components, `.`/`..`, reserved manifest filenames, and entries resolving outside the album are rejected with a `RuntimeError`.

3. **Symlink Attack Mitigation**:
   - The internal curator manifest (`.curator_manifest.json`) is validated to ensure it is a regular file. If a symlink is detected for any manifest or tracked file, execution is immediately aborted to prevent arbitrary file clobbering or symlink redirection attacks.

4. **Untracked File Protection**:
   - Photo destinations can replace existing files only when the previous JSON or legacy text manifest tracks them and the user specifies `--overwrite`. The flag permits reuse of a nonempty album, not replacement of untracked photo destinations. Existing preview CSV reports have their own explicit overwrite policy.
   - Photo, manifest, and restore destinations use no-clobber publication (hard links or `os.O_CREAT | os.O_EXCL`). Both manifest paths are validated independently as regular files. The album directory must remain under the caller’s control; the lock coordinates curator writers, not arbitrary external directory manipulation.

5. **Memory Management & Safe Image Decoding**:
   - Pillow's decompression-bomb limit rejects images above ~179 megapixels (twice its default `MAX_IMAGE_PIXELS` of ~89.5 MP); those errors are logged and the affected image is skipped. Images between ~89.5 MP and ~179 MP only raise a warning and are decoded at full resolution, so they can use substantial transient memory. Only JPEG, PNG, WebP, BMP, TIFF, and optionally HEIF decoders are enabled, regardless of the file’s extension.
   - Dimension-capped thumbnailing reduces retained raster buffers to at most 1024px on the longest edge. Initial file decompression and EXIF transposition still temporarily require memory proportional to the source image dimensions.
   - Unsigned 16-bit grayscale is normalized before scoring. TIFF bit depth and sample format are checked; signed/32-bit integer and floating-point grayscale require explicit conversion and are skipped. Album copies retain the original source bytes and metadata, including any EXIF location information.

6. **CSV Protection**:
   - Reports are staged before publication. New report destinations use no-clobber creation. Existing reports require `--overwrite` and are replaced as directory entries rather than truncated in place.
   - Symlinks and source-photo inode aliases are rejected; destination inode identity and source aliases are checked again before export. Formula-like path cells are escaped for spreadsheet use. CSV export follows the album commit, so a report failure does not roll back copied photos.

7. **Supported Dependency Baseline**:
   - Python 3.11+ is required. Security floors cover Pillow, Torch, Transformers, file locks, HTTP libraries, and optional backends.
   - CI audits every distinct package version in `uv.lock`, including optional and mutually exclusive platform/Python branches. A clean audit means no currently known database matches, not a proof that dependencies are vulnerability-free.

---

## Reporting a Vulnerability

We take the security of Photo Curator and its users seriously. If you believe you have found a security vulnerability in this project, please follow these steps:

1. **Do NOT disclose the vulnerability in a public issue.**
2. Report the vulnerability privately via **GitHub Security Advisories**:
   - Navigate to [https://github.com/bhargavkukadiya/ai-photo-curator/security/advisories](https://github.com/bhargavkukadiya/ai-photo-curator/security/advisories)
   - Click **"Report a vulnerability"**, when available, to submit your findings confidentially. This requires private vulnerability reporting to be enabled for the repository; publishing this policy does not enable it.
   - If the button is unavailable, open an [issue](https://github.com/bhargavkukadiya/ai-photo-curator/issues) asking the maintainer for a private reporting channel. Do not include the vulnerability, proof of concept, or affected private data in that request. Share the details only through the private channel the maintainer provides.
   - See [GitHub's private reporting instructions](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately).
3. Provide detailed steps to reproduce the vulnerability, including:
   - Operating system and Python version
   - A minimal proof of concept (PoC) or script
   - Potential impact and recommended mitigation, if known

### Response Timeline

- **Initial Response:** Within 48 hours acknowledging receipt of your report.
- **Triage & Status Update:** Within 5 business days with an assessment of severity and remediation plan.
- **Disclosure:** We coordinate disclosure with you once a security fix has been released.
