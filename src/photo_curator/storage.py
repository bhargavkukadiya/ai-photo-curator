"""Filesystem operations, transactional staging, and manifest management."""

from __future__ import annotations

import csv
import json
import logging
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from photo_curator.models import (
    LEGACY_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    ScoredImage,
)

logger = logging.getLogger(__name__)

FileIdentity = tuple[int, int]
_UNVALIDATED = object()


def file_identity(path: Path) -> FileIdentity | None:
    """Snapshot device/inode identity without following symlinks, not contents."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"File '{path}' is a symlink. Refusing to proceed.")
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"Path '{path}' exists and is not a regular file.")
    return info.st_dev, info.st_ino


@dataclass
class _Publication:
    path: Path
    identity: FileIdentity | None


def _publish_new_file(
    source: Path, destination: Path, journal: list[_Publication]
) -> None:
    """Publish to a vacant slot, recording intent before any mutation.

    Hard-link ownership can be reconciled even if interruption occurs after the
    syscall. If exclusive creation is interrupted before its inode is recorded,
    rollback retains recovery files rather than guessing which file to delete.
    """
    intent = _Publication(destination, file_identity(source))
    journal.append(intent)
    try:
        os.link(source, destination)
        return
    except FileExistsError as e:
        journal.remove(intent)
        raise RuntimeError(
            f"Destination file '{destination}' already exists. "
            "Refusing to overwrite untracked file."
        ) from e
    except OSError:
        pass  # Filesystems without hard links use exclusive streaming creation.

    intent.identity = None
    try:
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        journal.remove(intent)
        raise RuntimeError(
            f"Destination file '{destination}' already exists. "
            "Refusing to overwrite untracked file."
        ) from e
    except OSError:
        journal.remove(intent)
        raise
    with open(fd, "wb", closefd=True) as out_f:
        info = os.fstat(out_f.fileno())
        intent.identity = info.st_dev, info.st_ino
        with open(source, "rb") as in_f:
            shutil.copyfileobj(in_f, out_f)
    shutil.copystat(source, destination, follow_symlinks=False)


def _remove_publication(intent: _Publication) -> None:
    current = file_identity(intent.path)
    if current is None:
        return
    if intent.identity is None or current != intent.identity:
        raise RuntimeError(
            f"Ownership of '{intent.path}' is uncertain; retaining recovery files."
        )
    intent.path.unlink()


def _restore_backup(backup: Path, destination: Path) -> None:
    """Restore without clobbering a file created by another program."""
    journal: list[_Publication] = []
    try:
        _publish_new_file(backup, destination, journal)
    except BaseException:
        for intent in journal:
            _remove_publication(intent)
        raise
    backup.unlink()


_STAGED_REPORT_FLAGS = (
    os.O_CREAT
    | os.O_EXCL
    | os.O_WRONLY
    | getattr(os, "O_BINARY", 0)  # Windows: no CRT newline translation
    | getattr(os, "O_NOFOLLOW", 0)
)


def _create_staged_report(directory: Path, mode: int) -> tuple[int, Path]:
    """Exclusively create a hidden staging file before any content is written.

    *mode* is passed to open(), so the kernel applies the caller's umask (and
    any default ACL) without this process reading or changing the umask, which
    other threads share. Use 0o666 for an ordinary new file and 0o600 when the
    contents must stay private until permissions are finalized.
    """
    for _ in range(100):
        path = directory / f".curator_csv_{secrets.token_hex(8)}"
        try:
            return os.open(path, _STAGED_REPORT_FLAGS, mode), path
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique staging file in '{directory}'.")


def _check_csv_sources(csv_path: Path, protected_paths: Sequence[Path]) -> None:
    for source in protected_paths:
        if source.resolve() == csv_path or (
            csv_path.exists() and source.exists() and csv_path.samefile(source)
        ):
            raise RuntimeError(
                f"Preview CSV '{csv_path}' conflicts with an input photo ({source.name})."
            )


def export_scores_csv(
    scored_images: list[ScoredImage],
    csv_path: Path,
    statuses: dict[Path, str] | None = None,
    *,
    overwrite: bool = False,
    protected_paths: Sequence[Path] = (),
    expected_identity: FileIdentity | None | object = _UNVALIDATED,
) -> None:
    """Stage a report and publish it without truncating existing inodes.

    Existing destinations require overwrite=True. Source-photo aliases and
    symlinks are rejected; protected_paths extends protection to unscored inputs.
    expected_identity can preserve the destination inode identity checked before
    scoring. When supplying expected_identity (including None for an absent
    file), callers must canonicalize the parent before taking that snapshot:
    csv_path = csv_path.parent.resolve() / csv_path.name. The parent is not
    resolved again here, so redirection through a symlink after validation is
    rejected. Status is included only when statuses is supplied, with Unselected
    as the default for unmapped paths.
    """
    csv_path = csv_path.absolute()
    if expected_identity is _UNVALIDATED:
        csv_path = csv_path.parent.resolve() / csv_path.name
    if csv_path.parent.resolve() != csv_path.parent:
        raise RuntimeError(
            f"Preview CSV parent '{csv_path.parent}' resolves through a symlink."
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    protected = [s.path for s in scored_images] + list(protected_paths)
    _check_csv_sources(csv_path, protected)
    current = file_identity(csv_path)
    if expected_identity is _UNVALIDATED:
        expected_identity = current
    if current != expected_identity or (current is not None and not overwrite):
        raise RuntimeError(
            f"Preview CSV '{csv_path}' already exists or changed during scoring."
        )
    # A replacement keeps the permissions of the report it replaces, applied
    # only after writing; until then it stays owner-only so a private report is
    # never exposed through a broader umask default. A new report keeps the
    # umask-derived mode it was created with.
    replaced_mode = (
        stat.S_IMODE(csv_path.lstat().st_mode) if current is not None else None
    )
    fd, staged = _create_staged_report(
        csv_path.parent, 0o666 if replaced_mode is None else 0o600
    )
    journal: list[_Publication] = []
    try:
        with open(fd, "w", newline="", encoding="utf-8", closefd=True) as f:
            writer = csv.writer(f)
            header = ["Path", "TotalScore", "Technical", "Aesthetic", "Emotion"]
            if statuses is not None:
                header.append("Status")
            writer.writerow(header)
            for s in scored_images:
                row = [
                    ("'" + str(s.path))
                    if str(s.path).startswith(("=", "+", "-", "@", "\t", "\r"))
                    else str(s.path),
                    f"{s.total:.4f}",
                    f"{s.technical:.4f}",
                    f"{s.aesthetic:.4f}",
                    f"{s.emotion:.4f}",
                ]
                if statuses is not None:
                    row.append(statuses.get(s.path, "Unselected"))
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        if replaced_mode is not None:
            os.chmod(staged, replaced_mode)
        if csv_path.parent.resolve() != csv_path.parent:
            raise RuntimeError("Preview CSV parent changed during export.")
        _check_csv_sources(csv_path, protected)
        if file_identity(csv_path) != expected_identity:
            raise RuntimeError(f"Preview CSV '{csv_path}' changed during export.")
        if expected_identity is None:
            _publish_new_file(staged, csv_path, journal)
        else:
            os.replace(staged, csv_path)
    except BaseException:
        for intent in journal:
            _remove_publication(intent)
        raise
    finally:
        staged.unlink(missing_ok=True)
    logger.info("Preview CSV saved to %s", csv_path)


def copy_top_images(
    images: list[ScoredImage],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Copy an album under an exclusive, cross-process directory lock.

    A competing writer fails before reading the manifest. The lock lives beside
    the album so it does not interfere with the nonempty-output guard. After an
    uncatchable process termination, remove the abandoned lock directory only
    after confirming that no writer is running and inspecting recovery files.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{output_path.name}.curator.lock"
    try:
        lock_path.mkdir()
    except FileExistsError as e:
        raise RuntimeError(
            f"Output directory '{output_path}' is locked by another writer. "
            f"If the writer was terminated, inspect recovery files and remove "
            f"the abandoned lock directory '{lock_path}' before retrying."
        ) from e
    try:
        return _copy_top_images_locked(images, output_path, overwrite=overwrite)
    finally:
        lock_path.rmdir()


def _copy_top_images_locked(
    images: list[ScoredImage],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Copy the selected images to *output_path* with zero-padded index prefixes.

    Uses transactional staging and attempts rollback if commit fails:
    - All new images and manifest are staged in an isolated temporary directory.
    - Previous manifest and files to be replaced are safely backed up before committing.
    - Failed commits restore prior files where safe; uncertain ownership or
      concurrent replacements cause recovery backups to be retained.
    - JSON manifest avoids corruption from newlines or special characters in filenames.

    Returns the number of successfully copied images.
    Validation and handled transaction failures raise RuntimeError. Other
    filesystem errors can raise OSError; cancellation propagates after successful
    rollback. Cleanup is best effort, with uncertain recovery backups retained.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_output = output_path.resolve()
    manifest_file = output_path / MANIFEST_FILENAME
    legacy_manifest_file = output_path / LEGACY_MANIFEST_FILENAME

    # Reject symlinked manifest
    if manifest_file.is_symlink():
        raise RuntimeError(
            f"Manifest file '{manifest_file}' is a symlink. Refusing to proceed."
        )
    if legacy_manifest_file.is_symlink():
        raise RuntimeError(
            f"Manifest file '{legacy_manifest_file}' is a symlink. Refusing to proceed."
        )

    # Validate both reserved paths, including the unused legacy manifest.
    previous_identities: dict[Path, FileIdentity | None] = {
        path: file_identity(path) for path in (manifest_file, legacy_manifest_file)
    }

    # Enforce overwrite guard inside copy_top_images
    if not overwrite:
        has_existing_files = output_path.exists() and any(
            True for _ in output_path.iterdir()
        )
        if has_existing_files:
            raise RuntimeError(
                f"Output directory '{output_path}' is not empty. "
                "Specify overwrite=True to allow writing to it."
            )

    validated_previous_entries: set[str] = set()
    raw_entries: list[str] = []

    if manifest_file.exists():
        if not manifest_file.is_file():
            raise RuntimeError(
                f"Manifest path '{manifest_file}' exists and is not a regular file."
            )
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if (
                isinstance(data, dict)
                and "files" in data
                and isinstance(data["files"], list)
            ):
                raw_entries = data["files"]
            elif isinstance(data, list):
                raw_entries = data
            else:
                raise ValueError("Invalid manifest structure")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"Failed to read/parse manifest '{manifest_file}': {e}"
            ) from e
    elif legacy_manifest_file.exists():
        if not legacy_manifest_file.is_file():
            raise RuntimeError(
                f"Manifest path '{legacy_manifest_file}' exists and is not a regular file."
            )
        try:
            raw_entries = legacy_manifest_file.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise RuntimeError(
                f"Failed to read existing legacy manifest '{legacy_manifest_file}': {e}"
            ) from e

    for item in raw_entries:
        if not isinstance(item, str):
            raise RuntimeError(
                f"Manifest security error: invalid non-string entry in manifest: {item!r}"
            )
        fname = item
        if not fname:
            continue
        # Path traversal protection
        if Path(fname).name != fname or fname in (
            ".",
            "..",
            MANIFEST_FILENAME,
            LEGACY_MANIFEST_FILENAME,
        ):
            raise RuntimeError(
                f"Manifest security error: invalid or path-traversal entry '{fname}' in manifest."
            )
        target = output_path / fname
        try:
            resolved_target = target.resolve()
            if resolved_target.parent != resolved_output:
                raise RuntimeError(
                    f"Manifest security error: entry '{fname}' resolves outside output directory."
                )
        except (OSError, RuntimeError) as e:
            raise RuntimeError(
                f"Manifest security error validating '{fname}': {e}"
            ) from e

        # Reject directories and non-regular tracked entries before starting transaction
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(
                    f"Manifest security error: tracked entry '{fname}' in '{manifest_file}' is not a regular file (directory or special file)."
                )

        validated_previous_entries.add(fname)
        previous_identities[target] = file_identity(target)

    if not images:
        return 0

    width = len(str(len(images)))
    new_manifest_entries: list[str] = []

    # Pre-validate all target destinations before copying anything
    for idx, scored in enumerate(images, start=1):
        filename = f"{idx:0{width}d}_{scored.path.name}"
        dest = output_path / filename

        # Reject untracked destination collisions
        if dest.exists() or dest.is_symlink():
            if filename not in validated_previous_entries:
                raise RuntimeError(
                    f"Destination file '{dest}' already exists and is not tracked by the previous manifest. "
                    "Refusing to overwrite untracked file."
                )
        new_manifest_entries.append(filename)

    # Allocate staging and backup directories under cleanup protection
    staging_dir: Path | None = None
    backup_dir: Path | None = None
    preserve_backup = False

    try:
        staging_dir = Path(tempfile.mkdtemp(dir=output_path, prefix=".curator_stage_"))
        backup_dir = Path(tempfile.mkdtemp(dir=output_path, prefix=".curator_backup_"))

        # Step 1: Copy all images into the staging directory
        for scored, filename in zip(
            tqdm(images, desc="Copying"), new_manifest_entries, strict=True
        ):
            staged_dest = staging_dir / filename
            try:
                shutil.copy2(scored.path, staged_dest)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to copy '{scored.path}' to staging: {e}"
                ) from e

        # Step 2: Create staged JSON manifest
        staged_manifest = staging_dir / MANIFEST_FILENAME
        manifest_payload = {
            "version": 1,
            "files": new_manifest_entries,
        }
        manifest_bytes = json.dumps(manifest_payload, indent=2).encode("utf-8")
        try:
            fd = os.open(
                staged_manifest,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with open(fd, "wb", closefd=True) as mf:
                mf.write(manifest_bytes)
        except OSError as e:
            raise RuntimeError(
                f"Failed to write staging manifest '{staged_manifest}': {e}"
            ) from e

        # Step 3: Publish staged files, attempting rollback on commit failure.
        moved_to_backup: list[tuple[Path, Path]] = []
        committed_new_files: list[_Publication] = []
        manifest_backed_up = False
        legacy_manifest_backed_up = False

        # Keep recovery files unless commit or rollback explicitly completes.
        preserve_backup = True
        try:
            # 3A: Backup existing manifest files
            for path, identity in previous_identities.items():
                if file_identity(path) != identity:
                    raise RuntimeError(f"Tracked file '{path}' changed during staging.")
            if manifest_file.exists():
                manifest_backed_up = True
                os.replace(manifest_file, backup_dir / MANIFEST_FILENAME)
            if legacy_manifest_file.exists():
                legacy_manifest_backed_up = True
                os.replace(legacy_manifest_file, backup_dir / LEGACY_MANIFEST_FILENAME)

            # Backup previous tracked files (both obsolete ones and ones being replaced)
            for fname in sorted(validated_previous_entries):
                dest_file = output_path / fname
                if dest_file.exists() or dest_file.is_symlink():
                    bkp_file = backup_dir / fname
                    moved_to_backup.append((dest_file, bkp_file))
                    if file_identity(dest_file) != previous_identities[dest_file]:
                        raise RuntimeError(
                            f"Tracked file '{dest_file}' changed during commit."
                        )
                    os.replace(dest_file, bkp_file)

            # 3B: Publish each staged file without clobbering existing entries.
            for filename in new_manifest_entries:
                staged_src = staging_dir / filename
                final_dest = output_path / filename
                _publish_new_file(staged_src, final_dest, committed_new_files)

            # The manifest slot must still be vacant after backing up its old entry.
            _publish_new_file(staged_manifest, manifest_file, committed_new_files)

        except BaseException as e:
            # Rollback on ANY commit failure
            logger.error("Commit failed, rolling back changes: %s", e)
            rollback_errors: list[str] = []

            for publication in reversed(committed_new_files):
                try:
                    _remove_publication(publication)
                except (OSError, RuntimeError) as err:
                    rollback_errors.append(str(err))

            for orig_dest, bkp_file in reversed(moved_to_backup):
                try:
                    if bkp_file.exists() or bkp_file.is_symlink():
                        _restore_backup(bkp_file, orig_dest)
                except (OSError, RuntimeError) as err:
                    rollback_errors.append(
                        f"Failed to restore '{orig_dest}' from backup: {err}"
                    )

            if manifest_backed_up and (backup_dir / MANIFEST_FILENAME).exists():
                try:
                    _restore_backup(backup_dir / MANIFEST_FILENAME, manifest_file)
                except (OSError, RuntimeError) as err:
                    rollback_errors.append(
                        f"Failed to restore manifest '{manifest_file}': {err}"
                    )
            if (
                legacy_manifest_backed_up
                and (backup_dir / LEGACY_MANIFEST_FILENAME).exists()
            ):
                try:
                    _restore_backup(
                        backup_dir / LEGACY_MANIFEST_FILENAME,
                        legacy_manifest_file,
                    )
                except (OSError, RuntimeError) as err:
                    rollback_errors.append(f"Failed to restore legacy manifest: {err}")

            if rollback_errors:
                preserve_backup = True
                logger.critical(
                    "Rollback encountered errors. Backup files preserved at: %s",
                    backup_dir,
                )
                raise RuntimeError(
                    f"Commit failed ({e}) and rollback encountered errors. "
                    f"Recovery files preserved in '{backup_dir}': {'; '.join(rollback_errors)}"
                ) from e

            preserve_backup = False
            if not isinstance(e, Exception):
                raise
            raise RuntimeError(f"Commit phase failed and was rolled back: {e}") from e

        else:
            # Cancellation after the commit body must not begin rollback once
            # the recovery files have been marked disposable.
            preserve_backup = False

    finally:
        # Cleanup temporary staging and backup directories
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists() and not preserve_backup:
            shutil.rmtree(backup_dir, ignore_errors=True)

    logger.info("Copied %d images to %s", len(images), output_path)
    return len(images)
