"""Regression coverage for the comprehensive repository review."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from photo_curator.cli import main
from photo_curator.models import ScoredImage, get_deepface
from photo_curator.scoring import calculate_total_score, load_image
from photo_curator.selection import filter_near_duplicates
from photo_curator.storage import copy_top_images, export_scores_csv, file_identity
from tests.conftest import save_test_image


def scored(path):
    return ScoredImage(path, 0.9, 0.5, 0.5, 0)


@pytest.mark.parametrize(
    "operation", ["photo_backup", "manifest_backup", "photo_link", "manifest_link"]
)
@pytest.mark.parametrize("abort", [KeyboardInterrupt, SystemExit])
def test_interrupt_after_successful_mutation_restores_album(tmp_path, operation, abort):
    old, new, out = tmp_path / "old.jpg", tmp_path / "new.jpg", tmp_path / "out"
    save_test_image(old)
    save_test_image(new)
    copy_top_images([scored(old)], out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    real_replace, real_link = os.replace, os.link

    def replace(src, dst):
        result = real_replace(src, dst)
        if ".curator_backup_" in str(dst) and (
            (operation == "photo_backup" and Path(src).name == "1_old.jpg")
            or (
                operation == "manifest_backup"
                and Path(src).name == ".curator_manifest.json"
            )
        ):
            raise abort()
        return result

    def link(src, dst):
        result = real_link(src, dst)
        if ".curator_stage_" in str(src) and (
            (operation == "photo_link" and Path(dst).name == "1_new.jpg")
            or (
                operation == "manifest_link"
                and Path(dst).name == ".curator_manifest.json"
            )
        ):
            raise abort()
        return result

    with (
        mock.patch("photo_curator.storage.os.replace", side_effect=replace),
        mock.patch("photo_curator.storage.os.link", side_effect=link),
        pytest.raises(abort),
    ):
        copy_top_images([scored(new)], out, overwrite=True)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not (tmp_path / ".out.curator.lock").exists()


def test_both_manifest_paths_must_be_regular_files(tmp_path):
    photo, out = tmp_path / "a.jpg", tmp_path / "out"
    save_test_image(photo)
    copy_top_images([scored(photo)], out)
    legacy = out / ".curator_manifest.txt"
    legacy.mkdir()
    victim = legacy / "user-data"
    victim.write_text("keep me")
    with pytest.raises(RuntimeError, match="not a regular file"):
        copy_top_images([scored(photo)], out, overwrite=True)
    assert victim.read_text() == "keep me"


def test_concurrent_manifest_is_preserved_and_backups_retained(tmp_path):
    old, new, out = tmp_path / "old.jpg", tmp_path / "new.jpg", tmp_path / "out"
    save_test_image(old)
    save_test_image(new)
    copy_top_images([scored(old)], out)
    real_link = os.link

    def link(src, dst):
        result = real_link(src, dst)
        if ".curator_stage_" in str(src) and Path(dst).name == "1_new.jpg":
            (out / ".curator_manifest.json").write_text("concurrent data")
        return result

    with (
        mock.patch("photo_curator.storage.os.link", side_effect=link),
        pytest.raises(RuntimeError, match="Recovery files preserved"),
    ):
        copy_top_images([scored(new)], out, overwrite=True)
    assert (out / ".curator_manifest.json").read_text() == "concurrent data"
    assert (out / "1_old.jpg").read_bytes() == old.read_bytes()
    (backup,) = out.glob(".curator_backup_*")
    assert json.loads((backup / ".curator_manifest.json").read_text())["files"] == [
        "1_old.jpg"
    ]


def test_first_run_manifest_collision_does_not_clobber(tmp_path):
    photo, out = tmp_path / "a.jpg", tmp_path / "out"
    save_test_image(photo)
    real_link = os.link

    def link(src, dst):
        result = real_link(src, dst)
        if Path(dst).name == "1_a.jpg":
            (out / ".curator_manifest.json").write_text("concurrent data")
        return result

    with (
        mock.patch("photo_curator.storage.os.link", side_effect=link),
        pytest.raises(RuntimeError),
    ):
        copy_top_images([scored(photo)], out)
    assert (out / ".curator_manifest.json").read_text() == "concurrent data"
    assert not (out / "1_a.jpg").exists()


def test_tracked_file_changed_during_staging_is_preserved(tmp_path):
    photo, out = tmp_path / "a.jpg", tmp_path / "out"
    save_test_image(photo)
    copy_top_images([scored(photo)], out)
    import shutil

    real_copy = shutil.copy2

    def copy(src, dst):
        result = real_copy(src, dst)
        replacement = tmp_path / "replacement"
        replacement.write_text("new user data")
        os.replace(replacement, out / "1_a.jpg")
        return result

    with (
        mock.patch("photo_curator.storage.shutil.copy2", side_effect=copy),
        pytest.raises(RuntimeError, match="changed during staging"),
    ):
        copy_top_images([scored(photo)], out, overwrite=True)
    assert (out / "1_a.jpg").read_text() == "new user data"


@pytest.mark.parametrize("kind", ["file", "symlink", "hardlink"])
def test_cli_csv_changes_during_scoring_preserve_source(tmp_path, kind):
    source = tmp_path / "input" / "photo.jpg"
    save_test_image(source)
    before = source.read_bytes()
    preview = tmp_path / "scores.csv"
    argv = [
        "curator",
        "--input",
        str(source.parent),
        "--output",
        str(tmp_path / "out"),
        "--preview_csv",
        str(preview),
        "--dryrun",
        "--no_deepface",
    ]

    def aes(*args, **kwargs):
        if kind == "file":
            preview.write_text("new user data")
        elif kind == "symlink":
            preview.symlink_to(source)
        else:
            os.link(source, preview)
        return [0.5], [torch.tensor([1.0])]

    with (
        mock.patch("sys.argv", argv),
        mock.patch("photo_curator.cli.load_clip_model"),
        mock.patch("photo_curator.cli.encode_reference_text"),
        mock.patch("photo_curator.cli.aesthetic_score_batch", side_effect=aes),
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 1
    assert source.read_bytes() == before
    assert preview.read_bytes() == (b"new user data" if kind == "file" else before)


def test_existing_csv_hardlink_to_source_is_rejected(tmp_path):
    source, csv = tmp_path / "a.jpg", tmp_path / "scores.csv"
    save_test_image(source)
    before = source.read_bytes()
    os.link(source, csv)
    with pytest.raises(RuntimeError, match="conflicts with an input photo"):
        export_scores_csv([scored(source)], csv, overwrite=True)
    assert source.read_bytes() == before


def test_csv_replacement_does_not_truncate_other_hardlinks(tmp_path):
    csv, alias = tmp_path / "scores.csv", tmp_path / "other-report.txt"
    csv.write_text("old report")
    os.link(csv, alias)
    export_scores_csv([], csv, overwrite=True)
    assert alias.read_text() == "old report"
    assert csv.read_text().startswith("Path,")


def test_csv_atomic_commit_refuses_last_moment_collision(tmp_path):
    csv = tmp_path / "scores.csv"
    real_link = os.link

    def collide(src, dst):
        csv.write_text("concurrent report")
        return real_link(src, dst)

    with (
        mock.patch("photo_curator.storage.os.link", side_effect=collide),
        pytest.raises(RuntimeError),
    ):
        export_scores_csv([], csv)
    assert csv.read_text() == "concurrent report"
    assert not list(tmp_path.glob(".curator_csv_*"))


def test_csv_write_failure_leaves_existing_report_intact(tmp_path):
    csv = tmp_path / "scores.csv"
    csv.write_text("old report")
    with (
        mock.patch("photo_curator.storage.os.fsync", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        export_scores_csv([], csv, overwrite=True)
    assert csv.read_text() == "old report"
    assert not list(tmp_path.glob(".curator_csv_*"))


def test_optional_import_does_not_break_help_or_disable_flag(tmp_path):
    (tmp_path / "deepface.py").write_text("raise ValueError('tf-keras unavailable')")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "photo_curator", "--help", "--no_deepface"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--no_deepface" in result.stdout


def test_deepface_initialization_failure_disables_backend():
    import builtins

    original = builtins.__import__

    def import_backend(name, *args, **kwargs):
        if name == "deepface":
            raise ValueError("tf-keras missing")
        return original(name, *args, **kwargs)

    get_deepface.cache_clear()
    try:
        with (
            mock.patch("photo_curator.models.DEEPFACE_AVAILABLE", True),
            mock.patch("builtins.__import__", side_effect=import_backend),
        ):
            assert get_deepface() is None
    finally:
        get_deepface.cache_clear()


@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_16bit_grayscale_preserves_intensity_range(tmp_path, suffix):
    path = tmp_path / ("gradient" + suffix)
    Image.fromarray(np.array([[0, 32768, 65535]], dtype=np.uint16)).save(path)
    pil, cv = load_image(path)
    assert np.asarray(pil).reshape(-1, 3).tolist() == [
        [0, 0, 0],
        [128, 128, 128],
        [255, 255, 255],
    ]
    assert cv[0, 1].tolist() == [128, 128, 128]


@pytest.mark.parametrize("samples", [[0, 128, 255], [0, 32768, 65535]])
def test_32bit_integer_tiff_is_rejected_before_16bit_scaling(tmp_path, samples, caplog):
    path = tmp_path / "integer32.tiff"
    Image.fromarray(np.array([samples], dtype=np.int32)).save(path)
    with Image.open(path) as raw:
        assert raw.mode == "I"
        assert raw.tag_v2[258] == (32,)

    assert load_image(path) == (None, None)
    assert "Only unsigned 16-bit integer grayscale images are supported" in caplog.text


def test_unsupported_content_is_rejected_despite_supported_extension(tmp_path):
    disguised = tmp_path / "photo.jpg"
    Image.new("RGB", (10, 10)).save(disguised, format="GIF")
    assert load_image(disguised) == (None, None)


def test_decompression_bomb_is_skipped(tmp_path):
    path = tmp_path / "a.jpg"
    save_test_image(path)
    with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 100):
        assert load_image(path) == (None, None)


def test_non_square_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    img = Image.new("RGB", (30, 10))
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(path, exif=exif)
    pil, cv = load_image(path)
    assert pil.size == (10, 30)
    assert cv.shape == (30, 10, 3)


def test_real_cosine_removes_identical_embeddings_at_one():
    rng = torch.Generator().manual_seed(0)
    torch.randn(512, generator=rng)
    emb = torch.randn(512, generator=rng)
    images = [
        ScoredImage(Path(f"{i}.jpg"), 1 - i * 0.1, 0.5, 0.5, 0, emb.clone())
        for i in range(2)
    ]
    assert filter_near_duplicates(images, threshold=1) == images[:1]


def test_huge_finite_weights_are_normalized_without_overflow():
    assert calculate_total_score(
        0.8,
        0.4,
        0,
        weight_technical=1e308,
        weight_aesthetic=1e308,
        emotion_active=False,
    ) == pytest.approx(0.6)


def test_nonfinite_component_score_is_rejected():
    with pytest.raises(ValueError, match="Component scores"):
        calculate_total_score(float("nan"), 0.5, 0)


def test_uncertain_exclusive_creation_retains_recovery_files(tmp_path):
    old, new, out = tmp_path / "old.jpg", tmp_path / "new.jpg", tmp_path / "out"
    save_test_image(old)
    save_test_image(new)
    copy_top_images([scored(old)], out)
    real_open = os.open

    def interrupt_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if Path(path).name == "1_new.jpg":
            os.close(fd)
            raise KeyboardInterrupt()
        return fd

    with (
        mock.patch("photo_curator.storage.os.link", side_effect=OSError("no links")),
        mock.patch("photo_curator.storage.os.open", side_effect=interrupt_open),
        pytest.raises(RuntimeError, match="Recovery files preserved"),
    ):
        copy_top_images([scored(new)], out, overwrite=True)
    assert (out / "1_old.jpg").read_bytes() == old.read_bytes()
    assert list(out.glob(".curator_backup_*"))


def test_rollback_does_not_clobber_concurrent_photo(tmp_path):
    old, new, out = tmp_path / "old.jpg", tmp_path / "new.jpg", tmp_path / "out"
    save_test_image(old)
    save_test_image(new)
    copy_top_images([scored(old)], out)
    real_link = os.link

    def link(src, dst):
        if ".curator_stage_" in str(src) and Path(dst).name == ".curator_manifest.json":
            (out / "1_old.jpg").write_text("concurrent user photo")
            raise KeyboardInterrupt()
        return real_link(src, dst)

    with (
        mock.patch("photo_curator.storage.os.link", side_effect=link),
        pytest.raises(RuntimeError, match="Recovery files preserved"),
    ):
        copy_top_images([scored(new)], out, overwrite=True)
    assert (out / "1_old.jpg").read_text() == "concurrent user photo"
    (backup,) = out.glob(".curator_backup_*")
    assert (backup / "1_old.jpg").read_bytes() == old.read_bytes()
