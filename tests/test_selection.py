"""Unit tests for selection, ranking, and deduplication modules."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from photo_curator.models import ScoredImage
from photo_curator.selection import filter_near_duplicates


class TestScoredImage:
    """Tests for the ScoredImage dataclass."""

    def test_creation(self):
        img = ScoredImage(Path("a.jpg"), total=0.8, technical=0.7, aesthetic=0.9, emotion=0.0)
        assert img.path == Path("a.jpg")
        assert img.total == 0.8
        assert img.embedding is None

    def test_sorting(self):
        """ScoredImage objects should sort correctly by total descending."""
        img1 = ScoredImage(Path("1.jpg"), total=0.5, technical=0.5, aesthetic=0.5, emotion=0.0)
        img2 = ScoredImage(Path("2.jpg"), total=0.9, technical=0.9, aesthetic=0.9, emotion=0.0)
        img3 = ScoredImage(Path("3.jpg"), total=0.1, technical=0.1, aesthetic=0.1, emotion=0.0)

        sorted_images = sorted([img1, img2, img3], key=lambda s: s.total, reverse=True)
        assert [s.path.name for s in sorted_images] == ["2.jpg", "1.jpg", "3.jpg"]


class TestFilterNearDuplicates:
    """Tests for burst shot deduplication based on CLIP embedding similarity."""

    def test_dedup_removes_similar_images(self):
        """Images with similarity >= threshold should be filtered out."""
        emb1 = torch.tensor([1.0, 0.0])
        emb2 = torch.tensor([0.99, 0.01])  # Near duplicate of emb1
        emb3 = torch.tensor([0.0, 1.0])    # Completely different image

        images = [
            ScoredImage(Path("a.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb1),
            ScoredImage(Path("b.jpg"), total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb2),
            ScoredImage(Path("c.jpg"), total=0.7, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb3),
        ]

        selected = filter_near_duplicates(images, threshold=0.90)
        assert [s.path for s in selected] == [Path("a.jpg"), Path("c.jpg")]

    def test_dedup_disabled_when_none(self):
        """When threshold is None, all scored images are returned."""
        images = [
            ScoredImage(Path("a.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0),
            ScoredImage(Path("b.jpg"), total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0),
        ]
        selected = filter_near_duplicates(images, threshold=None)
        assert len(selected) == 2

    def test_target_count_limit(self):
        """Deduplication stops when target_count is reached."""
        images = [
            ScoredImage(Path(f"{i}.jpg"), total=1.0 - i * 0.1, technical=0.5, aesthetic=0.5, emotion=0.0)
            for i in range(10)
        ]
        selected = filter_near_duplicates(images, threshold=None, target_count=3)
        assert len(selected) == 3

    def test_dedup_threshold_one_removes_exact_duplicates(self):
        """Threshold 1.0 should remove candidates with exact similarity (1.0)."""
        emb1 = torch.tensor([1.0, 0.0])
        emb2 = torch.tensor([1.0, 0.0])

        images = [
            ScoredImage(Path("a.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb1),
            ScoredImage(Path("b.jpg"), total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb2),
        ]

        selected = filter_near_duplicates(images, threshold=1.0)
        assert [s.path for s in selected] == [Path("a.jpg")]

    def test_vectorized_dedup_multiple_candidates_correctness(self):
        """Vectorized deduplication correctly tracks multiple chosen candidates."""
        c1_a = torch.tensor([1.0, 0.0, 0.0])
        c1_b = torch.tensor([0.98, 0.02, 0.0])  # Near dup of c1_a
        c2_a = torch.tensor([0.0, 1.0, 0.0])
        c2_b = torch.tensor([0.0, 0.99, 0.01])  # Near dup of c2_a
        c3_a = torch.tensor([0.0, 0.0, 1.0])

        candidates = [
            ScoredImage(Path("1.jpg"), 0.95, 0.5, 0.5, 0.0, c1_a),
            ScoredImage(Path("2.jpg"), 0.90, 0.5, 0.5, 0.0, c1_b),  # Suppressed by 1.jpg
            ScoredImage(Path("3.jpg"), 0.85, 0.5, 0.5, 0.0, c2_a),  # Selected
            ScoredImage(Path("4.jpg"), 0.80, 0.5, 0.5, 0.0, c2_b),  # Suppressed by 3.jpg
            ScoredImage(Path("5.jpg"), 0.75, 0.5, 0.5, 0.0, c3_a),  # Selected
        ]

        selected, suppressed = filter_near_duplicates(
            candidates, threshold=0.90, return_suppressed=True
        )
        assert [s.path for s in selected] == [Path("1.jpg"), Path("3.jpg"), Path("5.jpg")]
        assert suppressed == {Path("2.jpg"), Path("4.jpg")}


def _reference_dedup(images, threshold, target_count):
    """Straightforward pairwise implementation used as a correctness oracle."""
    selected, suppressed = [], set()
    for cand in images:
        unit = torch.nn.functional.normalize(cand.embedding, dim=0)
        if any(
            torch.equal(unit, kept) or float(unit @ kept) >= threshold
            for kept in (torch.nn.functional.normalize(s.embedding, dim=0) for s in selected)
        ):
            suppressed.add(cand.path)
            continue
        selected.append(cand)
        if len(selected) >= target_count:
            break
    return selected, suppressed


@pytest.mark.parametrize("target_count", [5, 40, 400])
def test_matches_pairwise_reference(target_count):
    """The preallocated matrix path agrees with a naive pairwise comparison."""
    gen = torch.Generator().manual_seed(7)
    bases = torch.randn(30, 16, generator=gen)
    # Bursts: eight adjacent noisy variants of each base, as from a camera burst.
    embs = bases.repeat_interleave(8, dim=0) + 0.15 * torch.randn(240, 16, generator=gen)
    images = [
        ScoredImage(Path(f"{i}.jpg"), 1.0 - i / 240, 0.5, 0.5, 0.0, embs[i])
        for i in range(240)
    ]
    selected, suppressed = filter_near_duplicates(
        images, threshold=0.95, target_count=target_count, return_suppressed=True
    )
    ref_selected, ref_suppressed = _reference_dedup(images, 0.95, target_count)
    assert [s.path for s in selected] == [s.path for s in ref_selected]
    assert suppressed == ref_suppressed
    assert suppressed  # The fixture must actually exercise suppression.


def test_scaled_and_batched_embeddings_are_duplicates():
    """Cosine ignores magnitude, and (1, d) embeddings are accepted like (d,)."""
    images = [
        ScoredImage(Path("a.jpg"), 0.9, 0.5, 0.5, 0.0, torch.tensor([[3.0, 4.0]])),
        ScoredImage(Path("b.jpg"), 0.8, 0.5, 0.5, 0.0, torch.tensor([6.0, 8.0])),
        ScoredImage(Path("c.jpg"), 0.7, 0.5, 0.5, 0.0, torch.tensor([[-4.0, 3.0]])),
    ]
    selected, suppressed = filter_near_duplicates(
        images, threshold=1.0, return_suppressed=True
    )
    assert [s.path for s in selected] == [Path("a.jpg"), Path("c.jpg")]
    assert suppressed == {Path("b.jpg")}


def test_large_target_scales_linearly_per_candidate():
    """Selecting thousands of distinct photos must not re-stack prior selections."""
    import time

    embs = torch.randn(5000, 512, generator=torch.Generator().manual_seed(0))
    images = [
        ScoredImage(Path(f"{i}.jpg"), 1.0 - i / 5000, 0.5, 0.5, 0.0, embs[i])
        for i in range(5000)
    ]
    start = time.perf_counter()
    selected = filter_near_duplicates(images, threshold=0.9, target_count=5000)
    elapsed = time.perf_counter() - start
    assert len(selected) == 5000
    # The previous implementation took ~3.7s here; keep generous CI headroom.
    assert elapsed < 2.0


@pytest.mark.parametrize(
    "vector",
    [
        torch.zeros(4),
        torch.full((4,), 1e-14),
        # Normalizes to length ~0.9995: not unit, yet far from zero.
        torch.tensor([9.995e-13, 0.0, 0.0, 0.0]),
        torch.tensor([9.99e-13, 0.0, 0.0, 0.0]),
        torch.tensor([9.9995e-13, 0.0, 0.0, 0.0]),
    ],
)
def test_identical_degenerate_embeddings_are_exact_duplicates(vector):
    """Zero or sub-epsilon vectors never reach unit cosine, but identical ones match."""
    images = [
        ScoredImage(Path("a.jpg"), 0.9, 0.5, 0.5, 0.0, vector.clone()),
        ScoredImage(Path("b.jpg"), 0.8, 0.5, 0.5, 0.0, vector.clone()),
        ScoredImage(Path("c.jpg"), 0.7, 0.5, 0.5, 0.0, torch.tensor([1.0, 0, 0, 0])),
    ]
    selected, suppressed = filter_near_duplicates(
        images, threshold=1.0, return_suppressed=True
    )
    assert [s.path for s in selected] == [Path("a.jpg"), Path("c.jpg")]
    assert suppressed == {Path("b.jpg")}


def test_distinct_degenerate_embeddings_are_kept():
    images = [
        ScoredImage(Path("a.jpg"), 0.9, 0.5, 0.5, 0.0, torch.tensor([1e-14, 0.0])),
        ScoredImage(Path("b.jpg"), 0.8, 0.5, 0.5, 0.0, torch.tensor([0.0, 1e-14])),
    ]
    assert len(filter_near_duplicates(images, threshold=1.0)) == 2
