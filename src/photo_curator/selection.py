"""Selection and deduplication algorithms for photo curation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from photo_curator.models import ScoredImage


def filter_near_duplicates(
    scored_images: list[ScoredImage],
    threshold: float | None = None,
    target_count: int | None = None,
    *,
    return_suppressed: bool = False,
) -> list[ScoredImage] | tuple[list[ScoredImage], set[Path]]:
    """Filter out near-duplicate burst shots using CLIP cosine similarity.

    Assumes *scored_images* is pre-sorted descending by score.
    Iteratively selects the highest scoring candidate that is not too similar
    (cosine similarity >= threshold or identical unit embeddings) to any already
    selected candidate. With no threshold, selection follows score order.

    Selected embeddings are kept unit-normalized in one preallocated matrix, so
    each candidate costs a single matrix-vector product rather than re-stacking
    and re-normalizing every earlier selection.

    Stops when target_count is reached. With return_suppressed=True, also returns
    paths rejected as duplicates before that cutoff; later candidates are not
    checked and must not be treated as proven distinct.
    """
    if threshold is None or threshold > 1.0 or not scored_images:
        selected = scored_images[:target_count] if target_count else scored_images
        if return_suppressed:
            return selected, set()
        return selected

    selected: list[ScoredImage] = []
    suppressed_paths: set[Path] = set()
    capacity = min(len(scored_images), target_count or len(scored_images))
    selected_units: torch.Tensor | None = None
    unit_count = 0

    for candidate in scored_images:
        if candidate.embedding is None:
            selected.append(candidate)
            if target_count and len(selected) >= target_count:
                break
            continue

        unit = F.normalize(candidate.embedding.reshape(-1).float(), dim=0)

        if unit_count:
            rows = selected_units[:unit_count]
            sims = rows @ unit
            is_duplicate = bool((sims >= threshold).any())
            if not is_duplicate:
                # Float32 cosine can round just below one for identical vectors,
                # and zero or near-epsilon vectors never reach unit length. An
                # identical row's similarity equals the candidate's similarity
                # with itself, so compare elementwise only against rows near it.
                self_sim = float(unit @ unit)
                matches = rows[sims >= self_sim - 1e-4]
                is_duplicate = bool((matches == unit).all(dim=1).any())
            if is_duplicate:
                suppressed_paths.add(candidate.path)
                continue

        if selected_units is None:
            selected_units = unit.new_empty((capacity, unit.numel()))
        selected_units[unit_count] = unit
        unit_count += 1
        selected.append(candidate)
        if target_count and len(selected) >= target_count:
            break

    if return_suppressed:
        return selected, suppressed_paths
    return selected
