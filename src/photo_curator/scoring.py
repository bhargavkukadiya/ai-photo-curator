"""Computer vision, deep learning, and quality scoring functions."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer, util

from photo_curator.models import (
    DEEPFACE_AVAILABLE,
    DEFAULT_WEIGHT_AESTHETIC,
    DEFAULT_WEIGHT_EMOTION,
    DEFAULT_WEIGHT_TECHNICAL,
    MAX_ANALYSIS_DIMENSION,
    SHARPNESS_MAX,
    SHARPNESS_MIN,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
    get_deepface,
)

DeepFace = None

logger = logging.getLogger(__name__)


def is_image_file(path: Path) -> bool:
    """Check if a file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def technical_score(image: np.ndarray) -> float:
    """Sharpness + exposure score, normalized to [0, 1].

    Sharpness is the variance of the Laplacian, standardized by resizing
    the image to a canonical resolution (max edge MAX_ANALYSIS_DIMENSION)
    to reduce sensitivity to image resolution, clamped to [SHARPNESS_MIN, SHARPNESS_MAX]
    and scaled to [0, 1].

    Exposure is normalized to [0, 1] using distance from midpoint (0.5),
    where 0.5 is 1.0 (ideal exposure) and 0.0 or 1.0 is 0.0 (pure black or blown-out white).
    The two sub-scores are averaged.
    """
    if image.size == 0:
        raise ValueError("Image array must not be empty.")
    h, w = image.shape[:2]
    if max(h, w) > MAX_ANALYSIS_DIMENSION:
        scale = MAX_ANALYSIS_DIMENSION / max(h, w)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    if resized.ndim == 2:
        gray = resized
    elif resized.ndim == 3:
        channels = resized.shape[2]
        if channels == 1:
            gray = resized.squeeze(axis=2)
        elif channels == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image array shape: {image.shape}")

    # Sharpness: clamp then scale to [0, 1]
    raw_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = float(
        np.clip(raw_sharpness, SHARPNESS_MIN, SHARPNESS_MAX) / SHARPNESS_MAX
    )

    # Exposure: 1.0 when mean brightness is exactly 0.5, 0.0 at extremes (0.0 or 1.0)
    mean_brightness = float(np.mean(gray)) / 255.0
    exposure = float(np.clip(1.0 - 2.0 * abs(mean_brightness - 0.5), 0.0, 1.0))

    return (sharpness + exposure) / 2.0


def aesthetic_score_batch(
    pil_images: Sequence[Image.Image],
    clip_model: SentenceTransformer,
    ref_embedding: torch.Tensor,
    batch_size: int = 32,
) -> tuple[list[float], list[torch.Tensor]]:
    """CLIP cosine similarity between a batch of images and reference text embedding.

    Raw CLIP cosine similarity for photos typically falls in ~[0.1, 0.4].
    We rescale that range to [0, 1] so the score is comparable with the
    other signals.

    Returns ``(scores, embeddings)``. Raises RuntimeError on inference failure.
    """
    if not pil_images:
        return [], []
    try:
        img_embs = clip_model.encode(
            list(pil_images),
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        sims = util.cos_sim(img_embs, ref_embedding).squeeze(dim=-1)
        if sims.ndim == 0:
            sims = sims.unsqueeze(0)

        scores: list[float] = []
        embeddings: list[torch.Tensor] = []
        for i in range(len(pil_images)):
            sim = sims[i].item()
            score = float(np.clip((sim - 0.1) / 0.3, 0.0, 1.0))
            scores.append(score)
            embeddings.append(img_embs[i])
        return scores, embeddings
    except Exception as e:
        logger.error("Batch aesthetic scoring failed: %s", e)
        raise RuntimeError(f"CLIP aesthetic scoring failed: {e}") from e


def aesthetic_score(
    pil_img: Image.Image,
    clip_model: SentenceTransformer,
    ref_embedding: torch.Tensor,
) -> float:
    """Single-image aesthetic scoring helper."""
    scores, _ = aesthetic_score_batch(
        [pil_img], clip_model, ref_embedding, batch_size=1
    )
    return scores[0] if scores else 0.0


def emotion_score(
    image_input: Path | np.ndarray,
    *,
    deepface_enabled: bool,
) -> float:
    """Score based on detected happiness using DeepFace.

    Supports either a Path to an image file or an in-memory OpenCV BGR array.
    Aggregates all detected faces in group photos by taking the mean happiness.
    Filters out fallback results where no face was detected (face_confidence <= 0).
    Returns 0.0 if DeepFace is unavailable, disabled, or no faces are detected.
    """
    import sys

    album_mod = sys.modules.get("album_selector")
    is_available = getattr(album_mod, "DEEPFACE_AVAILABLE", None)
    if is_available is None:
        is_available = DEEPFACE_AVAILABLE

    if not deepface_enabled or not is_available:
        return 0.0
    deepface_cls = getattr(album_mod, "DeepFace", None) or DeepFace or get_deepface()
    if deepface_cls is None:
        return 0.0

    try:
        target = str(image_input) if isinstance(image_input, Path) else image_input
        result = deepface_cls.analyze(
            img_path=target,
            actions=["emotion"],
            enforce_detection=False,
        )

        if not isinstance(result, list):
            result = [result]

        face_scores: list[float] = []
        for face in result:
            if isinstance(face, dict) and "emotion" in face:
                confidence = face.get("face_confidence", face.get("confidence", 1.0))
                if confidence is None or confidence <= 0:
                    continue
                happy = face["emotion"].get("happy", 0.0)
                face_scores.append(happy / 100.0)

        if not face_scores:
            return 0.0

        return float(np.mean(face_scores))

    except Exception as e:
        name = image_input.name if isinstance(image_input, Path) else "in-memory image"
        logger.warning("DeepFace analysis failed for %s: %s", name, e)
        return 0.0


def calculate_total_score(
    technical: float,
    aesthetic: float,
    emotion: float,
    *,
    weight_technical: float = DEFAULT_WEIGHT_TECHNICAL,
    weight_aesthetic: float = DEFAULT_WEIGHT_AESTHETIC,
    weight_emotion: float = DEFAULT_WEIGHT_EMOTION,
    emotion_active: bool = True,
) -> float:
    """Calculate normalized total score across active signals.

    With emotion_active=False, only technical and aesthetic weights participate.
    Finite, non-negative weights are normalized without overflowing their sum.
    For component scores in [0, 1], the result is also in [0, 1].
    All-zero active weights return 0.0; non-finite active scores or invalid active
    weights raise ValueError.
    """
    weights = [weight_technical, weight_aesthetic]
    scores = [technical, aesthetic]
    if emotion_active:
        weights.append(weight_emotion)
        scores.append(emotion)
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Scoring weights must be finite and non-negative.")
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("Component scores must be finite.")
    scale = max(weights)
    if scale == 0:
        return 0.0
    normalized = [w / scale for w in weights]
    return math.fsum(w * s for w, s in zip(normalized, scores)) / math.fsum(normalized)


def load_clip_model(device: str = "cpu") -> SentenceTransformer:
    """Load the CLIP SentenceTransformer model."""
    return SentenceTransformer("clip-ViT-B-32", device=device)


def encode_reference_text(
    clip_model: SentenceTransformer,
    text: str = "a beautiful photograph",
) -> torch.Tensor:
    """Encode reference text once so it can be reused for every image."""
    return clip_model.encode([text], convert_to_tensor=True)


def load_image(
    path: Path,
    max_dimension: int = MAX_ANALYSIS_DIMENSION,
) -> tuple[Image.Image | None, np.ndarray | None]:
    """Read an image once and return both PIL (RGB) and OpenCV (BGR) arrays.

    Applies EXIF orientation transpose so phone/camera photos are correctly rotated.
    After full decoding and orientation, images exceeding *max_dimension* are
    downsampled to limit retained pixel buffers; transient decoding and integer
    conversion still allocate memory proportional to the source dimensions.
    Unsigned 16-bit grayscale is normalized. Signed/32-bit integer TIFFs and
    floating-point grayscale are rejected instead of being silently rescaled.
    Returns ``(pil_img, cv_img)`` or ``(None, None)`` on failure.
    """
    try:
        with Image.open(path, formats=SUPPORTED_FORMATS) as raw_img:
            oriented = ImageOps.exif_transpose(raw_img)
            if oriented.mode.startswith("I;16") or oriented.mode == "I":
                # Mode I also represents signed and 32-bit TIFFs. Validate
                # their encoded sample type before applying unsigned 16-bit scaling.
                if raw_img.format == "TIFF" and (
                    raw_img.tag_v2.get(258) != (16,)  # BitsPerSample
                    or raw_img.tag_v2.get(339, (1,)) != (1,)  # SampleFormat: unsigned
                ):
                    raise ValueError(
                        "Only unsigned 16-bit integer grayscale images are supported."
                    )
                samples = np.asarray(oriented)
                if np.any(samples < 0) or np.any(samples > 65535):
                    raise ValueError(
                        "Only unsigned 16-bit integer grayscale images are supported."
                    )
                pixels = np.rint(samples.astype(np.float64) * (255.0 / 65535.0)).astype(
                    np.uint8
                )
                pil_img = Image.fromarray(pixels).convert("RGB")
            elif oriented.mode == "F":
                raise ValueError(
                    "Floating-point HDR images require explicit tone mapping."
                )
            else:
                pil_img = oriented.convert("RGB")
            if max_dimension is not None and max(pil_img.size) > max_dimension:
                pil_img.thumbnail(
                    (max_dimension, max_dimension),
                    Image.Resampling.LANCZOS,
                )
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError) as e:
        logger.warning("Could not read image %s: %s", path.name, e)
        return None, None
