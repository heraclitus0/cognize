# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.perception
==================

Perception Layer for Cognize
----------------------------
Converts raw multi-modal inputs (text, images, sensors) into a single,
normalized evidence vector that an `EpistemicState` can consume.

What it gives you
-----------------
- **Pluggable encoders**: bring your own text / image / sensor encoders.
- **Shape safety**: vectors are float, 1-D, and aligned to a common dimension.
- **Fusion**: weighted fusion across modalities (supports per-sample confidences).
- **Deterministic normalization**: L2 normalize (modalities + fused output) for Θ stability.

Supported inputs to `Perception.process`
---------------------------------------
- `"text"`: `str` or `list[str]` (lists are joined with spaces)
- `"image"`: any type your `image_encoder` accepts
- `"sensor"`: `dict`/object your `sensor_fusion_fn` accepts
- Optional `"conf"`: `{"text": c_t, "image": c_i, "sensor": c_s}` with confidences in [0, 1]

Quick start
-----------
>>> import numpy as np
>>> def toy_text_encoder(s: str) -> np.ndarray:  # 4-dim toy embedding
...     return np.array([len(s), s.count(' '), s.count('a'), 1.0], dtype=float)
>>> P = Perception(text_encoder=toy_text_encoder)
>>> v = P.process({"text": "hello world"})
>>> v.shape
(4,)
>>> float(np.linalg.norm(v))  # L2-normalized by default
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Sequence
import numpy as np

__all__ = [
    "PerceptionConfig",
    "Perception",
]

Vector = np.ndarray
Encoder = Callable[[Any], Vector]
FusionFn = Callable[[Dict[str, Vector], Dict[str, float]], Vector]


# ---------------------------
# Config
# ---------------------------

@dataclass
class PerceptionConfig:
    """Configuration for fusion & normalization."""
    # Target embedding dimension; if None, inferred from first available vector
    target_dim: Optional[int] = None
    # Per-modality weights (multiplicative); missing keys default to 1.0
    weights: Dict[str, float] = None
    # If true, L2-normalize each modality vector before fusion
    norm_each: bool = True
    # If true, L2-normalize the fused vector
    norm_output: bool = True
    # How to handle dim mismatch: "pad" with zeros or "truncate" to target_dim
    dim_strategy: str = "pad"  # "pad" | "truncate"
    # Small epsilon for numeric stability
    eps: float = 1e-9


# ---------------------------
# Helpers
# ---------------------------

def _as_float_vec(x: Any) -> Vector:
    v = np.asarray(x, dtype=float)
    if v.ndim != 1:
        v = v.reshape(-1)
    return v


def _l2norm(v: Vector, eps: float) -> Vector:
    n = float(np.linalg.norm(v))
    return v if n <= eps else (v / n)


def _align_dim(v: Vector, target_dim: int, mode: str) -> Vector:
    if v.shape[0] == target_dim:
        return v
    if mode == "pad":
        if v.shape[0] < target_dim:
            pad = np.zeros(target_dim - v.shape[0], dtype=float)
            return np.concatenate([v, pad], axis=0)
        else:
            return v[:target_dim]
    if mode == "truncate":
        return v[:target_dim] if v.shape[0] >= target_dim else np.pad(v, (0, target_dim - v.shape[0]), mode="constant")
    raise ValueError("dim_strategy must be 'pad' or 'truncate'")


def _default_fusion(vectors: Dict[str, Vector], weights: Dict[str, float]) -> Vector:
    """Weighted mean of modality vectors (expects shape-aligned inputs)."""
    if not vectors:
        raise ValueError("No vectors to fuse.")
    W = []
    V = []
    for k, v in vectors.items():
        w = float(weights.get(k, 1.0))
        if w <= 0.0:
            continue
        W.append(w)
        V.append(v * w)
    if not V:
        raise ValueError("All modality weights are zero or negative.")
    return np.sum(V, axis=0) / (np.sum(W) or 1.0)


# ---------------------------
# Perception
# ---------------------------

class Perception:
    """
    Perception(text_encoder, image_encoder, sensor_fusion_fn, fusion_fn, config)

    Parameters
    ----------
    text_encoder : Callable[[str], np.ndarray], optional
        Your text -> vector encoder (1-D).
    image_encoder : Callable[[Any], np.ndarray], optional
        Your image -> vector encoder (1-D).
    sensor_fusion_fn : Callable[[Any], np.ndarray], optional
        Your sensor(s) -> vector encoder (1-D).
    fusion_fn : Callable[[Dict[str, Vector], Dict[str, float]], Vector], optional
        Function that fuses aligned modality vectors into a single vector.
        Defaults to a weighted mean.
    config : PerceptionConfig, optional
        Fusion/normalization configuration. If omitted, sensible defaults are used.
    """

    def __init__(
        self,
        text_encoder: Optional[Encoder] = None,
        image_encoder: Optional[Encoder] = None,
        sensor_fusion_fn: Optional[Encoder] = None,
        fusion_fn: Optional[FusionFn] = None,
        config: Optional[PerceptionConfig] = None,
    ):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.sensor_fusion_fn = sensor_fusion_fn
        self.fusion_fn = fusion_fn or _default_fusion
        self.config = config or PerceptionConfig(weights={"text": 1.0, "image": 1.0, "sensor": 1.0})

    # ---- internals ----

    def _encode_modality(self, key: str, value: Any) -> Optional[Vector]:
        enc = {
            "text": self.text_encoder,
            "image": self.image_encoder,
            "sensor": self.sensor_fusion_fn,
        }.get(key)
        if enc is None:
            return None
        try:
            vec = _as_float_vec(enc(value))
            return vec
        except Exception as e:
            raise RuntimeError(f"Perception: encoder for '{key}' failed: {e}")

    def _determine_target_dim(self, modality_vecs: Dict[str, Vector]) -> int:
        if self.config.target_dim is not None:
            return int(self.config.target_dim)
        # infer from first available vector (dict preserves insertion order)
        for v in modality_vecs.values():
            return int(v.shape[0])
        raise ValueError("Perception: cannot infer target_dim (no vectors).")

    def _apply_confidences(self, weights: Dict[str, float], confidences: Dict[str, float]) -> Dict[str, float]:
        # combine static weights with confidences in [0,1]
        out = dict(weights or {})
        for k, c in (confidences or {}).items():
            c = float(np.clip(c, 0.0, 1.0))
            out[k] = out.get(k, 1.0) * c
        return out

    # ---- public API ----

    def process(self, inputs: Dict[str, Any]) -> Vector:
        """
        Encode and fuse multi-modal inputs into a single evidence vector.

        Parameters
        ----------
        inputs : dict
            e.g. {"text": "...", "image": img, "sensor": {...}, "conf": {"text": 0.8}}

        Returns
        -------
        np.ndarray
            1-D float vector, shape-aligned and (optionally) L2-normalized.
        """
        if not isinstance(inputs, dict):
            raise ValueError("Perception.process expects a dict of modalities.")

        # Normalize list[str] for text
        data = dict(inputs)
        if isinstance(data.get("text"), (list, tuple)):
            data["text"] = " ".join(map(str, data["text"]))

        # Encode present modalities
        modality_vecs: Dict[str, Vector] = {}
        for key in ("text", "image", "sensor"):
            if key in data:
                v = self._encode_modality(key, data[key])
                if v is not None:
                    modality_vecs[key] = v

        if not modality_vecs:
            raise ValueError("Perception: no supported modalities provided or encoders missing.")

        # Determine target dimension and align
        target_dim = self._determine_target_dim(modality_vecs)
        aligned: Dict[str, Vector] = {}
        for k, v in modality_vecs.items():
            vv = _align_dim(v, target_dim, self.config.dim_strategy)
            if self.config.norm_each:
                vv = _l2norm(vv, self.config.eps)
            aligned[k] = vv

        # Apply confidences if present
        confidences = data.get("conf", {}) if isinstance(data.get("conf", {}), dict) else {}
        fused_weights = self._apply_confidences(self.config.weights or {}, confidences)

        # Fuse and normalize output
        fused = self.fusion_fn(aligned, fused_weights)
        fused = _as_float_vec(fused)
        fused = _align_dim(fused, target_dim, self.config.dim_strategy)

        if self.config.norm_output:
            fused = _l2norm(fused, self.config.eps)

        return fused
