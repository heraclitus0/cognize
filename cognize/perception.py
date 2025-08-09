# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
Perception Layer for Cognize
============================
Converts raw multi-modal inputs (text, images, sensors) into
normalized evidence vectors usable by EpistemicState instances.

Design goals
------------
- Encoders are optional callables you provide (no heavy deps here).
- Handles 'text', 'image', 'sensor' keys; easily extensible.
- Vector-safe: dtype=float, shape-aligned via pad/truncate.
- Confidence & weights: fuse with per-modality weights and optional confidences.
- Deterministic normalization for Θ stability (L2 normalize output).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Sequence
import numpy as np


Vector = np.ndarray
Encoder = Callable[[Any], Vector]
FusionFn = Callable[[Dict[str, Vector], Dict[str, float]], Vector]


@dataclass
class PerceptionConfig:
    """Configuration for fusion & normalization."""
    # Target embedding dimension; if None, inferred from first available vector
    target_dim: Optional[int] = None
    # Per-modality weights (multiplicative)
    weights: Dict[str, float] = None
    # If true, L2-normalize each modality vector before fusion
    norm_each: bool = True
    # If true, L2-normalize the fused vector
    norm_output: bool = True
    # How to handle dim mismatch: "pad" with zeros or "truncate" to target_dim
    dim_strategy: str = "pad"  # "pad" | "truncate"
    # Small epsilon for numeric stability
    eps: float = 1e-9


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
    elif mode == "truncate":
        return v[:target_dim] if v.shape[0] >= target_dim else np.pad(v, (0, target_dim - v.shape[0]), mode="constant")
    else:
        raise ValueError("dim_strategy must be 'pad' or 'truncate'")


def _default_fusion(vectors: Dict[str, Vector], weights: Dict[str, float]) -> Vector:
    # Weighted mean of modality vectors
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


class Perception:
    """
    Perception(text_encoder, image_encoder, sensor_fusion_fn, fusion_fn, config)

    - text_encoder(str)   -> 1D np.ndarray
    - image_encoder(img)  -> 1D np.ndarray
    - sensor_fusion_fn(dict/obj) -> 1D np.ndarray
    - fusion_fn(modality_vectors, weights) -> fused vector (same dim)
    """

    def __init__(self,
                 text_encoder: Optional[Encoder] = None,
                 image_encoder: Optional[Encoder] = None,
                 sensor_fusion_fn: Optional[Encoder] = None,
                 fusion_fn: Optional[FusionFn] = None,
                 config: Optional[PerceptionConfig] = None):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.sensor_fusion_fn = sensor_fusion_fn
        self.fusion_fn = fusion_fn or _default_fusion
        self.config = config or PerceptionConfig(weights={"text": 1.0, "image": 1.0, "sensor": 1.0})

    def _encode_modality(self, key: str, value: Any) -> Optional[Vector]:
        enc = {"text": self.text_encoder, "image": self.image_encoder, "sensor": self.sensor_fusion_fn}.get(key)
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
        # infer from first available vector
        for v in modality_vecs.values():
            return int(v.shape[0])
        raise ValueError("Perception: cannot infer target_dim (no vectors).")

    def _apply_confidences(self, weights: Dict[str, float], confidences: Dict[str, float]) -> Dict[str, float]:
        # combine static weights with confidences in [0,1]
        out = dict(weights)
        for k, c in confidences.items():
            c = float(np.clip(c, 0.0, 1.0))
            out[k] = out.get(k, 1.0) * c
        return out

    def process(self, inputs: Dict[str, Any]) -> Vector:
        """
        Process multi-modal inputs into a single evidence vector.

        Supported keys:
          - 'text': str or list[str] (concats with spaces)
          - 'image': any object your image_encoder accepts
          - 'sensor': dict/obj your sensor_fusion_fn accepts
          - optional 'conf': {'text': c_t, 'image': c_i, 'sensor': c_s}  # confidences [0,1]
        """
        if not isinstance(inputs, dict):
            raise ValueError("Perception.process expects a dict of modalities.")

        # Handle list[str] for text
        data = dict(inputs)
        if isinstance(data.get("text"), (list, tuple)):
            data["text"] = " ".join(map(str, data["text"]))

        # Encode available modalities
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

        # Fuse
        fused = self.fusion_fn(aligned, fused_weights)
        fused = _as_float_vec(fused)
        fused = _align_dim(fused, target_dim, self.config.dim_strategy)

        if self.config.norm_output:
            fused = _l2norm(fused, self.config.eps)

        return fused
