# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.perception
==================

Perception Layer for Cognize
----------------------------
Converts raw multi-modal inputs (text, images, sensors) into a single,
normalized evidence vector that an `EpistemicState` can consume.

Design goals
------------
- Pluggable encoders: bring your own text / image / sensor encoders.
- Shape safety: vectors are float, 1-D, and aligned to a common dimension.
- Weighted fusion: supports modality weights and per-sample confidences.
- Deterministic normalization: L2 normalize (modalities + fused output) for Θ stability.
- Telemetry hooks: optional lightweight introspection with `explain()`.
- Batch-first: `process_batch([...])` to amortize encoder overhead.
- Optional running calibration for sensor streams (EMA mean/var).
- Optional LRU cache for **text** encodings (most common hot path).
- **Pre-encoded passthrough**: accept a ready-made vector via key `"vec"`.
- **Per-call weights**: optional `{"weights": {...}}` to override fusion weights.

Compatibility
-------------
- Interfaces align with `EpistemicState.receive(...)` (expects `.process(dict)->np.ndarray`).
- Pure Python + NumPy only; no framework dependencies.

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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Sequence, List
import numpy as np
from collections import OrderedDict

__all__ = [
    "PerceptionError",
    "PerceptionConfig",
    "RunningCalibrator",
    "Perception",
]

Vector = np.ndarray
Encoder = Callable[[Any], Vector]
FusionFn = Callable[[Dict[str, Vector], Dict[str, float]], Vector]


# ---------------------------
# Errors
# ---------------------------

class PerceptionError(Exception):
    """Unified error class for perception-related failures."""
    pass


# ---------------------------
# Config
# ---------------------------

@dataclass
class PerceptionConfig:
    """Configuration for fusion & normalization."""
    # Target embedding dimension; if None, inferred from first available vector
    target_dim: Optional[int] = None
    # Per-modality weights (multiplicative); missing keys default to 1.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0, "image": 1.0, "sensor": 1.0, "vec": 1.0
    })
    # If true, L2-normalize each modality vector before fusion
    norm_each: bool = True
    # If true, L2-normalize the fused vector
    norm_output: bool = True
    # How to handle dim mismatch: "pad" with zeros or "truncate" to target_dim
    dim_strategy: str = "pad"  # "pad" | "truncate"
    # Small epsilon for numeric stability
    eps: float = 1e-9
    # Optional: clip fused output to [-clip, clip] before final normalization (0 disables)
    clip_value: float = 0.0
    # Optional running calibration for sensor vectors (before per-modality norm)
    use_calibration: bool = False
    calib_alpha: float = 0.05  # EMA factor for mean/var


# ---------------------------
# Helpers
# ---------------------------

def _as_float_vec(x: Any) -> Vector:
    v = np.asarray(x, dtype=float)
    if v.ndim != 1:
        v = v.reshape(-1)
    # replace non-finite with 0 to keep kernels stable
    if not np.isfinite(v).all():
        v = np.where(np.isfinite(v), v, 0.0)
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
    raise PerceptionError("dim_strategy must be 'pad' or 'truncate'")


def _default_fusion(vectors: Dict[str, Vector], weights: Dict[str, float]) -> Vector:
    """Weighted mean of modality vectors (expects shape-aligned inputs)."""
    if not vectors:
        raise PerceptionError("No vectors to fuse.")
    W = []
    V = []
    for k, v in vectors.items():
        w = float(weights.get(k, 1.0))
        if w <= 0.0:
            continue
        W.append(w)
        V.append(v * w)
    if not V:
        raise PerceptionError("All modality weights are zero or negative.")
    return np.sum(V, axis=0) / (np.sum(W) or 1.0)


# ---------------------------
# Running calibrator (optional)
# ---------------------------

@dataclass
class RunningCalibrator:
    """EMA mean/var normalizer for a single modality stream (e.g., sensors)."""
    alpha: float = 0.05
    eps: float = 1e-9
    mean: Optional[Vector] = None
    var: Optional[Vector] = None

    def update(self, v: Vector) -> None:
        v = _as_float_vec(v)
        if self.mean is None:
            self.mean = v.copy()
            self.var = np.ones_like(v)
            return
        self.mean = (1 - self.alpha) * self.mean + self.alpha * v
        diff = v - self.mean
        self.var = (1 - self.alpha) * self.var + self.alpha * (diff * diff)

    def apply(self, v: Vector) -> Vector:
        if self.mean is None or self.var is None:
            return v
        std = np.sqrt(np.maximum(self.var, self.eps))
        return (v - self.mean) / std


# ---------------------------
# Perception
# ---------------------------

class Perception:
    """
    Perception(text_encoder, image_encoder, sensor_fusion_fn, fusion_fn, config,
               text_cache_size=0)

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
        Fusion/normalization configuration. If omitted, defaults are used.
    text_cache_size : int
        Size of LRU cache for text encodings (0 disables). Only affects `text_encoder`.
    """

    def __init__(
        self,
        text_encoder: Optional[Encoder] = None,
        image_encoder: Optional[Encoder] = None,
        sensor_fusion_fn: Optional[Encoder] = None,
        fusion_fn: Optional[FusionFn] = None,
        config: Optional[PerceptionConfig] = None,
        text_cache_size: int = 0,
    ):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.sensor_fusion_fn = sensor_fusion_fn
        self.fusion_fn = fusion_fn or _default_fusion
        self.config = config or PerceptionConfig()

        # optional calibrator for sensor streams
        self._sensor_cal: Optional[RunningCalibrator] = (
            RunningCalibrator(self.config.calib_alpha) if self.config.use_calibration else None
        )

        # simple LRU for text
        self._text_cache: Optional[OrderedDict[str, Vector]] = None
        self._text_cache_cap = max(0, int(text_cache_size))
        if self._text_cache_cap > 0:
            self._text_cache = OrderedDict()

    # ---- internals ----

    def _encode_text(self, s: Any) -> Optional[Vector]:
        enc = self.text_encoder
        if enc is None or s is None:
            return None
        s = str(s)
        if self._text_cache is not None:
            v = self._text_cache.get(s)
            if v is not None:
                # move to end (recently used)
                self._text_cache.move_to_end(s)
                return v.copy()
        try:
            v = _as_float_vec(enc(s))
        except Exception as e:
            raise PerceptionError(f"text_encoder failed: {e}")
        if self._text_cache is not None:
            self._text_cache[s] = v.copy()
            if len(self._text_cache) > self._text_cache_cap:
                self._text_cache.popitem(last=False)
        return v

    def _encode_image(self, img: Any) -> Optional[Vector]:
        enc = self.image_encoder
        if enc is None or img is None:
            return None
        try:
            return _as_float_vec(enc(img))
        except Exception as e:
            raise PerceptionError(f"image_encoder failed: {e}")

    def _encode_sensor(self, sensor_obj: Any, *, update: bool = True) -> Optional[Vector]:
        enc = self.sensor_fusion_fn
        if enc is None or sensor_obj is None:
            return None
        try:
            v = _as_float_vec(enc(sensor_obj))
        except Exception as e:
            raise PerceptionError(f"sensor_fusion_fn failed: {e}")
        if self._sensor_cal is not None:
            if update:
                self._sensor_cal.update(v)
            v = self._sensor_cal.apply(v)
        return v

    def _determine_target_dim(self, modality_vecs: Dict[str, Vector]) -> int:
        if self.config.target_dim is not None:
            return int(self.config.target_dim)
        # infer from first available vector (dict preserves insertion order)
        for v in modality_vecs.values():
            return int(v.shape[0])
        raise PerceptionError("cannot infer target_dim (no vectors)")

    @staticmethod
    def _apply_confidences(weights: Dict[str, float], confidences: Dict[str, float]) -> Dict[str, float]:
        # combine static weights with confidences in [0,1]
        out: Dict[str, float] = dict(weights or {})
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
            e.g. {"text": "...", "image": img, "sensor": {...}, "vec": np.array([...]),
                  "conf": {"text": 0.8}, "weights": {"text": 2.0}}

        Returns
        -------
        np.ndarray
            1-D float vector, shape-aligned and (optionally) L2-normalized.
        """
        if not isinstance(inputs, dict):
            raise PerceptionError("Perception.process expects a dict of modalities.")

        data = dict(inputs)
        # normalize list[str] for text
        if isinstance(data.get("text"), (list, tuple)):
            data["text"] = " ".join(map(str, data["text"]))

        # pre-encoded direct vector
        direct_vec: Optional[Vector] = None
        if "vec" in data and data["vec"] is not None:
            direct_vec = _as_float_vec(data["vec"])

        # encode present modalities
        modality_vecs: Dict[str, Vector] = {}
        if "text" in data:
            v = self._encode_text(data.get("text"))
            if v is not None:
                modality_vecs["text"] = v
        if "image" in data:
            v = self._encode_image(data.get("image"))
            if v is not None:
                modality_vecs["image"] = v
        if "sensor" in data:
            v = self._encode_sensor(data.get("sensor"), update=True)
            if v is not None:
                modality_vecs["sensor"] = v
        if direct_vec is not None:
            modality_vecs["vec"] = direct_vec

        if not modality_vecs:
            raise PerceptionError("no supported modalities provided or encoders missing")

        # Determine target dimension and align
        target_dim = self._determine_target_dim(modality_vecs)
        aligned: Dict[str, Vector] = {}
        for k, v in modality_vecs.items():
            vv = _align_dim(v, target_dim, self.config.dim_strategy)
            if self.config.norm_each:
                vv = _l2norm(vv, self.config.eps)
            aligned[k] = vv

        # Apply confidences if present + per-call weights override
        confidences = data.get("conf", {}) if isinstance(data.get("conf", {}), dict) else {}
        fused_weights = self._apply_confidences(self.config.weights, confidences)
        if isinstance(data.get("weights"), dict):
            for k, w in data["weights"].items():
                fused_weights[k] = float(fused_weights.get(k, 1.0) * float(w))

        # Fuse
        fused = self.fusion_fn(aligned, fused_weights)
        fused = _as_float_vec(fused)
        fused = _align_dim(fused, target_dim, self.config.dim_strategy)

        # Optional clip (helps with outlier encoders before final norm)
        if self.config.clip_value and self.config.clip_value > 0:
            c = float(self.config.clip_value)
            fused = np.clip(fused, -c, c)

        if self.config.norm_output:
            fused = _l2norm(fused, self.config.eps)

        return fused

    def process_batch(self, batch: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Vectorize a batch of inputs; returns array with shape (B, D)."""
        if not isinstance(batch, (list, tuple)) or not batch:
            raise PerceptionError("process_batch expects a non-empty list of input dicts")

        first = self.process(batch[0])
        # honor target_dim if set; else lock to first vector's dim for determinism
        D = int(self.config.target_dim) if self.config.target_dim is not None else int(first.shape[0])
        vecs: List[Vector] = [ _align_dim(first, D, self.config.dim_strategy) ]
        for x in batch[1:]:
            v = self.process(x)
            vecs.append(_align_dim(v, D, self.config.dim_strategy))
        return np.stack(vecs, axis=0).astype(float)

    # ---- introspection ----

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict explaining per-modality contributions used for fusion (no calibrator updates)."""
        data = dict(inputs)
        if isinstance(data.get("text"), (list, tuple)):
            data["text"] = " ".join(map(str, data["text"]))

        raw: Dict[str, Vector] = {}
        if "text" in data:
            v = self._encode_text(data.get("text"))
            if v is not None:
                raw["text"] = v
        if "image" in data:
            v = self._encode_image(data.get("image"))
            if v is not None:
                raw["image"] = v
        if "sensor" in data:
            v = self._encode_sensor(data.get("sensor"), update=False)  # no state mutation
            if v is not None:
                raw["sensor"] = v
        if "vec" in data and data["vec"] is not None:
            raw["vec"] = _as_float_vec(data["vec"])

        if not raw:
            raise PerceptionError("No encodable modalities.")

        # alignment + (optional) per-modality norm
        D = self._determine_target_dim(raw)
        aligned = {k: _align_dim(v, D, self.config.dim_strategy) for k, v in raw.items()}
        if self.config.norm_each:
            aligned = {k: _l2norm(v, self.config.eps) for k, v in aligned.items()}

        confidences = data.get("conf", {}) if isinstance(data.get("conf", {}), dict) else {}
        fused_weights = self._apply_confidences(self.config.weights, confidences)
        if isinstance(data.get("weights"), dict):
            for k, w in data["weights"].items():
                fused_weights[k] = float(fused_weights.get(k, 1.0) * float(w))

        # fused vector & per-modality contributions prior to final normalization
        contribs = {k: aligned[k] * float(fused_weights.get(k, 1.0)) for k in aligned}
        denom = (sum(fused_weights.get(k, 1.0) for k in aligned) or 1.0)
        fused = sum(contribs.values()) / denom
        if self.config.clip_value and self.config.clip_value > 0:
            c = float(self.config.clip_value)
            fused = np.clip(fused, -c, c)
        fused_out = _l2norm(fused, self.config.eps) if self.config.norm_output else fused

        return {
            "target_dim": int(D),
            "weights": {k: float(fused_weights.get(k, 1.0)) for k in aligned},
            "aligned": {k: aligned[k].tolist() for k in aligned},
            "contribs": {k: contribs[k].tolist() for k in contribs},
            "fused_pre_norm": fused.tolist(),
            "fused": fused_out.tolist(),
        }
