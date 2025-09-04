# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.perception
==================

Perception — fuse text/image/sensor into one safe, normalized vector.

Design
------
- Bring-your-own encoders (text / image / sensor) or pass a ready vector via "vec".
- Shape-safe: 1-D float vectors, aligned to a common dim with pad/truncate (or strict).
- Deterministic: optional L2-norm on each modality and on the fused output.
- Weighted: static per-modality weights + per-sample confidences; per-call overrides.
- Resilient: NaN/Inf sanitized; fusion fallback if a custom fusion_fn fails.
- Calibrated: optional EMA mean/var for sensor streams.
- Inspectable: `explain()` and lightweight `trace` buffer; pluggable hooks.

Quick start
-----------
>>> import numpy as np
>>> def toy_text(s: str) -> np.ndarray:
...     return np.array([len(s), s.count(' '), s.count('a'), 1.0], float)
>>> from cognize.perception import Perception, PerceptionConfig
>>> P = Perception(text_encoder=toy_text, config=PerceptionConfig(trace=True))
>>> v = P.process({"text": "hello world"})
>>> v.shape, round(float(np.linalg.norm(v)), 5)
((4,), 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, List
import numpy as np
from collections import OrderedDict, deque

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
    """Unified error for perception-related failures."""


# ---------------------------
# Config
# ---------------------------

@dataclass
class PerceptionConfig:
    """Fusion & normalization configuration."""
    # Target embedding dimension; if None, inferred from first available vector
    target_dim: Optional[int] = None
    # Per-modality weights (multiplicative); missing keys default to 1.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0, "image": 1.0, "sensor": 1.0, "vec": 1.0
    })
    # Normalize each modality vector before fusion
    norm_each: bool = True
    # Normalize the fused vector
    norm_output: bool = True
    # Dim mismatch handling: "pad" | "truncate"
    dim_strategy: str = "pad"
    # Numeric stability epsilon
    eps: float = 1e-9
    # Optional clip for fused output before final normalization (0 disables)
    clip_value: float = 0.0
    # Sensor EMA calibration
    use_calibration: bool = False
    calib_alpha: float = 0.05
    # If True, reject dimension mismatch instead of pad/truncate
    strict_dim: bool = False
    # Debug trace: keep last N processed items with aligned/fused vectors
    trace: bool = False
    trace_cap: int = 50


# ---------------------------
# Helpers
# ---------------------------

def _sanitize_vec(x: Any) -> Vector:
    """Convert to 1-D float vector, replace non-finite with 0."""
    v = np.asarray(x, dtype=float).reshape(-1)
    if not np.isfinite(v).all():
        v = np.where(np.isfinite(v), v, 0.0)
    return v


def _l2norm(v: Vector, eps: float) -> Vector:
    n = float(np.linalg.norm(v))
    return v if n <= eps else (v / n)


def _align_dim(v: Vector, target_dim: int, mode: str, strict: bool) -> Vector:
    if v.shape[0] == target_dim:
        return v
    if strict:
        raise PerceptionError(f"dim mismatch: got {v.shape[0]}, expected {target_dim}")
    if mode == "pad":
        return np.pad(v, (0, max(0, target_dim - v.shape[0])), mode="constant")[:target_dim]
    if mode == "truncate":
        return v[:target_dim] if v.shape[0] >= target_dim else np.pad(v, (0, target_dim - v.shape[0]), mode="constant")
    raise PerceptionError("dim_strategy must be 'pad' or 'truncate'")


def _default_fusion(vectors: Dict[str, Vector], weights: Dict[str, float]) -> Vector:
    """Weighted mean (expects shape-aligned vectors)."""
    if not vectors:
        raise PerceptionError("No vectors to fuse.")
    V, W = [], []
    for k, v in vectors.items():
        w = float(weights.get(k, 1.0))
        if w > 0.0:
            V.append(v * w)
            W.append(w)
    if not V:
        raise PerceptionError("All modality weights are ≤ 0.")
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
        try:
            v = _sanitize_vec(v)
            if self.mean is None:
                self.mean = v.copy()
                self.var = np.ones_like(v)
                return
            self.mean = (1 - self.alpha) * self.mean + self.alpha * v
            diff = v - self.mean
            self.var = (1 - self.alpha) * self.var + self.alpha * (diff * diff)
        except Exception:
            # Fail closed: never propagate calibrator errors
            pass

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
    Perception(text_encoder=None, image_encoder=None, sensor_fusion_fn=None,
               fusion_fn=None, config=None, text_cache_size=0)

    Encodes available modalities, aligns dims, applies weights/confidences,
    fuses safely, and returns a single 1-D float vector.
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

        # Optional calibrator for sensor streams
        self._sensor_cal: Optional[RunningCalibrator] = (
            RunningCalibrator(self.config.calib_alpha) if self.config.use_calibration else None
        )

        # Simple LRU for text
        self._text_cache: Optional[OrderedDict[str, Vector]] = None
        self._text_cache_cap = max(0, int(text_cache_size))
        if self._text_cache_cap > 0:
            self._text_cache = OrderedDict()

        # Trace + hooks
        self._trace = deque(maxlen=int(max(1, self.config.trace_cap))) if self.config.trace else None
        self._hooks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    # ---- hooks ----

    def register_hook(self, event: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to 'on_process' or future events."""
        self._hooks.setdefault(event, []).append(fn)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for fn in self._hooks.get(event, []):
            try:
                fn(dict(payload))
            except Exception:
                # Hooks must never destabilize the pipeline
                pass

    # ---- encoders ----

    def _encode_text(self, s: Any) -> Optional[Vector]:
        enc = self.text_encoder
        if enc is None or s is None:
            return None
        s = str(s)
        if self._text_cache is not None:
            v = self._text_cache.get(s)
            if v is not None:
                # Move to MRU
                self._text_cache.move_to_end(s)
                return v.copy()
        try:
            v = _sanitize_vec(enc(s))
        except Exception as e:
            raise PerceptionError(f"text_encoder failed: {e}")
        if self._text_cache is not None:
            self._text_cache[s] = v.copy()
            if len(self._text_cache) > self._text_cache_cap:
                # Pop LRU
                self._text_cache.popitem(last=False)
        return v

    def _encode_image(self, img: Any) -> Optional[Vector]:
        enc = self.image_encoder
        if enc is None or img is None:
            return None
        try:
            return _sanitize_vec(enc(img))
        except Exception as e:
            raise PerceptionError(f"image_encoder failed: {e}")

    def _encode_sensor(self, sensor_obj: Any, *, update: bool = True) -> Optional[Vector]:
        enc = self.sensor_fusion_fn
        if enc is None or sensor_obj is None:
            return None
        try:
            v = _sanitize_vec(enc(sensor_obj))
        except Exception as e:
            raise PerceptionError(f"sensor_fusion_fn failed: {e}")
        if self._sensor_cal is not None:
            if update:
                self._sensor_cal.update(v)
            v = self._sensor_cal.apply(v)
        return v

    # ---- internals ----

    def _determine_target_dim(self, modality_vecs: Dict[str, Vector]) -> int:
        if self.config.target_dim is not None:
            return int(self.config.target_dim)
        for v in modality_vecs.values():
            return int(v.shape[0])
        raise PerceptionError("cannot infer target_dim (no vectors)")

    @staticmethod
    def _apply_confidences(weights: Dict[str, float], confidences: Dict[str, float]) -> Dict[str, float]:
        """Combine static weights with confidences in [0,1]."""
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
            Example:
            {
              "text": "...", "image": img, "sensor": {...}, "vec": np.array([...]),
              "conf": {"text": 0.8}, "weights": {"text": 2.0}
            }

        Returns
        -------
        np.ndarray
            1-D float vector, shape-aligned and (optionally) L2-normalized.
        """
        if not isinstance(inputs, dict):
            raise PerceptionError("Perception.process expects a dict of modalities.")

        data = dict(inputs)
        # Normalize list[str] for text
        if isinstance(data.get("text"), (list, tuple)):
            data["text"] = " ".join(map(str, data["text"]))

        # Encode present modalities
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
        if "vec" in data and data["vec"] is not None:
            modality_vecs["vec"] = _sanitize_vec(data["vec"])

        if not modality_vecs:
            raise PerceptionError("no supported modalities provided or encoders missing")

        # Determine target dimension and align
        target_dim = self._determine_target_dim(modality_vecs)
        aligned: Dict[str, Vector] = {}
        for k, v in modality_vecs.items():
            vv = _align_dim(v, target_dim, self.config.dim_strategy, self.config.strict_dim)
            if self.config.norm_each:
                vv = _l2norm(vv, self.config.eps)
            aligned[k] = vv

        # Apply confidences + per-call weight overrides
        confidences = data.get("conf", {}) if isinstance(data.get("conf", {}), dict) else {}
        fused_weights = self._apply_confidences(self.config.weights, confidences)
        if isinstance(data.get("weights"), dict):
            for k, w in data["weights"].items():
                fused_weights[k] = float(fused_weights.get(k, 1.0) * float(w))

        # Fuse (with safe fallback)
        try:
            fused = self.fusion_fn(aligned, fused_weights)
        except Exception:
            fused = _default_fusion(aligned, fused_weights)

        fused = _sanitize_vec(fused)
        fused = _align_dim(fused, target_dim, self.config.dim_strategy, self.config.strict_dim)

        # Optional clip → final norm
        if self.config.clip_value and self.config.clip_value > 0:
            c = float(self.config.clip_value)
            fused = np.clip(fused, -c, c)
        if self.config.norm_output:
            fused = _l2norm(fused, self.config.eps)

        # Trace + hooks
        if self._trace is not None:
            # Store shallow copies to keep memory in check
            self._trace.append({"inputs": {k: (None if k in ("image",) else data[k]) for k in data},
                                "aligned": {k: aligned[k].copy() for k in aligned},
                                "fused": fused.copy()})
        self._emit("on_process", {"fused": fused.copy()})

        return fused.copy()

    def process_batch(self, batch: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Vectorize a batch of inputs; returns array with shape (B, D)."""
        if not isinstance(batch, (list, tuple)) or not batch:
            raise PerceptionError("process_batch expects a non-empty list of input dicts")

        first = self.process(batch[0])
        D = int(self.config.target_dim) if self.config.target_dim is not None else int(first.shape[0])
        out = np.zeros((len(batch), D), dtype=float)
        out[0] = first
        for i, x in enumerate(batch[1:], start=1):
            out[i] = _align_dim(self.process(x), D, self.config.dim_strategy, self.config.strict_dim)
        return out

    # ---- introspection ----

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain per-modality contributions and fused vector (uses `process`).
        Returns target_dim, weights, aligned vectors, contributions, and fused outputs.
        """
        v = self.process(inputs)  # ensures identical path/guards
        last = (self._trace[-1] if self._trace else None)

        # Recompute contributions with the same weights on the aligned cache
        if last is not None:
            aligned = last["aligned"]
        else:
            # Fallback: recompute aligned from inputs without mutating calibrator
            data = dict(inputs)
            if isinstance(data.get("text"), (list, tuple)):
                data["text"] = " ".join(map(str, data["text"]))
            raw: Dict[str, Vector] = {}
            if "text" in data:
                tv = self._encode_text(data.get("text"));  raw.update(text=tv) if tv is not None else None
            if "image" in data:
                iv = self._encode_image(data.get("image")); raw.update(image=iv) if iv is not None else None
            if "sensor" in data:
                sv = self._encode_sensor(data.get("sensor"), update=False); raw.update(sensor=sv) if sv is not None else None
            if "vec" in data and data["vec"] is not None:
                raw["vec"] = _sanitize_vec(data["vec"])
            if not raw:
                raise PerceptionError("No encodable modalities.")
            D = self._determine_target_dim(raw)
            aligned = {k: _align_dim(_l2norm(_sanitize_vec(raw[k]), self.config.eps) if self.config.norm_each else _sanitize_vec(raw[k]),
                                     D, self.config.dim_strategy, self.config.strict_dim) for k in raw}

        confidences = inputs.get("conf", {}) if isinstance(inputs.get("conf", {}), dict) else {}
        weights = self._apply_confidences(self.config.weights, confidences)
        if isinstance(inputs.get("weights"), dict):
            for k, w in inputs["weights"].items():
                weights[k] = float(weights.get(k, 1.0) * float(w))

        denom = (sum(weights.get(k, 1.0) for k in aligned) or 1.0)
        contribs = {k: aligned[k] * float(weights.get(k, 1.0)) for k in aligned}
        fused_pre = sum(contribs.values()) / denom
        fused_pre = _sanitize_vec(fused_pre)
        if self.config.clip_value and self.config.clip_value > 0:
            c = float(self.config.clip_value)
            fused_pre = np.clip(fused_pre, -c, c)
        fused_out = _l2norm(fused_pre, self.config.eps) if self.config.norm_output else fused_pre

        return {
            "target_dim": int(len(v)),
            "weights": {k: float(weights.get(k, 1.0)) for k in aligned},
            "aligned": {k: aligned[k].tolist() for k in aligned},
            "contribs": {k: contribs[k].tolist() for k in contribs},
            "fused_pre_norm": fused_pre.tolist(),
            "fused": fused_out.tolist(),
            "trace": last,  # includes shallow copies of inputs/aligned/fused if trace=True
        }
