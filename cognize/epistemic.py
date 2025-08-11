# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
EpistemicState — Continuity Theory aligned cognition kernel
===========================================================

Implements the core loop of Continuity Theory (CT) + RCC primitives:

Symbols (with ASCII fallbacks):
- ⊙  (CONTINUITY): Continuity Monad (recursive realignment operator)
- ∆  (DISTORTION): Distortion magnitude between projection V and reception R
- S̄  (DIVERGENCE): Projected divergence (directional deviation of reception vs projection)
- Θ  (THRESHOLD):  Rupture threshold (structured failure boundary)
- E  (MISALIGNMENT): Cumulative misalignment memory
- ∅  (EMPTY): No cognition received
- ⚠  (RUPTURE): Rupture detected
- Ĉ  (COGNITION): Cognitive agent (optional) that edits memory and/or policies

Key design goals
----------------
1) Keep the *core loop* pure and auditable:
   V(t) → R(t) → ∆(t) → (⊙ if absorbable) else Θ; cognition may mutate V or policy.

2) Make everything *pluggable*:
   - Perception adapters (text/image/sensor) → normalized evidence vectors (R)
   - Policies (threshold/realign/collapse/divergence) as callables
   - Meta-policy manager (runtime selection/promotion with safety)
   - Memory, safety monitors, goal manager: optional satellites

3) Operate on scalars *and* vectors safely (shape checks, bounded gains).

4) Determinism when needed (local RNG), and rich telemetry for explainability.

This file intentionally hosts the kernel + thin integration hooks so downstream modules
(perception/network/meta/memory/safety/goals) can attach without bloating imports.
"""

from __future__ import annotations

__author__ = "Pulikanti Sashi Bharadwaj"
__license__ = "Apache-2.0"
__version__ = "0.1.6"

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import uuid
import json
import csv
import copy
import math
import warnings

# ---------------------------
# Typing shorthands
# ---------------------------
Number = Union[int, float, np.number]
Vector = np.ndarray
Evidence = Union[Number, Vector, Dict[str, Any]]  # Dict → routed via Perception if present

# ---------------------------
# Utility helpers
# ---------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _l2_norm(v: Vector) -> float:
    n = float(np.linalg.norm(v))
    return n if n != 0.0 else 1.0

def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))

def _sanitize(v: Any) -> Any:
    """Make values JSON-/CSV-serializable and human-friendly."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, (str, bool)) or v is None:
        return v
    try:
        return float(v)
    except Exception:
        return str(v)

# ---------------------------
# Default policy primitives (safe templates)
# ---------------------------

# Threshold policies Θ(E, t, context)
def threshold_static(state: "EpistemicState") -> float:
    """Constant threshold (baseline)."""
    return float(state.Θ)

def threshold_adaptive(state: "EpistemicState", a: float = 0.05) -> float:
    """Θ_t = base + a * E ; bounded for stability."""
    return float(state.Θ + _clip(a * state.E, 0.0, 1.0))

def threshold_stochastic(state: "EpistemicState", base_sigma: float = 0.01) -> float:
    """Adds Gaussian noise to baseline threshold using state's local RNG."""
    sigma = _clip(base_sigma, 0.0, 0.2)
    return float(state.Θ + state.rng.normal(0.0, sigma))

def threshold_combined(state: "EpistemicState", a: float = 0.05, sigma: float = 0.01) -> float:
    """Adaptive + stochastic threshold model."""
    return float(
        state.Θ
        + _clip(a * state.E, 0.0, 1.0)
        + state.rng.normal(0.0, _clip(sigma, 0.0, 0.2))
    )

# Realignment ⊙(V, ∆, E, k, context) → V'
def realign_linear(state: "EpistemicState", R_val: float, delta: float) -> float:
    """
    Scalar path (state.V is float). For vectors, the kernel handles a separate bounded step.
    """
    step = state.k * delta * (1.0 + state.E)
    step = _clip(step, -state.step_cap, state.step_cap)
    sign = 1.0 if (R_val - float(state.V)) >= 0.0 else -1.0
    return float(float(state.V) + sign * step)

def realign_tanh(state: "EpistemicState", R_val: float, delta: float) -> float:
    """
    Smoothly saturating realignment; attenuates as delta grows; uses E via tanh.
    """
    eps = state.epsilon if state.epsilon > 0 else 1e-6
    gain = np.tanh(state.k * delta) * (1.0 + np.tanh(state.E / eps))
    gain = _clip(gain, -state.step_cap, state.step_cap)
    sign = 1.0 if (R_val - float(state.V)) >= 0.0 else -1.0
    return float(float(state.V) + sign * gain)

def realign_decay_adaptive(state: "EpistemicState", R_val: float, delta: float) -> float:
    """
    Gain decays with E: k_eff = k / (1+E); stabilizes when memory high.
    """
    k_eff = _clip(state.k / (1.0 + state.E), 0.0, 1.0)
    step = _clip(k_eff * delta, -state.step_cap, state.step_cap)
    sign = 1.0 if (R_val - float(state.V)) >= 0.0 else -1.0
    return float(float(state.V) + sign * step)

# Collapse policies (post-Θ): return (V', E')
# Allow vector-aware V' by permitting Union[float, Vector]
def collapse_reset(
    state: "EpistemicState", R_val: Optional[float] = None
) -> Tuple[Union[float, Vector], float]:
    """Hard reset to zero; E cleared."""
    if isinstance(state.V, np.ndarray):
        return np.zeros_like(state.V, dtype=float), 0.0
    return 0.0, 0.0

def collapse_soft_decay(
    state: "EpistemicState",
    R_val: Optional[float] = None,
    gamma: float = 0.5,
    beta: float = 0.3,
) -> Tuple[Union[float, Vector], float]:
    """Partial decay of projection and memory."""
    if isinstance(state.V, np.ndarray):
        return np.asarray(state.V, dtype=float) * _clip(gamma, 0.0, 1.0), float(state.E) * _clip(beta, 0.0, 1.0)
    return float(state.V) * _clip(gamma, 0.0, 1.0), float(state.E) * _clip(beta, 0.0, 1.0)

def collapse_adopt_R(
    state: "EpistemicState", R_val: Optional[float] = None
) -> Tuple[Union[float, Vector], float]:
    """
    Adopt incoming reality as new projection; memory cleared.
    Vector path: adopt last seen R_vec if available.
    """
    if isinstance(state.V, np.ndarray) and getattr(state, "_last_R_vec", None) is not None:
        return np.asarray(state._last_R_vec, dtype=float), 0.0
    if R_val is None:
        R_val = float(state.V) if not isinstance(state.V, np.ndarray) else 0.0
    return float(R_val), 0.0

def collapse_randomized(
    state: "EpistemicState", R_val: Optional[float] = None, sigma: float = 0.1
) -> Tuple[Union[float, Vector], float]:
    """Collapse to a small random perturbation around 0; memory cleared."""
    if isinstance(state.V, np.ndarray):
        return state.rng.normal(0.0, _clip(sigma, 0.0, 1.0), size=state.V.shape), 0.0
    return float(state.rng.normal(0.0, _clip(sigma, 0.0, 1.0))), 0.0

# Divergence S̄ — vector-aware projection; fallback to directional scalar proxy
def sbar_projection(state: "EpistemicState", R_val: float) -> float:
    """
    If V and last R are vectors: project (R-V) onto V̂ and normalize by ‖V‖.
    Else: signed scalar difference normalized by (1+|V|).
    """
    if isinstance(state.V, np.ndarray) and getattr(state, "_last_R_vec", None) is not None:
        V = np.asarray(state.V, dtype=float)
        R = np.asarray(state._last_R_vec, dtype=float)
        d = R - V
        nV = _l2_norm(V)
        # Project d along V direction, then scale by 1/‖V‖ for a unitless signal
        proj = float(np.dot(d, V / nV)) / nV
        return proj
    denom = 1.0 + abs(float(state.V)) if not isinstance(state.V, np.ndarray) else 1.0
    return float((R_val - (float(state.V) if not isinstance(state.V, np.ndarray) else float(np.linalg.norm(state.V)))) / denom)

# ---------------------------
# Optional: lightweight Perception hook contract
# ---------------------------

class Perception:
    """
    Optional adapter that converts dict inputs (text/image/sensor/etc.)
    into a (optionally normalized) evidence vector (np.ndarray). You can supply a custom instance.

    Set normalize=False if magnitude encodes signal strength (e.g., sensors where norm matters).
    """
    def __init__(
        self,
        text_encoder: Optional[Callable[[str], Vector]] = None,
        image_encoder: Optional[Callable[[Any], Vector]] = None,
        sensor_fusion: Optional[Callable[[Dict[str, float]], Vector]] = None,
        normalize: bool = True,
    ):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.sensor_fusion = sensor_fusion
        self.normalize = bool(normalize)

    def process(self, inputs: Dict[str, Any]) -> Vector:
        vecs: List[Vector] = []
        if "text" in inputs and self.text_encoder:
            try:
                vecs.append(np.asarray(self.text_encoder(inputs["text"]), dtype=float))
            except Exception as e:
                raise ValueError(f"Perception.text_encoder failed: {e}")
        if "image" in inputs and self.image_encoder:
            try:
                vecs.append(np.asarray(self.image_encoder(inputs["image"]), dtype=float))
            except Exception as e:
                raise ValueError(f"Perception.image_encoder failed: {e}")
        if "sensor" in inputs and self.sensor_fusion:
            try:
                vecs.append(np.asarray(self.sensor_fusion(inputs["sensor"]), dtype=float))
            except Exception as e:
                raise ValueError(f"Perception.sensor_fusion failed: {e}")
        if not vecs:
            raise ValueError("Perception: no supported modalities in inputs.")
        V = np.stack(vecs, axis=0).mean(axis=0)
        if self.normalize:
            n = _l2_norm(V)
            V = V / n
        return V.astype(float)

# ---------------------------
# Optional: Meta-policy scaffolding (safe-by-construction)
# ---------------------------

@dataclass
class PolicySpec:
    id: str
    threshold_fn: Callable[["EpistemicState"], float]
    realign_fn: Callable[["EpistemicState", float, float], float]
    collapse_fn: Callable[
        ["EpistemicState", Optional[float]], Tuple[Union[float, Vector], float]
    ]
    params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def fns(self) -> Dict[str, Callable]:
        return {"threshold": self.threshold_fn, "realign": self.realign_fn, "collapse": self.collapse_fn}

class PolicyMemory:
    """Rolling memory of (ctx, policy id, params, reward) with JSON I/O hooks."""
    def __init__(self, cap: int = 4096):
        self.records: List[Dict[str, Any]] = []
        self.cap = int(cap)

    def remember(self, ctx: Dict[str, Any], spec: PolicySpec, reward: float) -> None:
        self.records.append({
            "ctx": {k: _sanitize(v) for k, v in ctx.items()},
            "id": spec.id,
            "params": {k: _sanitize(v) for k, v in spec.params.items()},
            "reward": float(reward),
            "ts": _now_iso(),
        })
        if len(self.records) > self.cap:
            self.records = self.records[-self.cap:]

    def top_ids(self, k: int = 3) -> List[str]:
        if not self.records:
            return []
        ordered = sorted(self.records, key=lambda r: r["reward"], reverse=True)
        seen, out = set(), []
        for r in ordered:
            pid = r["id"]
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
            if len(out) >= k:
                break
        return out

    # Optional persistence (used by EpistemicState.save/load_policy_memory if provided)
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        # sanity: cap
        if len(self.records) > self.cap:
            self.records = self.records[-self.cap:]

# Reward & Safety utilities
def _default_reward(window: List[Dict[str, Any]]) -> float:
    """
    Reward = -(rupture_rate) - 0.2*mean_drift - 0.05*time_to_realign_mean
    Larger is better (closer to 0 up to negative values).
    """
    if not window:
        return -1e9
    import numpy as _np
    drift = _np.array([w.get("∆", 0.0) for w in window], dtype=float)
    rupt = _np.array([1.0 if w.get("ruptured", False) else 0.0 for w in window], dtype=float).mean()
    ttr  = _np.array([w.get("time_to_realign", 1.0) for w in window], dtype=float).mean()
    return float((-1.0 * rupt) + (-0.2 * drift.mean()) + (-0.05 * ttr))

def _safety_flags(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple online safety checks:
      - rupture_storm: rupture rate above 0.5 over window
      - oscillation: alternating rupture pattern (rudimentary)
      - runaway_E: E grows monotonically (if E logged in entries)
    """
    import numpy as _np
    if not window:
        return {"rupture_storm": False, "oscillation": False, "runaway_E": False, "count": 0}

    rupt_seq = _np.array([1 if w.get("ruptured", False) else 0 for w in window], dtype=int)
    storm = rupt_seq.mean() > 0.5

    # Oscillation heuristic: many sign changes between 0/1 across the window
    changes = _np.abs(_np.diff(rupt_seq)).sum()
    osc = changes >= max(4, len(rupt_seq) // 3)

    # Runaway E if present
    Es = _np.array([w.get("E", None) for w in window if "E" in w], dtype=float)
    runaway = False
    if Es.size >= 4:
        diffs = _np.diff(Es)
        runaway = _np.all(diffs > 0.0) and Es[-1] > Es[0] * 1.2  # monotonic + 20% increase

    return {"rupture_storm": bool(storm), "oscillation": bool(osc), "runaway_E": bool(runaway), "count": len(window)}

# Shadow evaluation runner
class ShadowRunner:
    """Clone state → inject candidate policy → replay recent evidence → compute reward & safety flags."""
    def __init__(
        self,
        reward_fn: Callable[[List[Dict[str, Any]]], float] = _default_reward,
        min_window: int = 10,
        max_window: int = 50,
    ):
        self.reward_fn = reward_fn
        self.min_window = int(min_window)
        self.max_window = int(max_window)

    def replay(self, state: "EpistemicState", spec: PolicySpec, evidence_seq: List[Evidence]) -> Tuple[float, Dict[str, Any]]:
        if not evidence_seq:
            return -1e9, {"reason": "empty_window"}
        seq = evidence_seq[-self.max_window:]
        if len(seq) < self.min_window:
            return -1e9, {"reason": "short_window", "n": len(seq)}

        s = copy.deepcopy(state)
        s.inject_policy(**spec.fns())
        base_t = len(s.history)
        for r in seq:
            s.receive(r, source="shadow")
        window = s.history[base_t:]
        if not window:
            return -1e9, {"reason": "no_delta"}

        r = self.reward_fn(window)
        flags = _safety_flags(window)
        return float(r), flags

# Bounded, safety-gated PolicyManager
class PolicyManager:
    """
    ε-greedy policy selector with shadow evaluation and safety gating.
    - epsilon: exploration rate
    - promote_margin: require candidate_reward >= promote_margin * current_reward
    - cooldown_steps: prevent thrashing; skip promotion if last promotion too recent

    Optional bounded evolution (parameter mutation) can be enabled by providing
    a 'param_space' via set_param_space() and using evolve_if_due().
    """
    def __init__(
        self,
        base_specs: List[PolicySpec],
        memory: Optional[PolicyMemory] = None,
        shadow: Optional[ShadowRunner] = None,
        epsilon: float = 0.15,
        promote_margin: float = 1.03,
        cooldown_steps: int = 30,
        rng: Optional[np.random.Generator] = None,
    ):
        if not base_specs:
            raise ValueError("PolicyManager requires at least one PolicySpec.")
        self.specs: Dict[str, PolicySpec] = {s.id: s for s in base_specs}
        self.memory = memory or PolicyMemory()
        self.shadow = shadow or ShadowRunner()
        self.epsilon = float(_clip(epsilon, 0.0, 1.0))
        self.promote_margin = float(max(1.0, promote_margin))
        self.cooldown_steps = int(max(0, cooldown_steps))
        self._last_promotion_step: Optional[int] = None
        self.last_promotion: Optional[Dict[str, Any]] = None

        # RNG for exploration (deterministic if provided)
        self.rng: np.random.Generator = rng or np.random.default_rng()

        # Evolution scaffolding
        self.param_space: Dict[str, Dict[str, Any]] = {}
        self.evolve_every: int = 50
        self.evolve_rate: float = 1.0
        self.evolve_margin: float = 1.02

    @staticmethod
    def _bucket(x: float, bins: List[float]) -> str:
        for i, b in enumerate(bins):
            if x <= b:
                return f"b{i}"
        return f"b{len(bins)}"

    def context_bucket(self, ctx: Dict[str, Any]) -> str:
        """
        Map raw context into a coarse, stable bucket string.
        Expected ctx keys (optional, defaults handled): E, mean_drift_20, rupt_rate_20, domain, step
        """
        E = float(ctx.get("E", 0.0))
        md = float(ctx.get("mean_drift_20", 0.0))
        rr = float(ctx.get("rupt_rate_20", 0.0))
        dom = str(ctx.get("domain", "generic"))
        Eb = self._bucket(E, [0.1, 0.5, 1.0, 2.0, 5.0])
        Db = self._bucket(md, [0.05, 0.1, 0.2, 0.4, 1.0])
        Rb = self._bucket(rr, [0.05, 0.1, 0.2, 0.4, 0.8])
        return f"{dom}|E:{Eb}|D:{Db}|R:{Rb}"

    def _current_spec(self, state: "EpistemicState") -> PolicySpec:
        """Snapshot the state's current policy triple for comparison."""
        th = state._threshold_fn if state._threshold_fn else threshold_static
        rl = state._realign_fn if state._realign_fn else realign_linear
        cp = state._collapse_fn if state._collapse_fn else collapse_reset
        return PolicySpec("__current__", th, rl, cp, params={"k": getattr(state, "k", None), "Θ": getattr(state, "Θ", None)})

    def _top_candidates(self, bucket: str, k: int = 3) -> List[PolicySpec]:
        top_ids = self.memory.top_ids(k=k)
        if top_ids:
            return [self.specs[i] for i in top_ids if i in self.specs]
        return list(self.specs.values())

    @staticmethod
    def _apply_params_to_state(state: "EpistemicState", params: Dict[str, Any]) -> None:
        """Apply recognized params from PolicySpec to EpistemicState."""
        if not params:
            return
        for key in ("k", "Θ", "step_cap", "decay_rate", "epsilon"):
            if key in params and hasattr(state, key):
                try:
                    setattr(state, key, float(params[key]))
                except Exception:
                    pass  # ignore bad types silently; params are advisory

    def maybe_adjust(self, state: "EpistemicState", ctx: Dict[str, Any], recent_evidence: List[Evidence]) -> Optional[str]:
        """
        Evaluate candidates and promote the best safe policy if it beats current by margin.
        Returns promoted policy id or None.
        """
        step = int(ctx.get("t", len(state.history)))
        if self._last_promotion_step is not None and (step - self._last_promotion_step) < self.cooldown_steps:
            return None

        bucket = self.context_bucket(ctx)
        explore = (self.rng.random() < self.epsilon) or (not self.memory.records)

        candidates = list(self.specs.values()) if explore else self._top_candidates(bucket, k=min(3, len(self.specs)))

        # Evaluate candidates
        best_spec: Optional[PolicySpec] = None
        best_r: float = -1e12
        best_flags: Dict[str, Any] = {}

        for spec in candidates:
            r, flags = self.shadow.replay(state, spec, recent_evidence)
            if r > best_r:
                best_r, best_spec, best_flags = r, spec, flags

        # Compare with current behavior
        current_spec = self._current_spec(state)
        current_r, current_flags = self.shadow.replay(state, current_spec, recent_evidence)

        # Safety gating
        unsafe = bool(best_flags.get("rupture_storm") or best_flags.get("oscillation") or best_flags.get("runaway_E"))

        if best_spec and (best_r > self.promote_margin * current_r) and not unsafe:
            # Promote
            state.inject_policy(**best_spec.fns())
            # Apply advisory params (k, Θ, etc.)
            self._apply_params_to_state(state, best_spec.params or {})
            self.memory.remember(ctx, best_spec, best_r)
            self.last_promotion = {
                "id": best_spec.id,
                "bucket": bucket,
                "reward": float(best_r),
                "baseline_reward": float(current_r),
                "flags": best_flags,
                "step": step,
            }
            self._last_promotion_step = step
            if hasattr(state, "_log_event"):
                state._log_event("policy_promoted", self.last_promotion)
            return best_spec.id

        # Remember baseline too (for analytics)
        self.memory.remember(ctx, current_spec, current_r)
        return None

    # ---------- Evolution scaffolding (bounded, no arbitrary codegen) ----------

    class ParamRange:
        """Simple numeric range for mutations: uniform(low, high) with optional gaussian jitter."""
        def __init__(self, low: float, high: float, sigma: float = 0.05):
            self.low = float(low)
            self.high = float(high)
            self.sigma = float(max(0.0, sigma))

        def sample(self, rng: np.random.Generator, rate: float = 1.0) -> float:
            base = float(rng.uniform(self.low, self.high))
            if self.sigma > 0.0:
                base += float(rng.normal(0.0, self.sigma * rate))
            return float(min(max(base, self.low), self.high))

    def set_param_space(self, space: Dict[str, Dict[str, Any]]) -> None:
        """
        space: {policy_id: {param_name: ParamRange or (low, high[, sigma])}}
        """
        norm: Dict[str, Dict[str, Any]] = {}
        for pid, params in space.items():
            norm[pid] = {}
            for k, v in params.items():
                if isinstance(v, tuple) and len(v) in (2, 3):
                    low, high, *rest = v
                    sigma = rest[0] if rest else 0.05
                    norm[pid][k] = PolicyManager.ParamRange(float(low), float(high), float(sigma))
                elif isinstance(v, PolicyManager.ParamRange):
                    norm[pid][k] = v
                else:
                    norm[pid][k] = v  # pass-through (fixed)
        self.param_space = norm

    def _mutate_spec(self, state: "EpistemicState", spec: PolicySpec) -> PolicySpec:
        """
        Produce a mutated copy of a spec within the allowable param ranges.
        Only touches keys present in param_space[spec.id].
        """
        if spec.id not in self.param_space:
            return spec
        new_params = dict(spec.params)
        ranges = self.param_space[spec.id]
        for k, r in ranges.items():
            if isinstance(r, PolicyManager.ParamRange):
                new_params[k] = r.sample(state.rng, self.evolve_rate)
            else:
                new_params[k] = r
        return PolicySpec(
            id=f"{spec.id}*",
            threshold_fn=spec.threshold_fn,
            realign_fn=spec.realign_fn,
            collapse_fn=spec.collapse_fn,
            params=new_params,
            notes=(spec.notes or "") + " [mutated]"
        )

    def evolve_if_due(self, state: "EpistemicState", ctx: Dict[str, Any], recent: List[Evidence]) -> Optional[str]:
        """
        Attempt bounded mutation/promotion every `evolve_every` steps.
        This is conservative: it never changes callables, only params metadata.
        """
        if not self.param_space:
            return None
        step = int(ctx.get("t", len(state.history)))
        if step % max(5, int(self.evolve_every)) != 0:
            return None

        # Choose a base (top performer in context or random explore)
        top_ids = self.memory.top_ids(k=min(3, len(self.specs)))
        base = self.specs.get(top_ids[0], None) if top_ids else None
        if base is None:
            # fallback: any
            base = next(iter(self.specs.values()))

        cand = self._mutate_spec(state, base)
        # Evaluate candidate (without altering functions, only params used for context/record)
        r_cand, flags_cand = self.shadow.replay(state, cand, recent)
        r_curr, flags_curr = self.shadow.replay(state, self._current_spec(state), recent)

        unsafe = bool(flags_cand.get("rupture_storm") or flags_cand.get("oscillation") or flags_cand.get("runaway_E"))
        if not unsafe and r_cand > self.evolve_margin * r_curr:
            # Record and (optionally) apply advisory params
            self.memory.remember(ctx, cand, r_cand)
            PolicyManager._apply_params_to_state(state, cand.params or {})
            self.last_promotion = {"id": cand.id, "reward": r_cand, "mutated_from": base.id, "step": step}
            if hasattr(state, "_log_event"):
                state._log_event("policy_mutation_promoted", self.last_promotion)
            return cand.id
        return None

# ---------------------------
# EpistemicState core
# ---------------------------

class EpistemicState:
    """
    CT-aligned epistemic kernel with:
      - V (projection), E (misalignment memory), Θ (base threshold), k (⊙ gain)
      - Per-step ∆ and S̄
      - Optional Perception and PolicyManager hooks
      - Explainability logs and exports
    """

    def __init__(
        self,
        V0: Union[float, Vector] = 0.0,
        E0: float = 0.0,
        threshold: float = 0.35,
        realign_strength: float = 0.3,
        decay_rate: float = 0.9,
        identity: Optional[Dict[str, Any]] = None,
        log_history: bool = True,
        rng_seed: Optional[int] = None,
        step_cap: float = 1.0,
        epsilon: float = 1e-3,
        perception: Optional[Perception] = None,
        policy_manager: Optional[PolicyManager] = None,
        divergence_fn: Optional[Callable[["EpistemicState", float], float]] = None,
        max_history: Optional[int] = None,
    ):
        # Core state
        self.V: Union[float, Vector] = float(V0) if isinstance(V0, (int, float, np.number)) else np.asarray(V0, dtype=float)
        self.E: float = float(E0)
        self.Θ: float = float(threshold)
        self.k: float = float(realign_strength)
        self.decay_rate: float = float(_clip(decay_rate, 0.0, 1.0))
        self.step_cap: float = float(max(1e-6, step_cap))
        self.epsilon: float = float(max(1e-9, epsilon))  # small constant used in tanh-based realign

        # Local RNG (no global pollution)
        self.rng: np.random.Generator = np.random.default_rng(int(rng_seed) if rng_seed is not None else None)

        # Policy hooks
        self._threshold_fn: Optional[Callable[["EpistemicState"], float]] = None
        self._realign_fn: Optional[Callable[["EpistemicState", float, float], float]] = None
        self._collapse_fn: Optional[Callable[["EpistemicState", Optional[float]], Tuple[Union[float, Vector], float]]] = None
        self._context_fn: Optional[Callable[..., Any]] = None
        self._divergence_fn: Callable[["EpistemicState", float], float] = divergence_fn or sbar_projection

        # Satellites
        self.perception: Optional[Perception] = perception
        self.policy_manager: Optional[PolicyManager] = policy_manager

        # Runtime/logging (initialized before any _log_event calls)
        self.history: List[Dict[str, Any]] = []
        self.meta_ruptures: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self._rupture_count: int = 0
        self._last_symbol: str = "∅"  # EMPTY
        self._triggers: Dict[str, Callable[[], Any]] = {}
        self._time: int = 0
        self._id: str = str(uuid.uuid4())[:8]
        self.identity: Dict[str, Any] = identity or {}
        self._log = bool(log_history)
        self._max_history: Optional[int] = int(max_history) if max_history is not None else None

        # Derived metrics (rolling)
        self._rolling_ruptures: List[int] = []
        self._rolling_drift: List[float] = []

        # Divergence helper
        self._last_R_vec: Optional[Vector] = None

        # Real time-to-realign tracker
        self._since_last_rupture: Optional[int] = None

        # Note seed after logs exist
        if rng_seed is not None:
            self._log_event("rng_seeded", {"seed": int(rng_seed)})

    # -----------------------
    # High-level API
    # -----------------------

    def receive(self, R: Evidence, source: str = "default") -> None:
        """
        Feed reception R (float/vector OR dict for multi-modal if Perception is attached).
        Implements the CT loop per step and logs the outcome.

        Flow:
          1) Resolve reality → R_val (scalar proxy even if V is scalar; vectors handled separately)
          2) Compute ∆ (scalar for scalars; for vectors, use norm of (R_vec - V_vec))
          3) Compute S̄ (directional; for vectors, cosine-ish signed component if possible)
          4) Evaluate Θ policy; if ∆ > Θ' => Θ (rupture) → collapse policy
             else ⊙ (realign) policy
          5) Update E (misalignment memory)
          6) Meta-policy manager (if present) may evaluate and promote/evolve periodically
        """
        # 1) Resolve R
        R_is_dict = isinstance(R, dict)
        R_vec: Optional[Vector] = None

        if R_is_dict:
            if not self.perception:
                raise ValueError("Dict evidence requires a Perception adapter.")
            R_vec = self.perception.process(R)  # normalized or not based on config
        elif isinstance(R, (list, tuple, np.ndarray)):
            R_vec = np.asarray(R, dtype=float)
        elif isinstance(R, (int, float, np.number)):
            R_val_scalar = float(R)
        else:
            raise ValueError("Unsupported evidence type for R.")

        V_is_scalar = not isinstance(self.V, np.ndarray)

        # Align types for computation
        if R_vec is not None and V_is_scalar:
            # Promote V to vector of same shape (soft-start alternative could project magnitude)
            self.V = np.zeros_like(R_vec, dtype=float)
            V_is_scalar = False

        # 2) Compute ∆ and scalar R_val for policy signatures
        if not V_is_scalar:
            assert R_vec is not None, "Vector V requires vector R."
            V_vec = np.asarray(self.V, dtype=float)
            delta_vec = R_vec - V_vec
            delta_mag = float(np.linalg.norm(delta_vec))
            R_val = float(np.linalg.norm(R_vec))  # scalar proxy for policy signatures
            self._last_R_vec = np.asarray(R_vec, dtype=float)
        else:
            R_val = float(R_val_scalar)  # type: ignore
            delta_mag = abs(R_val - float(self.V))
            self._last_R_vec = None  # not used for scalar path

        # 3) Divergence S̄ (projected directional deviation)
        try:
            S_bar = float(self._divergence_fn(self, R_val))
        except Exception as e:
            warnings.warn(f"Divergence function failed ({e}); using directional proxy.")
            S_bar = float(
                (R_val - (float(self.V) if V_is_scalar else float(np.linalg.norm(self.V))))
            )  # fallback

        # 4) Evaluate threshold & rupture
        threshold_val = float(self._threshold_fn(self) if self._threshold_fn else self.Θ)
        ruptured = bool(delta_mag > threshold_val)
        self._last_symbol = "⚠" if ruptured else "⊙"

        if ruptured:
            # Collapse policy
            if callable(self._collapse_fn):
                try:
                    # Adopt-R fast path for vectors if available
                    if self._collapse_fn is collapse_adopt_R and self._last_R_vec is not None:
                        Vp, Ep = np.asarray(self._last_R_vec, dtype=float), 0.0
                    else:
                        Vp, Ep = self._collapse_fn(self, R_val)
                except Exception as e:
                    raise RuntimeError(f"Collapse function failed: {e}")
            else:
                Vp, Ep = collapse_reset(self, R_val)

            # Update V, E — scalar and vector paths
            self.E = float(Ep)
            if isinstance(self.V, np.ndarray):
                if isinstance(Vp, np.ndarray):
                    self.V = np.asarray(Vp, dtype=float)
                else:
                    basis = np.asarray(self.V, dtype=float)
                    n = _l2_norm(basis)
                    if n == 0.0 and R_vec is not None:
                        basis = R_vec
                        n = _l2_norm(basis)
                    self.V = (basis / n) * float(Vp)
            else:
                self.V = float(Vp)

            self._rupture_count += 1
            self.meta_ruptures.append({
                "time": int(self._time),
                "rupture_pressure": float(delta_mag - threshold_val),
                "source": source,
                "S̄": float(S_bar),
            })
            self._trigger("on_rupture")
            # Track real re-stabilization time
            self._since_last_rupture = 0
            time_to_realign = 0.0

        else:
            # Realign policy
            if V_is_scalar:
                if callable(self._realign_fn):
                    try:
                        self.V = float(self._realign_fn(self, R_val, delta_mag))
                    except Exception as e:
                        raise RuntimeError(f"Realign function failed: {e}")
                else:
                    self.V = float(realign_linear(self, R_val, delta_mag))
            else:
                # Vector realignment: move V toward R with bounded step
                V_vec = np.asarray(self.V, dtype=float)
                assert R_vec is not None
                step = self.k * (R_vec - V_vec) * (1.0 + self.E)  # type: ignore
                # Cap step by norm
                norm_step = float(np.linalg.norm(step))
                if norm_step > self.step_cap:
                    step = step * (self.step_cap / norm_step)
                self.V = (V_vec + step).astype(float)

            # Misalignment memory update
            self.E += 0.1 * delta_mag
            self.E *= self.decay_rate

            # Real time-to-realign: count steps since last rupture until stable
            if self._since_last_rupture is not None:
                self._since_last_rupture += 1
                if delta_mag <= 0.5 * threshold_val:
                    time_to_realign = float(self._since_last_rupture)
                    self._since_last_rupture = None
                else:
                    time_to_realign = 1.0
            else:
                time_to_realign = 1.0

        # 5) Log step
        if self._log:
            log_entry = {
                "t": int(self._time),
                "V": _sanitize(self.V),
                "R": _sanitize(R_vec if R_vec is not None else R_val),
                "∆": float(delta_mag),
                "S̄": float(S_bar),
                "Θ": float(threshold_val),
                "E": float(self.E),
                "ruptured": bool(ruptured),
                "symbol": self._last_symbol,
                "source": source,
                "time_to_realign": float(time_to_realign),
            }
            self.history.append(log_entry)
            if self._max_history is not None and len(self.history) > self._max_history:
                self.history.pop(0)
            self._rolling_ruptures.append(1 if ruptured else 0)
            self._rolling_drift.append(float(delta_mag))
            if len(self._rolling_ruptures) > 256:
                self._rolling_ruptures.pop(0)
            if len(self._rolling_drift) > 256:
                self._rolling_drift.pop(0)

        # 6) Advance time and tick meta-policy manager / evolution (if present)
        self._time += 1
        if self.policy_manager and len(self.history) >= 12 and (self._time % 10 == 0):
            recent_evidence = [h["R"] for h in self.history[-20:]]
            ctx = {
                "t": int(self._time),
                "E": float(self.E),
                "rupt_rate_20": float(np.mean(self._rolling_ruptures[-20:])) if self._rolling_ruptures else 0.0,
                "mean_drift_20": float(np.mean(self._rolling_drift[-20:])) if self._rolling_drift else 0.0,
            }
            try:
                self.policy_manager.maybe_adjust(self, ctx, recent_evidence)
            except Exception as e:
                self._log_event("policy_manager_error", {"error": str(e)})

            # Optional bounded evolution pass
            if hasattr(self.policy_manager, "evolve_if_due"):
                try:
                    self.policy_manager.evolve_if_due(self, ctx, recent_evidence)  # type: ignore
                except Exception as e:
                    self._log_event("policy_evolve_error", {"error": str(e)})

    # -----------------------
    # Auto-evolution wiring helper
    # -----------------------

    def enable_auto_evolution(
        self,
        param_space: Dict[str, Dict[str, Any]],
        every: int = 30,
        rate: float = 1.0,
        margin: float = 1.02,
    ) -> None:
        """
        Wire the state's PolicyManager for bounded runtime evolution.

        param_space: {"policy_id": {"param": ParamRange or (low, high, sigma)}}
        every:       attempt evolution every N steps
        rate:        mutation intensity multiplier
        margin:      candidate must beat baseline by this factor to be promoted
        """
        if not self.policy_manager:
            raise ValueError("Auto-evolution requires a PolicyManager on this state.")
        if hasattr(self.policy_manager, "set_param_space"):
            # normalize tuples into ParamRange if needed
            try:
                # Use manager's ParamRange
                normalized: Dict[str, Dict[str, Any]] = {}
                for pid, params in param_space.items():
                    normalized[pid] = {}
                    for k, v in params.items():
                        if isinstance(v, tuple) and len(v) in (2, 3):
                            low, high, *rest = v
                            sigma = rest[0] if rest else 0.05
                            normalized[pid][k] = self.policy_manager.ParamRange(float(low), float(high), float(sigma))  # type: ignore
                        else:
                            normalized[pid][k] = v
                self.policy_manager.set_param_space(normalized)  # type: ignore
            except Exception:
                # fall back: pass through
                self.policy_manager.set_param_space(param_space)  # type: ignore

        # cadence & thresholds (only if manager exposes these knobs)
        for attr, val in (
            ("evolve_every", int(max(5, every))),
            ("evolve_rate", float(rate)),
            ("evolve_margin", float(max(1.0, margin))),
        ):
            if hasattr(self.policy_manager, attr):
                setattr(self.policy_manager, attr, val)

        self._log_event("auto_evolution_enabled", {
            "every": int(every), "rate": float(rate), "margin": float(margin)
        })

    # -----------------------
    # Persistence helpers for policy memory
    # -----------------------

    def save_policy_memory(self, path: str) -> None:
        if self.policy_manager and hasattr(self.policy_manager, "memory") and hasattr(self.policy_manager.memory, "save"):
            try:
                self.policy_manager.memory.save(path)  # type: ignore
                self._log_event("policy_memory_saved", {"path": path})
            except Exception as e:
                self._log_event("policy_memory_save_error", {"error": str(e)})

    def load_policy_memory(self, path: str) -> None:
        if self.policy_manager and hasattr(self.policy_manager, "memory") and hasattr(self.policy_manager.memory, "load"):
            try:
                self.policy_manager.memory.load(path)  # type: ignore
                self._log_event("policy_memory_loaded", {"path": path})
            except Exception as e:
                self._log_event("policy_memory_load_error", {"error": str(e)})

    # -----------------------
    # Queries / tools
    # -----------------------

    def rupture_risk(self) -> Optional[float]:
        if not self.history:
            return None
        last = self.history[-1]
        return float(last["∆"] - last["Θ"])

    def should_intervene(self, margin: float = 0.0) -> bool:
        risk = self.rupture_risk()
        return bool(risk is not None and risk > float(margin))

    def intervene_if_ruptured(self, fallback_fn: Callable[[], Any], margin: float = 0.0) -> Optional[Any]:
        if self.should_intervene(margin):
            return fallback_fn()
        return None

    def reset(self) -> None:
        self.V = 0.0 if not isinstance(self.V, np.ndarray) else np.zeros_like(self.V)
        self.E = 0.0
        self._rupture_count = 0
        self._time = 0
        self._last_symbol = "∅"
        self.history.clear()
        self.meta_ruptures.clear()
        self.event_log.clear()
        self._rolling_ruptures.clear()
        self._rolling_drift.clear()
        self._last_R_vec = None
        self._since_last_rupture = None
        self._log_event("reset", {})

    def realign(self, R: Evidence) -> None:
        R_val = self._resolve_reality(R)
        # Manual adopt-R realign (safe)
        if isinstance(self.V, np.ndarray):
            if isinstance(R, (list, tuple, np.ndarray)):
                self.V = np.asarray(R, dtype=float)
            else:
                # keep direction; set magnitude to R_val
                v = np.asarray(self.V, dtype=float)
                n = _l2_norm(v)
                self.V = (v / n) * R_val
        else:
            self.V = float(R_val)
        self.E *= 0.5
        self._last_symbol = "⊙"
        if self._log:
            self.history.append({
                "t": int(self._time),
                "V": _sanitize(self.V),
                "R": _sanitize(R),
                "∆": 0.0,
                "S̄": 0.0,
                "Θ": float(self.Θ),
                "E": float(self.E),
                "ruptured": False,
                "symbol": "⊙",
                "source": "manual_realign",
            })
            if self._max_history is not None and len(self.history) > self._max_history:
                self.history.pop(0)
        self._log_event("manual_realign", {"aligned_to": _sanitize(R)})
        self._time += 1

    def inject_policy(
        self,
        threshold: Optional[Callable[["EpistemicState"], float]] = None,
        realign: Optional[Callable[["EpistemicState", float, float], float]] = None,
        collapse: Optional[Callable[["EpistemicState", Optional[float]], Tuple[Union[float, Vector], float]]] = None,
    ) -> None:
        if threshold and not callable(threshold):
            raise TypeError("Threshold policy must be callable.")
        if realign and not callable(realign):
            raise TypeError("Realign policy must be callable.")
        if collapse and not callable(collapse):
            raise TypeError("Collapse policy must be callable.")
        self._threshold_fn = threshold
        self._realign_fn = realign
        self._collapse_fn = collapse
        self._log_event("policy_injected", {
            "threshold": bool(threshold),
            "realign": bool(realign),
            "collapse": bool(collapse),
        })

    def bind_context(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError("Context function must be callable.")
        self._context_fn = fn
        self._log_event("context_bound", {"bound": True})

    def run_context(self, *args, **kwargs) -> Any:
        if self._context_fn is None:
            raise ValueError("No context function bound.")
        result = self._context_fn(*args, **kwargs)
        self._log_event("context_executed", {"result": _sanitize(result)})
        return result

    def register_trigger(self, event: str, fn: Callable[[], Any]) -> None:
        if not isinstance(event, str):
            raise TypeError("Event must be a string.")
        if not callable(fn):
            raise ValueError("Trigger must be callable.")
        self._triggers[event] = fn
        self._log_event("trigger_registered", {"event": event})

    def _trigger(self, event: str) -> Optional[Any]:
        if event in self._triggers:
            try:
                result = self._triggers[event]()
                self._log_event("trigger_invoked", {"event": event})
                return result
            except Exception as e:
                self._log_event("trigger_error", {"event": event, "error": str(e)})
                raise
        return None

    def symbol(self) -> str:
        return self._last_symbol

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "t": int(self._time),
            "V": _sanitize(self.V),
            "E": float(self.E),
            "Θ": float(self.Θ),
            "ruptures": int(self._rupture_count),
            "last_symbol": self._last_symbol,
            "identity": self.identity,
        }

    def last(self) -> Optional[Dict[str, Any]]:
        return self.history[-1] if self.history else None

    def log(self) -> List[Dict[str, Any]]:
        return self.history

    def rupture_log(self) -> List[Dict[str, Any]]:
        return self.meta_ruptures

    def drift_stats(self, window: int = 10) -> Dict[str, float]:
        deltas = [step["∆"] for step in self.history[-int(window):] if "∆" in step]
        if not deltas:
            return {}
        arr = np.array(deltas, dtype=float)
        return {
            "mean_drift": float(arr.mean()),
            "std_drift": float(arr.std()),
            "max_drift": float(arr.max()),
            "min_drift": float(arr.min()),
        }

    def explain_last(self) -> str:
        last = self.last()
        if not last:
            return "No steps yet."
        if last["ruptured"]:
            return f"Rupture (Θ) triggered: drift {last['∆']:.3f} > threshold {last['Θ']:.3f}; S̄={last.get('S̄', 0):.3f}"
        else:
            return f"Realigned (⊙): drift {last['∆']:.3f} within threshold {last['Θ']:.3f}; S̄={last.get('S̄', 0):.3f}"

    # -----------------------
    # I/O
    # -----------------------

    def export_json(self, path: str) -> None:
        try:
            safe_history = [{k: _sanitize(v) for k, v in entry.items()} for entry in self.history]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(safe_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to export JSON log to {path}: {e}")

    def export_csv(self, path: str) -> None:
        if not self.history:
            return
        try:
            keys = sorted(set().union(*(entry.keys() for entry in self.history)))
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.history:
                    safe_row = {k: _sanitize(row.get(k, "")) for k in keys}
                    writer.writerow(safe_row)
        except Exception as e:
            raise RuntimeError(f"Failed to export CSV log to {path}: {e}")

    def event_log_summary(self) -> List[Dict[str, Any]]:
        return self.event_log

    # -----------------------
    # Internal helpers
    # -----------------------

    def _log_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        sanitized = {}
        if details:
            for k, v in details.items():
                sanitized[k] = _sanitize(v)
        self.event_log.append({
            "event": event_type,
            "time": int(self._time),
            "timestamp": _now_iso(),
            "details": sanitized,
        })

    def _resolve_reality(self, R: Evidence) -> float:
        # For manual realign path and policy signatures needing scalar proxy
        if isinstance(R, (int, float, np.number)):
            return float(R)
        elif isinstance(R, (list, tuple)):
            arr = np.asarray(R, dtype=float)
            return float(np.linalg.norm(arr))
        elif isinstance(R, np.ndarray):
            return float(np.linalg.norm(np.asarray(R, dtype=float)))
        elif isinstance(R, dict):
            if not self.perception:
                raise ValueError("Dict evidence requires a Perception adapter.")
            vec = self.perception.process(R)
            return float(np.linalg.norm(vec))
        else:
            raise ValueError("Reality must be float, list, ndarray, or dict (with Perception).")

# ---------------------------
# Preset safe specs (for immediate use with PolicyManager)
# ---------------------------

SAFE_SPECS: List[PolicySpec] = [
    PolicySpec(
        "conservative",
        threshold_fn=threshold_static,
        realign_fn=realign_linear,
        collapse_fn=collapse_soft_decay,
        params={"k": 0.2, "note": "slow linear realign; soft decay on Θ"},
    ),
    PolicySpec(
        "cautious",
        threshold_fn=threshold_adaptive,
        realign_fn=realign_tanh,
        collapse_fn=collapse_soft_decay,
        params={"k": 0.15, "a": 0.05, "note": "adaptive Θ and bounded ⊙"},
    ),
    PolicySpec(
        "adoptive",
        threshold_fn=threshold_combined,
        realign_fn=realign_decay_adaptive,
        collapse_fn=collapse_adopt_R,
        params={"k": 0.25, "a": 0.05, "sigma": 0.01, "note": "adopt-R under Θ"},
    ),
]

# ---------------------------
# Usage notes (informational, not executed)
# ---------------------------
"""
Quickstart
----------
from cognize.epistemic import EpistemicState, PolicyManager, SAFE_SPECS

state = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.3, rng_seed=42)
pm = PolicyManager(SAFE_SPECS, cooldown_steps=30, rng=state.rng)  # share RNG for full determinism
state.policy_manager = pm

# Stream scalar evidence
for r in [0.1, 0.2, 0.15, 0.9, 1.1, 0.95, 0.5, 0.4]:
    state.receive(r, source="demo")

print(state.explain_last())
print(state.drift_stats(5))

# Multimodal example with Perception
enc_text = lambda t: np.ones(8) * len(t)  # toy embedding
enc_img  = lambda i: np.ones(8) * 0.5     # toy image vec
perception = Perception(text_encoder=enc_text, image_encoder=enc_img, normalize=True)

state2 = EpistemicState(V0=np.zeros(8), perception=perception, rng_seed=123)
state2.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
state2.receive({"text": "hello", "image": object()}, source="demo-multi")

# Enable bounded evolution (optional)
pm.set_param_space({
    "cautious": {"k": (0.05, 0.35, 0.02), "a": (0.01, 0.1, 0.01)}
})
state.enable_auto_evolution(param_space={"cautious": {"k": (0.05, 0.35)}}, every=50, rate=1.0, margin=1.02)
"""
