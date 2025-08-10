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

4) Determinism when needed (seeded RNG), and rich telemetry for explainability.

This file intentionally hosts the kernel + thin integration hooks so downstream modules
(perception/network/meta/memory/safety/goals) can attach without bloating imports.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import uuid
import math
import json
import csv
import copy
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

def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x, dtype=float))

def _l2_norm(v: Vector) -> float:
    n = float(np.linalg.norm(v))
    return n if n != 0.0 else 1.0

def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))

def _sanitize(v: Any) -> Any:
    if isinstance(v, (np.integer, np.int64)): return int(v)
    if isinstance(v, (np.floating, np.float64)): return float(v)
    if isinstance(v, (np.ndarray, list)): return np.asarray(v).tolist()
    if isinstance(v, (str, bool, type(None))): return v
    try: return float(v)
    except: return str(v)

# ---------------------------
# Default policy primitives (safe templates)
# ---------------------------

# Threshold policies Θ(E, t, context)
def threshold_static(state: "EpistemicState") -> float:
    return state.Θ

def threshold_adaptive(state: "EpistemicState", a: float = 0.05) -> float:
    # Θ_t = base + a * E ; bounded for stability
    return float(state.Θ + _clip(a * state.E, 0.0, 1.0))

def threshold_stochastic(state: "EpistemicState", base_sigma: float = 0.01) -> float:
    sigma = _clip(base_sigma, 0.0, 0.2)
    return float(state.Θ + np.random.normal(0.0, sigma))

def threshold_combined(state: "EpistemicState", a: float = 0.05, sigma: float = 0.01) -> float:
    return float(state.Θ + _clip(a * state.E, 0.0, 1.0) + np.random.normal(0.0, _clip(sigma, 0.0, 0.2)))

# Realignment ⊙(V, ∆, E, k, context) → V'
def realign_linear(state: "EpistemicState", R_val: float, delta: float) -> float:
    # Scalar V path (used when state.V is float). When V is vector, specialized handler used.
    step = state.k * delta * (1.0 + state.E)
    cap = state.step_cap
    step = _clip(step, -cap, cap)
    # Direction toward R:
    sign = 1.0 if (R_val - state.V) >= 0.0 else -1.0
    return float(state.V + sign * step)

def realign_tanh(state: "EpistemicState", R_val: float, delta: float) -> float:
    gain = np.tanh(state.k * delta) * (1.0 + np.tanh(state.E / (state.epsilon if state.epsilon>0 else 1e-6)))
    cap = state.step_cap
    gain = _clip(gain, -cap, cap)
    sign = 1.0 if (R_val - state.V) >= 0.0 else -1.0
    return float(state.V + sign * gain)

def realign_decay_adaptive(state: "EpistemicState", R_val: float, delta: float) -> float:
    k_eff = _clip(state.k / (1.0 + state.E), 0.0, 1.0)
    step = k_eff * delta
    step = _clip(step, -state.step_cap, state.step_cap)
    sign = 1.0 if (R_val - state.V) >= 0.0 else -1.0
    return float(state.V + sign * step)

# Collapse policies (post-Θ): return (V', E')
def collapse_reset(state: "EpistemicState", R_val: Optional[float] = None) -> Tuple[float, float]:
    return (0.0 if isinstance(state.V, (int, float)) else 0.0, 0.0)

def collapse_soft_decay(state: "EpistemicState", R_val: Optional[float] = None,
                        gamma: float = 0.5, beta: float = 0.3) -> Tuple[float, float]:
    Vp = float(state.V) * _clip(gamma, 0.0, 1.0)
    Ep = float(state.E) * _clip(beta, 0.0, 1.0)
    return (Vp, Ep)

def collapse_adopt_R(state: "EpistemicState", R_val: Optional[float] = None) -> Tuple[float, float]:
    if R_val is None:
        R_val = float(state.V)
    return (float(R_val), 0.0)

def collapse_randomized(state: "EpistemicState", R_val: Optional[float] = None, sigma: float = 0.1) -> Tuple[float, float]:
    return (float(np.random.normal(0.0, _clip(sigma, 0.0, 1.0))), 0.0)

# Divergence S̄ — scalar proxy using directionality of drift (works even without sequences)
def sbar_directional(state: "EpistemicState", R_val: float) -> float:
    # Signed difference normalized by (1+|V|)
    denom = 1.0 + abs(float(state.V))
    return float((R_val - float(state.V)) / denom)

# ---------------------------
# Optional: lightweight Perception hook contract
# ---------------------------

class Perception:
    """
    Optional adapter that converts dict inputs (text/image/sensor/etc.)
    into a normalized evidence vector (np.ndarray). You can supply a custom instance.
    """
    def __init__(self,
                 text_encoder: Optional[Callable[[str], Vector]] = None,
                 image_encoder: Optional[Callable[[Any], Vector]] = None,
                 sensor_fusion: Optional[Callable[[Dict[str, float]], Vector]] = None):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.sensor_fusion = sensor_fusion

    def process(self, inputs: Dict[str, Any]) -> Vector:
        vecs: List[Vector] = []
        if "text" in inputs and self.text_encoder:
            vecs.append(np.asarray(self.text_encoder(inputs["text"]), dtype=float))
        if "image" in inputs and self.image_encoder:
            vecs.append(np.asarray(self.image_encoder(inputs["image"]), dtype=float))
        if "sensor" in inputs and self.sensor_fusion:
            vecs.append(np.asarray(self.sensor_fusion(inputs["sensor"]), dtype=float))
        if not vecs:
            raise ValueError("Perception: no supported modalities in inputs.")
        V = np.stack(vecs, axis=0).mean(axis=0)
        n = _l2_norm(V)
        return (V / n).astype(float)

# ---------------------------
# Optional: Meta-policy manager
# ---------------------------

@dataclass
class PolicySpec:
    id: str
    threshold_fn: Callable[["EpistemicState"], float]
    realign_fn: Callable[["EpistemicState", float, float], float]
    collapse_fn: Callable[["EpistemicState", Optional[float]], Tuple[float, float]]
    params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def fns(self) -> Dict[str, Callable]:
        return {"threshold": self.threshold_fn, "realign": self.realign_fn, "collapse": self.collapse_fn}

class PolicyMemory:
    def __init__(self, cap: int = 4096):
        self.records: List[Dict[str, Any]] = []
        self.cap = int(cap)

    def remember(self, ctx: Dict[str, Any], spec: PolicySpec, reward: float):
        self.records.append({"ctx": {k: _sanitize(v) for k, v in ctx.items()},
                             "id": spec.id, "params": {k: _sanitize(v) for k, v in spec.params.items()},
                             "reward": float(reward), "ts": _now_iso()})
        if len(self.records) > self.cap:
            self.records = self.records[-self.cap:]

    def top_ids(self, k: int = 3) -> List[str]:
        if not self.records: return []
        # Global top by reward; later you can bucket by context
        sorted_recs = sorted(self.records, key=lambda r: r["reward"], reverse=True)
        seen, out = set(), []
        for r in sorted_recs:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r["id"])
            if len(out) >= k: break
        return out

class ShadowRunner:
    """Replays a recent evidence window against a cloned state with a candidate policy to score reward."""
    def replay(self, state: "EpistemicState", spec: PolicySpec, evidence_seq: List[Evidence]) -> float:
        s = copy.deepcopy(state)
        s.inject_policy(**spec.fns())
        base_t = len(s.history)
        for r in evidence_seq:
            s.receive(r, source="shadow")
        window = s.history[base_t:]
        if not window: return -1e9
        drift = np.array([w["∆"] for w in window], dtype=float)
        rupt = np.array([1.0 if w["ruptured"] else 0.0 for w in window], dtype=float).mean()
        time_to_realign = float(np.mean([w.get("time_to_realign", 1.0) for w in window])) if window else 1.0
        # Reward: fewer ruptures, lower drift, quicker stabilization
        reward = (-1.0 * rupt) + (-0.2 * drift.mean()) + (-0.05 * time_to_realign)
        return float(reward)

class PolicyManager:
    """
    ε-greedy policy selector with shadow evaluation. Template-bound (no arbitrary code gen).
    """
    def __init__(self,
                 base_specs: List[PolicySpec],
                 memory: Optional[PolicyMemory] = None,
                 shadow: Optional[ShadowRunner] = None,
                 epsilon: float = 0.15,
                 promote_margin: float = 1.03):
        self.specs = {s.id: s for s in base_specs}
        self.memory = memory or PolicyMemory()
        self.shadow = shadow or ShadowRunner()
        self.epsilon = float(_clip(epsilon, 0.0, 1.0))
        self.promote_margin = float(max(1.0, promote_margin))
        self.last_promotion: Optional[Dict[str, Any]] = None

    def _current_spec(self, state: "EpistemicState") -> PolicySpec:
        # Snapshot current policies
        return PolicySpec(
            id="__current__",
            threshold_fn=state._threshold_fn if state._threshold_fn else threshold_static,
            realign_fn=state._realign_fn if state._realign_fn else realign_linear,
            collapse_fn=state._collapse_fn if state._collapse_fn else collapse_reset,
            params={"k": state.k, "Θ": state.Θ}
        )

    def maybe_adjust(self, state: "EpistemicState", ctx: Dict[str, Any], recent: List[Evidence]) -> Optional[str]:
        if not self.specs:
            return None
        explore = (np.random.random() < self.epsilon)
        candidates: List[PolicySpec] = []
        if explore or not self.memory.records:
            candidates = list(self.specs.values())
        else:
            top_ids = self.memory.top_ids(k=min(3, len(self.specs)))
            candidates = [self.specs[i] for i in top_ids if i in self.specs] or list(self.specs.values())

        best_spec, best_r = None, -1e12
        for spec in candidates:
            r = self.shadow.replay(state, spec, recent)
            if r > best_r:
                best_r, best_spec = r, spec

        current_reward = self.shadow.replay(state, self._current_spec(state), recent)
        if best_spec and best_r > (self.promote_margin * current_reward):
            state.inject_policy(**best_spec.fns())
            self.memory.remember(ctx, best_spec, best_r)
            self.last_promotion = {"id": best_spec.id, "reward": best_r, "ctx": ctx, "ts": _now_iso()}
            state._log_event("policy_promoted", {"id": best_spec.id, "reward": best_r})
            return best_spec.id
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

    def __init__(self,
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
                 divergence_fn: Optional[Callable[["EpistemicState", float], float]] = None):
        # Core state
        self.V: Union[float, Vector] = float(V0) if isinstance(V0, (int, float)) else np.asarray(V0, dtype=float)
        self.E: float = float(E0)
        self.Θ: float = float(threshold)
        self.k: float = float(realign_strength)
        self.decay_rate: float = float(_clip(decay_rate, 0.0, 1.0))
        self.step_cap: float = float(max(1e-6, step_cap))
        self.epsilon: float = float(max(1e-9, epsilon))  # small constant used in tanh-based realign

        # Policy hooks
        self._threshold_fn: Optional[Callable[["EpistemicState"], float]] = None
        self._realign_fn: Optional[Callable[["EpistemicState", float, float], float]] = None
        self._collapse_fn: Optional[Callable[["EpistemicState", Optional[float]], Tuple[float, float]]] = None
        self._context_fn: Optional[Callable[..., Any]] = None
        self._divergence_fn: Callable[["EpistemicState", float], float] = divergence_fn or sbar_directional

        # Satellites
        self.perception: Optional[Perception] = perception
        self.policy_manager: Optional[PolicyManager] = policy_manager

        # Runtime/logging
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

        # Determinism
        if rng_seed is not None:
            np.random.seed(int(rng_seed))
            self._log_event("rng_seeded", {"seed": int(rng_seed)})

        # Derived metrics (rolling)
        self._rolling_ruptures: List[int] = []
        self._rolling_drift: List[float] = []

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
          6) Meta-policy manager (if present) may evaluate and promote periodically
        """
        # 1) Resolve R
        R_is_dict = isinstance(R, dict)
        R_vec: Optional[Vector] = None

        if R_is_dict:
            if not self.perception:
                raise ValueError("Dict evidence requires a Perception adapter.")
            R_vec = self.perception.process(R)  # normalized vector
        elif isinstance(R, (list, np.ndarray)):
            R_vec = np.asarray(R, dtype=float)
        elif isinstance(R, (int, float, np.number)):
            R_val_scalar = float(R)
        else:
            raise ValueError("Unsupported evidence type for R.")

        V_is_scalar = isinstance(self.V, (int, float, np.number))

        # Align types for computation
        if R_vec is not None and V_is_scalar:
            # Promote V to vector of same shape
            self.V = np.zeros_like(R_vec, dtype=float)
            V_is_scalar = False

        # 2) Compute ∆ and scalar R_val for policy signatures
        if not V_is_scalar:
            assert R_vec is not None, "Vector V requires vector R."
            V_vec = np.asarray(self.V, dtype=float)
            delta_vec = R_vec - V_vec
            delta_mag = float(np.linalg.norm(delta_vec))
            R_val = float(np.linalg.norm(R_vec))  # scalar proxy for policy signatures
        else:
            R_val = float(R_val_scalar)  # type: ignore
            delta_mag = abs(R_val - float(self.V))

        # 3) Divergence S̄ (projected directional deviation)
        try:
            S_bar = float(self._divergence_fn(self, R_val))
        except Exception as e:
            warnings.warn(f"Divergence function failed ({e}); using directional proxy.")
            S_bar = float((R_val - (float(self.V) if V_is_scalar else float(np.linalg.norm(self.V)))))  # fallback

        # 4) Evaluate threshold & rupture
        threshold_val = self._threshold_fn(self) if self._threshold_fn else self.Θ
        ruptured = bool(delta_mag > threshold_val)
        self._last_symbol = "⚠" if ruptured else "⊙"

        if ruptured:
            # Collapse policy
            if callable(self._collapse_fn):
                try:
                    Vp, Ep = self._collapse_fn(self, R_val)
                except Exception as e:
                    raise RuntimeError(f"Collapse function failed: {e}")
            else:
                Vp, Ep = collapse_reset(self, R_val)
            # Update V, E (scalar or vector paths)
            if isinstance(self.V, np.ndarray):
                # When vector: adopt directionally toward R if collapse_adopt_R; else scale to zero
                if self._collapse_fn is collapse_adopt_R and R_vec is not None:
                    self.V = np.asarray(R_vec, dtype=float)
                    self.E = 0.0
                else:
                    self.V = np.zeros_like(self.V)
                    self.E = Ep
            else:
                self.V = Vp
                self.E = Ep

            self._rupture_count += 1
            self.meta_ruptures.append({
                "time": self._time,
                "rupture_pressure": float(delta_mag - threshold_val),
                "source": source,
                "S̄": float(S_bar)
            })
            self._trigger("on_rupture")
            # For explainability: mark time_to_realign = 0 at rupture step
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
                    self.V = realign_linear(self, R_val, delta_mag)
            else:
                # Vector realignment: move V toward R with bounded step
                V_vec = np.asarray(self.V, dtype=float)
                step = self.k * (R_vec - V_vec) * (1.0 + self.E)  # type: ignore
                # Cap step by norm
                norm_step = _l2_norm(step)
                if norm_step > self.step_cap:
                    step = step * (self.step_cap / norm_step)
                self.V = (V_vec + step).astype(float)

            # Misalignment memory update
            self.E += 0.1 * delta_mag
            self.E *= self.decay_rate
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
                "ruptured": ruptured,
                "symbol": self._last_symbol,
                "source": source,
                "time_to_realign": float(time_to_realign),
            }
            self.history.append(log_entry)
            self._rolling_ruptures.append(1 if ruptured else 0)
            self._rolling_drift.append(float(delta_mag))
            if len(self._rolling_ruptures) > 256: self._rolling_ruptures.pop(0)
            if len(self._rolling_drift) > 256: self._rolling_drift.pop(0)

        # 6) Meta-policy manager tick (every N steps; requires recent window)
        self._time += 1
        if self.policy_manager and len(self.history) >= 12 and (self._time % 10 == 0):
            recent_evidence = [h["R"] for h in self.history[-20:]]
            ctx = {
                "t": self._time,
                "E": float(self.E),
                "rupt_rate_20": float(np.mean(self._rolling_ruptures[-20:])) if len(self._rolling_ruptures) >= 1 else 0.0,
                "mean_drift_20": float(np.mean(self._rolling_drift[-20:])) if len(self._rolling_drift) >= 1 else 0.0,
            }
            try:
                self.policy_manager.maybe_adjust(self, ctx, recent_evidence)
            except Exception as e:
                self._log_event("policy_manager_error", {"error": str(e)})

            # try bounded evolution if the manager supports it (0.2.0 feature)
            if hasattr(self.policy_manager, "evolve_if_due"):
                try:
                    self.policy_manager.evolve_if_due(self, ctx, recent_evidence)
                except Exception as e:
                    self._log_event("policy_evolve_error", {"error": str(e)})

    # 0) auto evolution enabling
    def enable_auto_evolution(self,
                              param_space: Dict[str, Dict[str, Any]],
                              every: int = 30,
                              rate: float = 1.0,
                              margin: float = 1.02) -> None:
        """
        Wire the state's PolicyManager for bounded runtime evolution.

        param_space: {"policy_id": {"param": ParamRange or (low, high, sigma)}}
                     (The concrete ParamRange class lives in meta_learning.py.)
        every:       attempt evolution every N steps
        rate:        mutation intensity multiplier
        margin:      candidate must beat baseline by this factor to be promoted
        """
        if not self.policy_manager:
            raise ValueError("Auto-evolution requires a PolicyManager on this state.")
        # allow both ParamRange objects and tuples
        if hasattr(self.policy_manager, "set_param_space"):
            # normalize tuples into ParamRange if needed
            try:
                from .meta_learning import ParamRange  # type: ignore
                norm_space: Dict[str, Dict[str, Any]] = {}
                for pid, params in param_space.items():
                    norm_space[pid] = {}
                    for k, v in params.items():
                        if isinstance(v, tuple) and len(v) in (2, 3):
                            low, high, *rest = v
                            sigma = rest[0] if rest else 0.05
                            norm_space[pid][k] = ParamRange(float(low), float(high), float(sigma))
                        else:
                            norm_space[pid][k] = v
            except Exception:
                norm_space = param_space
            self.policy_manager.set_param_space(norm_space)

        # cadence & thresholds (only if manager exposes these knobs)
        for attr, val in (("evolve_every", int(max(5, every))),
                          ("evolve_rate", float(rate)),
                          ("evolve_margin", float(max(1.0, margin)))):
            if hasattr(self.policy_manager, attr):
                setattr(self.policy_manager, attr, val)

        self._log_event("auto_evolution_enabled", {
            "every": int(every), "rate": float(rate), "margin": float(margin)
        })

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
        self.V = 0.0 if isinstance(self.V, (int, float, np.number)) else np.zeros_like(self.V)
        self.E = 0.0
        self._rupture_count = 0
        self._time = 0
        self._last_symbol = "∅"
        self.history.clear()
        self.meta_ruptures.clear()
        self.event_log.clear()
        self._rolling_ruptures.clear()
        self._rolling_drift.clear()
        self._log_event("reset", {})

    def realign(self, R: Evidence) -> None:
        R_val = self._resolve_reality(R)
        # Manual adopt-R realign (safe)
        if isinstance(self.V, np.ndarray):
            # Construct a vector of appropriate norm along previous direction if R is scalar
            if isinstance(R_val, float) and not isinstance(R, (list, np.ndarray)):
                # keep direction; set magnitude to R_val
                v = np.asarray(self.V, dtype=float)
                n = _l2_norm(v)
                self.V = (v / n) * R_val
            else:
                self.V = np.asarray(R, dtype=float)
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
                "ruptured": False,
                "symbol": "⊙",
                "source": "manual_realign"
            })
        self._log_event("manual_realign", {"aligned_to": _sanitize(R)})
        self._time += 1

    def inject_policy(self,
                      threshold: Optional[Callable[["EpistemicState"], float]] = None,
                      realign: Optional[Callable[["EpistemicState", float, float], float]] = None,
                      collapse: Optional[Callable[["EpistemicState", Optional[float]], Tuple[float, float]]] = None) -> None:
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
            "collapse": bool(collapse)
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
            "identity": self.identity
        }

    def last(self) -> Optional[Dict[str, Any]]:
        return self.history[-1] if self.history else None

    def log(self) -> List[Dict[str, Any]]:
        return self.history

    def rupture_log(self) -> List[Dict[str, Any]]:
        return self.meta_ruptures

    def drift_stats(self, window: int = 10) -> Dict[str, float]:
        deltas = [step['∆'] for step in self.history[-int(window):] if '∆' in step]
        if not deltas:
            return {}
        arr = np.array(deltas, dtype=float)
        return {
            "mean_drift": float(arr.mean()),
            "std_drift": float(arr.std()),
            "max_drift": float(arr.max()),
            "min_drift": float(arr.min())
        }

    def explain_last(self) -> str:
        last = self.last()
        if not last:
            return "No steps yet."
        if last['ruptured']:
            return f"Rupture (Θ) triggered: drift {last['∆']:.3f} > threshold {last['Θ']:.3f}; S̄={last.get('S̄', 0):.3f}"
        else:
            return f"Realigned (⊙): drift {last['∆']:.3f} within threshold {last['Θ']:.3f}; S̄={last.get('S̄', 0):.3f}"

    # -----------------------
    # I/O
    # -----------------------

    def export_json(self, path: str) -> None:
        try:
            safe_history = [{k: _sanitize(v) for k, v in entry.items()} for entry in self.history]
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(safe_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to export JSON log to {path}: {e}")

    def export_csv(self, path: str) -> None:
        if not self.history:
            return
        try:
            keys = sorted(set().union(*(entry.keys() for entry in self.history)))
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.history:
                    safe_row = {k: _sanitize(row.get(k, '')) for k in keys}
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
            "details": sanitized
        })

    def _resolve_reality(self, R: Evidence) -> float:
        # For manual realign path and policy signatures needing scalar proxy
        if isinstance(R, (int, float, np.number)):
            return float(R)
        elif isinstance(R, list):
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
    PolicySpec("conservative",
               threshold_fn=threshold_static,
               realign_fn=realign_linear,
               collapse_fn=collapse_soft_decay,
               params={"k": 0.2, "note": "slow linear realign; soft decay on Θ"}),
    PolicySpec("cautious",
               threshold_fn=threshold_adaptive,
               realign_fn=realign_tanh,
               collapse_fn=collapse_soft_decay,
               params={"k": 0.15, "a": 0.05, "note": "adaptive Θ and bounded ⊙"}),
    PolicySpec("adoptive",
               threshold_fn=threshold_combined,
               realign_fn=realign_decay_adaptive,
               collapse_fn=collapse_adopt_R,
               params={"k": 0.25, "a": 0.05, "sigma": 0.01, "note": "adopt-R under Θ"}),
]
