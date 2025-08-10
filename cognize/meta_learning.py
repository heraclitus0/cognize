# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
Cognize Meta-Learning (Runtime Policy Adaptation)
=================================================

Selects (and can evolve) epistemic policies at runtime via shadow evaluation.
Safeguards: ε-greedy, safety flags, promotion margins, and bounded parameter evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple
import copy
import json
import time
import numpy as np

# Reuses kernel's PolicySpec to avoid duplication.
from .epistemic import PolicySpec


# ---------------------------------------------------------------------
# Parameter ranges for bounded evolution
# ---------------------------------------------------------------------

@dataclass
class ParamRange:
    low: float
    high: float
    sigma: float = 0.05  # relative mutation noise scale (fraction of range)

    def clamp(self, x: float) -> float:
        return float(min(max(x, self.low), self.high))


class ParamSpace:
    """
    Mapping: policy_id -> param_name -> ParamRange
    Example:
      {
        "cautious": {"k": ParamRange(0.05, 0.4), "a": ParamRange(0.01, 0.2)}
      }
    """
    def __init__(self, space: Optional[Dict[str, Dict[str, ParamRange]]] = None):
        self._space: Dict[str, Dict[str, ParamRange]] = space or {}

    def set(self, space: Dict[str, Dict[str, ParamRange]]) -> None:
        self._space = space

    def bounds_for(self, policy_id: str) -> Dict[str, ParamRange]:
        return self._space.get(policy_id, {})

    def is_empty(self) -> bool:
        return not any(self._space.values())


def _mutate_params(params: Dict[str, Any],
                   bounds: Dict[str, ParamRange],
                   rate: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """
    Gaussian mutation within bounds for numeric params only.
    Non-numeric params are copied through unchanged.
    """
    rng = rng or np.random.default_rng()
    out: Dict[str, Any] = dict(params)
    for k, pr in bounds.items():
        base = float(params.get(k, (pr.low + pr.high) * 0.5))
        span = (pr.high - pr.low)
        noise = rng.normal(loc=0.0, scale=max(1e-9, pr.sigma)) * span * float(rate)
        out[k] = pr.clamp(base + noise)
    return out


# ---------------------------------------------------------------------
# PolicyMemory
# ---------------------------------------------------------------------

class PolicyMemory:
    """
    Rolling memory of (context bucket, policy id, params, reward).
    Provides top-k lookup; supports JSON persistence.
    """
    def __init__(self, cap: int = 5000):
        self.records: List[Dict[str, Any]] = []
        self.cap = int(cap)

    @staticmethod
    def _sanitize(v: Any) -> Any:
        import numpy as _np
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, dict):
            return {k: PolicyMemory._sanitize(val) for k, val in v.items()}
        if isinstance(v, _np.integer):
            return int(v)
        if isinstance(v, _np.floating):
            return float(v)
        if isinstance(v, _np.ndarray):
            return _np.asarray(v).tolist()
        return str(v)

    def remember(self, ctx_bucket: str, spec: PolicySpec, reward: float) -> None:
        rec = {
            "ctx_bucket": ctx_bucket,
            "id": spec.id,
            "params": {k: self._sanitize(v) for k, v in getattr(spec, "params", {}).items()},
            "reward": float(reward),
            "ts": time.time(),
        }
        self.records.append(rec)
        if len(self.records) > self.cap:
            self.records = self.records[-self.cap:]

    def top_ids(self, ctx_bucket: str, k: int = 3) -> List[str]:
        """Return top-k policy ids for this context bucket; fallback to global."""
        if not self.records:
            return []
        filt = [r for r in self.records if r["ctx_bucket"] == ctx_bucket]
        source = filt if filt else self.records
        ordered = sorted(source, key=lambda r: r["reward"], reverse=True)
        seen, out = set(), []
        for r in ordered:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r["id"])
            if len(out) >= k:
                break
        return out

    # Persistence helpers used by EpistemicState.save/load helpers
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        except FileNotFoundError:
            self.records = []

    # Optional inspection
    def to_json(self) -> str:
        return json.dumps(self.records, indent=2)

    def dump_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Reward & Safety utilities
# ---------------------------------------------------------------------

def default_reward(window: List[Dict[str, Any]]) -> float:
    """
    Reward = -(rupture_rate) - 0.2*mean_drift - 0.05*time_to_realign
    Larger is better (closer to 0).
    """
    if not window:
        return -1e9
    import numpy as _np
    drift = _np.array([w.get("∆", 0.0) for w in window], dtype=float)
    rupt = _np.array([1.0 if w.get("ruptured", False) else 0.0 for w in window], dtype=float).mean()
    ttr  = _np.array([w.get("time_to_realign", 1.0) for w in window], dtype=float).mean()
    return float((-1.0 * rupt) + (-0.2 * drift.mean()) + (-0.05 * ttr))


def safety_flags(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple online safety checks:
      - rupture_storm: rupture rate above 0.5 over window
      - oscillation: alternating rupture pattern
      - runaway_E: E grows monotonically (if E logged)
    """
    import numpy as _np
    if not window:
        return {"rupture_storm": False, "oscillation": False, "runaway_E": False, "count": 0}

    rupt_seq = _np.array([1 if w.get("ruptured", False) else 0 for w in window], dtype=int)
    storm = rupt_seq.mean() > 0.5

    # Oscillation heuristic: many flips across window
    changes = _np.abs(_np.diff(rupt_seq)).sum()
    osc = changes >= max(4, len(rupt_seq) // 3)

    # Runaway E if present
    Es = _np.array([w.get("E", None) for w in window if "E" in w], dtype=float)
    runaway = False
    if Es.size >= 4:
        diffs = _np.diff(Es)
        runaway = _np.all(diffs > 0.0) and Es[-1] > Es[0] * 1.2

    return {"rupture_storm": bool(storm), "oscillation": bool(osc), "runaway_E": bool(runaway), "count": len(window)}


# ---------------------------------------------------------------------
# ShadowRunner
# ---------------------------------------------------------------------

class ShadowRunner:
    """
    Clone state, inject candidate policy, replay evidence, return (reward, safety_flags).
    """
    def __init__(self,
                 reward_fn: Callable[[List[Dict[str, Any]]], float] = default_reward,
                 min_window: int = 10,
                 max_window: int = 50):
        self.reward_fn = reward_fn
        self.min_window = int(min_window)
        self.max_window = int(max_window)

    def replay(self, state, spec: PolicySpec, evidence_seq: List[Any]) -> Tuple[float, Dict[str, Any]]:
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
        r = self.reward_fn(window)
        flags = safety_flags(window)
        return float(r), flags


# ---------------------------------------------------------------------
# PolicyManager (ε-greedy + contextual top-k + bounded evolution)
# ---------------------------------------------------------------------

class PolicyManager:
    """
    ε-greedy + contextual top-k promotion with safety gating.
    Optional param evolution: mutate best-in-bucket within bounds and promote if safe.
    """
    def __init__(self,
                 base_specs: List[PolicySpec],
                 memory: Optional[PolicyMemory] = None,
                 shadow: Optional[ShadowRunner] = None,
                 epsilon: float = 0.15,
                 promote_margin: float = 1.03,
                 cooldown_steps: int = 30):
        if not base_specs:
            raise ValueError("PolicyManager requires at least one PolicySpec.")
        self.specs: Dict[str, PolicySpec] = {s.id: s for s in base_specs}
        self.memory = memory or PolicyMemory()
        self.shadow = shadow or ShadowRunner()
        self.epsilon = float(max(0.0, min(1.0, epsilon)))
        self.promote_margin = float(max(1.0, promote_margin))
        self.cooldown_steps = int(max(0, cooldown_steps))
        self._last_promotion_step: Optional[int] = None
        self.last_promotion: Optional[Dict[str, Any]] = None

        # Evolution controls
        self.param_space: ParamSpace = ParamSpace()
        self.evolve_every: int = 30
        self.evolve_rate: float = 1.0
        self.evolve_margin: float = 1.02
        self._rng = np.random.default_rng()

    # --- param evolution configuration ---
    def set_param_space(self, space: Dict[str, Dict[str, ParamRange]]) -> None:
        self.param_space = ParamSpace(space)

    # ---- context bucketing ----
    @staticmethod
    def _bucket(x: float, bins: List[float]) -> str:
        for i, b in enumerate(bins):
            if x <= b:
                return f"b{i}"
        return f"b{len(bins)}"

    def context_bucket(self, ctx: Dict[str, Any]) -> str:
        """
        Map raw context into a coarse, stable bucket string.
        Expected ctx keys (optional): E, mean_drift_20, rupt_rate_20, domain, t
        """
        E = float(ctx.get("E", 0.0))
        md = float(ctx.get("mean_drift_20", 0.0))
        rr = float(ctx.get("rupt_rate_20", 0.0))
        dom = str(ctx.get("domain", "generic"))
        Eb = self._bucket(E, [0.1, 0.5, 1.0, 2.0, 5.0])
        Db = self._bucket(md, [0.05, 0.1, 0.2, 0.4, 1.0])
        Rb = self._bucket(rr, [0.05, 0.1, 0.2, 0.4, 0.8])
        return f"{dom}|E:{Eb}|D:{Db}|R:{Rb}"

    # ---- utilities ----
    def _current_spec(self, state) -> PolicySpec:
        """Snapshot the state's current policy triple for comparison."""
        th = state._threshold_fn if state._threshold_fn else (lambda s: state.Θ)
        rl = state._realign_fn if state._realign_fn else (lambda s, R, d: state.V + state.k * d)
        cp = state._collapse_fn if state._collapse_fn else (lambda s, R: (0.0, 0.0))
        return PolicySpec("__current__", th, rl, cp, params={"k": getattr(state, "k", None), "Θ": getattr(state, "Θ", None)})

    def _top_candidates(self, bucket: str, k: int = 3) -> List[PolicySpec]:
        top_ids = self.memory.top_ids(bucket, k=k)
        if top_ids:
            return [self.specs[i] for i in top_ids if i in self.specs]
        return list(self.specs.values())

    # ---- main entry: selection/promotion ----
    def maybe_adjust(self, state, ctx: Dict[str, Any], recent_evidence: List[Any]) -> Optional[str]:
        """
        Evaluate candidates and promote the best safe policy if it beats current by margin.
        Returns promoted policy id or None.
        """
        step = int(ctx.get("t", len(state.history)))
        if self._last_promotion_step is not None and (step - self._last_promotion_step) < self.cooldown_steps:
            return None

        bucket = self.context_bucket(ctx)
        explore = (np.random.random() < self.epsilon) or (not self.memory.records)
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
        unsafe = best_flags.get("rupture_storm") or best_flags.get("oscillation") or best_flags.get("runaway_E")

        if best_spec and (best_r > self.promote_margin * current_r) and not unsafe:
            state.inject_policy(**best_spec.fns())
            self.memory.remember(bucket, best_spec, best_r)
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

        # Remember baseline too (analytics)
        self.memory.remember(bucket, current_spec, current_r)
        return None

    # ---- bounded evolution (optional) ----
    def evolve_if_due(self, state, ctx: Dict[str, Any], recent_evidence: List[Any]) -> Optional[PolicySpec]:
        """
        If the cadence hits, mutate top policy in bucket within ParamSpace bounds and
        promote if it safely outperforms baseline by evolve_margin.
        """
        if self.param_space.is_empty():
            return None

        step = int(ctx.get("t", len(state.history)))
        if step % max(1, self.evolve_every) != 0:
            return None

        bucket = self.context_bucket(ctx)

        # Choose a base spec: best in bucket or first available
        base_ids = self.memory.top_ids(bucket, k=1)
        base_spec_id = base_ids[0] if base_ids else (next(iter(self.specs)) if self.specs else None)
        if base_spec_id is None:
            return None

        base_spec = self.specs[base_spec_id]
        bounds = self.param_space.bounds_for(base_spec.id)
        if not bounds:
            return None

        # Mutate parameters within bounds
        candidate_params = _mutate_params(getattr(base_spec, "params", {}), bounds, rate=self.evolve_rate, rng=self._rng)
        cand = PolicySpec(
            id=f"{base_spec.id}#mut@{step}",
            threshold_fn=base_spec.threshold_fn,
            realign_fn=base_spec.realign_fn,
            collapse_fn=base_spec.collapse_fn,
            params=candidate_params,
            notes=f"mutated from {base_spec.id}",
        )

        # Helper to temporarily map common params to state attributes
        def _apply_params_to_state(st, p):
            if "k" in p: st.k = float(p["k"])
            if "Θ" in p: st.Θ = float(p["Θ"])

        # Score candidate vs baseline, applying params to the cloned state
        s = copy.deepcopy(state)
        _apply_params_to_state(s, candidate_params)
        cand_reward, cand_flags = self.shadow.replay(s, cand, recent_evidence)

        s2 = copy.deepcopy(state)
        _apply_params_to_state(s2, getattr(base_spec, "params", {}))
        base_reward, _ = self.shadow.replay(s2, base_spec, recent_evidence)

        unsafe = cand_flags.get("rupture_storm") or cand_flags.get("oscillation") or cand_flags.get("runaway_E")
        if (cand_reward > self.evolve_margin * base_reward) and not unsafe:
            # Register and promote
            self.specs[cand.id] = cand
            state.inject_policy(**cand.fns())
            _apply_params_to_state(state, candidate_params)  # map onto live state
            self.memory.remember(bucket, cand, cand_reward)
            self.last_promotion = {
                "id": cand.id,
                "bucket": bucket,
                "reward": float(cand_reward),
                "baseline_reward": float(base_reward),
                "flags": cand_flags,
                "step": step,
            }
            if hasattr(state, "_log_event"):
                state._log_event("policy_evolved", self.last_promotion)
            return cand

        # Track baseline even if candidate loses (learning signal)
        self.memory.remember(bucket, base_spec, base_reward)
        return None
