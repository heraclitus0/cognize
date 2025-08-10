# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
Cognize Meta-Learning (Runtime Policy Adaptation)
=================================================

Purpose
-------
Selects (and later, synthesizes) the best epistemic policies at *runtime* by
evaluating candidates in a shadow run, comparing rewards, and promoting wins.
Bounded, template-based: no arbitrary code-gen.

Key Concepts
------------
- PolicySpec:    Threshold (Θ), Realign (⊙), Collapse (post-Θ) callables + params.
- PolicyMemory:  (Context bucket → policy → reward) records, with cap + JSON I/O.
- ShadowRunner:  Clones state, replays recent evidence, computes reward & safety.
- PolicyManager: ε-greedy + contextual top-k, promotion margin, cooldown, logging.

Notes
-----
- Context is intentionally small: we *bucket* it to avoid overfitting.
- Safety first: we do not promote if safety flags trigger (oscillation, storms).
- Determinism: Determined by upstream seeding of numpy in EpistemicState.

Public API (stable)
-------------------
- class PolicySpec
- class PolicyMemory
- class ShadowRunner
- class PolicyManager
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple
import numpy as np
import copy
import json
import time


# ---------------------------
# PolicySpec
# ---------------------------

@dataclass
class PolicySpec:
    """Container for a safe policy triple with metadata."""
    id: str
    threshold_fn: Callable  # fn(state) -> float
    realign_fn: Callable    # fn(state, R_val: float, delta: float) -> float
    collapse_fn: Callable   # fn(state, R_val: Optional[float]) -> (V', E')
    params: Dict[str, Any] = field(default_factory=dict)
    safety_level: str = "standard"  # "standard" | "strict" | "experimental"
    notes: str = ""

    def fns(self) -> Dict[str, Callable]:
        return {
            "threshold": self.threshold_fn,
            "realign": self.realign_fn,
            "collapse": self.collapse_fn,
        }


# ---------------------------
# PolicyMemory
# ---------------------------

class PolicyMemory:
    """
    Rolling memory of (context bucket, policy id, params, reward).
    Provides simple top-k lookup; can be JSON-dumped for audit.
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
        if isinstance(v, (dict,)):
            return {k: PolicyMemory._sanitize(val) for k, val in v.items()}
        if isinstance(v, (_np.integer,)):
            return int(v)
        if isinstance(v, (_np.floating,)):
            return float(v)
        if isinstance(v, (_np.ndarray,)):
            return _np.asarray(v).tolist()
        return str(v)

    def remember(self, ctx_bucket: str, spec: PolicySpec, reward: float) -> None:
        rec = {
            "ctx_bucket": ctx_bucket,
            "id": spec.id,
            "params": {k: self._sanitize(v) for k, v in spec.params.items()},
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
        # Filter by bucket first
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
   
    def save(self,path: str) -> None:
        with open(parh,"w",encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        try:
            with open(path,"r",encoding="utf-8") as f:
                self.records = json.load(f)
        except FileNotFoundError :
            self.records = []

    # Optional: inspection / persistence
    def to_json(self) -> str:
        return json.dumps(self.records, indent=2)

    def dump_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)


# ---------------------------
# Reward & Safety utilities
# ---------------------------

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


# ---------------------------
# ShadowRunner
# ---------------------------

class ShadowRunner:
    """
    Clones the state, injects candidate policy, replays recent evidence, and computes reward & safety.
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
        # Bound window length for cost
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


# ---------------------------
# PolicyManager
# ---------------------------

class PolicyManager:
    """
    ε-greedy + contextual top-k promotion with safety gating.
    - epsilon: exploration rate
    - promote_margin: require candidate_reward >= promote_margin * current_reward
    - cooldown_steps: prevent thrashing; skip promotion if last promotion too recent
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
    def set_param_space(self, space: Dict[str, Dict[str, ParamRange]]) -> None:
        self.param_space = ParamSpace(space)

    def evolve_if_due(self, state, ctx: Dict[str, Any], recent_evidence: List[Any]) -> Optional[PolicySpec]:
         step = int(ctx.get("t", len(state.history)))
    if step % max(1, self.evolve_every) != 0:
        return None

    bucket = self.context_bucket(ctx)
    # pick a base spec: top for bucket or any available
    base_ids = self.memory.top_ids(bucket, k=1)
    base_spec_id = base_ids[0] if base_ids else (next(iter(self.specs)) if self.specs else None)
    if base_spec_id is None:
        return None
    base_spec = self.specs[base_spec_id]
    bounds = self.param_space.bounds_for(base_spec.id)
    if not bounds:
        return None  # nothing tunable for this spec

    # mutate within bounds
    candidate_params = _mutate_params(base_spec.params, bounds, rate=self.evolve_rate)
    cand = PolicySpec(
        id=f"{base_spec.id}#mut@{step}",
        threshold_fn=base_spec.threshold_fn,
        realign_fn=base_spec.realign_fn,
        collapse_fn=base_spec.collapse_fn,
        params=candidate_params,
        safety_level="experimental",
        notes=f"mutated from {base_spec.id}"
    )

    # Temporarily apply params to state during shadow replay:
    # We assume policies read state.k, state.Θ only indirectly; params here are metadata.
    # If you want params to actually alter behavior, read them inside policy callables or map them to state before replay.
    # Minimal approach: map common params onto state via a thin wrapper.
    # Quick wrapper:
    def _apply_params_to_state(st, p):
        if "k" in p: st.k = float(p["k"])
        if "Θ" in p: st.Θ = float(p["Θ"])

    s = copy.deepcopy(state)
    _apply_params_to_state(s, candidate_params)
    cand_reward, cand_flags = self.shadow.replay(s, cand, recent_evidence)

    s2 = copy.deepcopy(state)
    _apply_params_to_state(s2, base_spec.params)
    base_reward, _ = self.shadow.replay(s2, base_spec, recent_evidence)

    unsafe = cand_flags.get("rupture_storm") or cand_flags.get("oscillation") or cand_flags.get("runaway_E")
    if (cand_reward > self.evolve_margin * base_reward) and not unsafe:
        # Register and promote
        self.specs[cand.id] = cand
        state.inject_policy(**cand.fns())
        # Also map params onto live state
        _apply_params_to_state(state, candidate_params)
        self.memory.remember(bucket, cand, cand_reward)
        self.last_promotion = {"id": cand.id, "bucket": bucket, "reward": float(cand_reward), "baseline_reward": float(base_reward), "flags": cand_flags, "step": step}
        if hasattr(state, "_log_event"):
            state._log_event("policy_evolved", self.last_promotion)
        return cand
    else:
        # Remember trial baseline for learning, even if not promoted
        self.memory.remember(bucket, base_spec, base_reward)
        return None
        self.specs: Dict[str, PolicySpec] = {s.id: s for s in base_specs}
        self.memory = memory or PolicyMemory()
        self.shadow = shadow or ShadowRunner()
        self.epsilon = float(max(0.0, min(1.0, epsilon)))
        self.promote_margin = float(max(1.0, promote_margin))
        self.cooldown_steps = int(max(0, cooldown_steps))
        self._last_promotion_step: Optional[int] = None
        self.last_promotion: Optional[Dict[str, Any]] = None
        self.param_space: ParamSpace = ParamSpace()
        self.evolve_every: int = 30
        self.evolve_rate: float = 1.0
        self.evolve_margin: float = 1.02

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
        # fallback: all specs
        return list(self.specs.values())

    # ---- main entry ----
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

        # Safety gating: never promote if candidate triggers storms/oscillation/runaway
        unsafe = best_flags.get("rupture_storm") or best_flags.get("oscillation") or best_flags.get("runaway_E")

        if best_spec and (best_r > self.promote_margin * current_r) and not unsafe:
            # Promote
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
            # Let the state know for its event log (if available)
            if hasattr(state, "_log_event"):
                state._log_event("policy_promoted", self.last_promotion)
            return best_spec.id

        # Optionally remember baseline too (for analytics)
        self.memory.remember(bucket, current_spec, current_r)
        return None
