# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.policies
================
Safe, ready-to-use policy functions for `EpistemicState` plus presets.

Signatures
----------
- Threshold:  fn(state) -> float
- Realign:    fn(state, R_val: float, delta: float) -> float          # scalar V path
- Collapse:   fn(state, R_val: float | None) -> (V_new: float, E_new: float)

Notes
-----
- Vector states are handled in `EpistemicState`; realign/collapse here target
  the scalar path. Vector steps are bounded directionally in the kernel.
- All functions enforce conservative bounds to prevent runaway behavior.
"""

from __future__ import annotations
from typing import Dict, Callable, Tuple, Optional, Any, List
import numpy as np

# ----------------------------------------------------------------------
# Utilities (keep behavior deterministic and bounded across the library)
# ----------------------------------------------------------------------

def _rng(state):
    """Prefer state's RNG for determinism; otherwise a fresh generator."""
    return getattr(state, "rng", np.random.default_rng())

def _cap(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))

def _nz_cap(x: float, floor: float = 1e-6) -> float:
    return float(max(float(x), float(floor)))

def _signed_step(v: float, r: float, step: float, cap: float) -> float:
    """Apply a signed, bounded step from v toward r."""
    sign = 1.0 if (float(r) - float(v)) >= 0.0 else -1.0
    step = _cap(step, -_nz_cap(cap), _nz_cap(cap))
    return float(v) + sign * float(step)

# ----------------------------------------------------------------------
# Threshold (Θ) policies
# ----------------------------------------------------------------------

def threshold_static(state) -> float:
    """Fixed threshold: return current Θ unchanged."""
    return float(state.Θ)

def threshold_adaptive(state, a: float = 0.05, cap: float = 1.0) -> float:
    """
    Adaptive threshold: Θ_t = Θ + a·E, bounded by `cap`.
    More memory (E) → more conservative threshold.
    """
    a   = _cap(a,   0.0, 1.0)
    cap = _cap(cap, 0.0, 10.0)
    return float(state.Θ + _cap(a * float(state.E), 0.0, cap))

def threshold_stochastic(state, sigma: float = 0.01) -> float:
    """Add small Gaussian noise to Θ (deterministic via state's RNG)."""
    sigma = _cap(sigma, 0.0, 0.2)
    return float(state.Θ + float(_rng(state).normal(0.0, sigma)))

def threshold_combined(state, a: float = 0.05, sigma: float = 0.01, cap: float = 1.0) -> float:
    """Adaptive + stochastic threshold, both bounded."""
    a     = _cap(a,     0.0, 1.0)
    sigma = _cap(sigma, 0.0, 0.2)
    adapt = _cap(a * float(state.E), 0.0, cap)
    return float(state.Θ + adapt + float(_rng(state).normal(0.0, sigma)))

def threshold_hysteresis(state, band: float = 0.10, refractory: int = 5) -> float:
    """
    Hysteresis threshold with a short refractory after rupture.
    - When recently ruptured, raise Θ temporarily to reduce flip-flop.
    - `band` increases Θ by a fraction of current Θ.
    - `refractory` (steps) uses state's `_since_last_rupture` if available.
    """
    base = float(state.Θ)
    band = _cap(band, 0.0, 1.0)
    ref  = int(max(0, refractory))
    since = getattr(state, "_since_last_rupture", None)
    if since is not None and since <= ref:
        return float(base * (1.0 + band))
    return base

# ----------------------------------------------------------------------
# Realign (⊙) policies — scalar path
# ----------------------------------------------------------------------

def realign_linear(state, R_val: float, delta: float) -> float:
    """Linear step toward R. Larger delta and memory E increase step size."""
    k    = _cap(getattr(state, "k", 0.3), 0.0, 2.0)
    step = k * float(delta) * (1.0 + float(state.E))
    cap  = _nz_cap(getattr(state, "step_cap", 1.0))
    return _signed_step(state.V, R_val, step, cap)

def realign_tanh(state, R_val: float, delta: float) -> float:
    """Tanh-bounded step; saturates for large deltas. E enters via tanh(E/eps)."""
    k    = _cap(getattr(state, "k", 0.3), 0.0, 2.0)
    eps  = _nz_cap(getattr(state, "epsilon", 1e-3), 1e-9)
    gain = float(np.tanh(k * float(delta))) * (1.0 + float(np.tanh(float(state.E) / eps)))
    cap  = _nz_cap(getattr(state, "step_cap", 1.0))
    return _signed_step(state.V, R_val, gain, cap)

def realign_bounded(state, R_val: float, delta: float, cap: float = 1.0) -> float:
    """Hard cap on per-step shift, independent of kernel cap (use tighter of the two)."""
    k     = _cap(getattr(state, "k", 0.3), 0.0, 2.0)
    step  = k * float(delta) * (1.0 + float(state.E))
    cap   = _nz_cap(min(float(cap), float(getattr(state, "step_cap", 1.0))))
    return _signed_step(state.V, R_val, step, cap)

def realign_decay_adaptive(state, R_val: float, delta: float) -> float:
    """Damp gain when memory E is high: k' = k / (1 + E)."""
    k    = _cap(getattr(state, "k", 0.3) / (1.0 + float(state.E)), 0.0, 1.0)
    step = k * float(delta)
    cap  = _nz_cap(getattr(state, "step_cap", 1.0))
    return _signed_step(state.V, R_val, step, cap)

def realign_antioscillatory(state, R_val: float, delta: float, alpha: float = 0.8) -> float:
    """
    Anti-oscillation step: blend current step with momentum damp.
    Uses last ∆ sign from history (if present) to reduce ping-pong near Θ.
    """
    alpha = _cap(alpha, 0.0, 1.0)
    k     = _cap(getattr(state, "k", 0.3), 0.0, 2.0)
    base  = k * float(delta) * (1.0 + float(state.E))

    # Estimate prior drift sign to damp reversals.
    hist = getattr(state, "history", None)
    if hist and len(hist) >= 2:
        prev = hist[-2]
        prev_d = float(prev.get("∆", 0.0))
        going_up = (R_val - float(state.V)) >= 0.0
        if prev_d > 0 and not going_up:
            base *= (1.0 - 0.5 * alpha)
        elif prev_d <= 0 and going_up:
            base *= (1.0 - 0.5 * alpha)

    cap = _nz_cap(getattr(state, "step_cap", 1.0))
    return _signed_step(state.V, R_val, base, cap)

# ----------------------------------------------------------------------
# Collapse (post-Θ) policies — scalar path
# ----------------------------------------------------------------------

def collapse_reset(state, R_val: Optional[float] = None) -> Tuple[float, float]:
    """Hard reset to zero; memory cleared (E→0)."""
    return 0.0, 0.0

def collapse_soft_decay(state, R_val: Optional[float] = None,
                        gamma: float = 0.5, beta: float = 0.3) -> Tuple[float, float]:
    """Conservative softening: V' = γ·V, E' = β·E (0 ≤ γ,β ≤ 1)."""
    gamma = _cap(gamma, 0.0, 1.0)
    beta  = _cap(beta,  0.0, 1.0)
    return float(state.V) * gamma, float(state.E) * beta

def collapse_adopt_R(state, R_val: Optional[float] = None) -> Tuple[float, float]:
    """Adopt incoming scalar R as new projection; memory cleared."""
    rv = float(R_val if R_val is not None else state.V)
    return rv, 0.0

def collapse_adopt_inertia(state, R_val: Optional[float] = None,
                           eta: float = 0.5, beta: float = 0.2) -> Tuple[float, float]:
    """Partial adoption with inertia: V' = (1-η)·V + η·R, E' = β·E."""
    eta  = _cap(eta,  0.0, 1.0)
    beta = _cap(beta, 0.0, 1.0)
    rv   = float(R_val if R_val is not None else state.V)
    vnew = (1.0 - eta) * float(state.V) + eta * rv
    enew = beta * float(state.E)
    return float(vnew), float(enew)

def collapse_randomized(state, R_val: Optional[float] = None, sigma: float = 0.1) -> Tuple[float, float]:
    """Jump near zero with small noise; memory cleared. Deterministic via state's RNG."""
    sigma = _cap(sigma, 0.0, 1.0)
    return float(_rng(state).normal(0.0, sigma)), 0.0

# ----------------------------------------------------------------------
# Kernel-compatible wrappers (for EpistemicState.inject_policy)
# ----------------------------------------------------------------------

threshold_static_fn       = lambda state: threshold_static(state)
threshold_adaptive_fn     = lambda state: threshold_adaptive(state)
threshold_stochastic_fn   = lambda state: threshold_stochastic(state)
threshold_combined_fn     = lambda state: threshold_combined(state)
threshold_hysteresis_fn   = lambda state: threshold_hysteresis(state)

realign_linear_fn            = lambda state, R, d: realign_linear(state, float(R), float(d))
realign_tanh_fn              = lambda state, R, d: realign_tanh(state, float(R), float(d))
realign_bounded_fn           = lambda state, R, d: realign_bounded(state, float(R), float(d))
realign_decay_adaptive_fn    = lambda state, R, d: realign_decay_adaptive(state, float(R), float(d))
realign_antioscillatory_fn   = lambda state, R, d: realign_antioscillatory(state, float(R), float(d))

collapse_reset_fn         = lambda state, R=None: collapse_reset(state, R)
collapse_soft_decay_fn    = lambda state, R=None: collapse_soft_decay(state, R)
collapse_adopt_R_fn       = lambda state, R=None: collapse_adopt_R(state, R)
collapse_adopt_inertia_fn = lambda state, R=None: collapse_adopt_inertia(state, R)
collapse_randomized_fn    = lambda state, R=None: collapse_randomized(state, R)

# ----------------------------------------------------------------------
# Registry (nested + flat for ergonomic lookups)
# ----------------------------------------------------------------------

POLICY_REGISTRY_NESTED: Dict[str, Dict[str, Callable[..., Any]]] = {
    "threshold": {
        "static":           threshold_static,
        "adaptive":         threshold_adaptive,
        "stochastic":       threshold_stochastic,
        "combined":         threshold_combined,
        "hysteresis":       threshold_hysteresis,
    },
    "realign": {
        "linear":           realign_linear,
        "tanh":             realign_tanh,
        "bounded":          realign_bounded,
        "decay_adaptive":   realign_decay_adaptive,
        "antiosc":          realign_antioscillatory,
    },
    "collapse": {
        "reset":            collapse_reset,
        "soft_decay":       collapse_soft_decay,
        "adopt_R":          collapse_adopt_R,
        "adopt_inertia":    collapse_adopt_inertia,
        "randomized":       collapse_randomized,
    },
}

# Flat aliases for convenience
POLICY_REGISTRY_FLAT: Dict[str, Callable[..., Any]] = {
    # threshold
    "threshold_static":         threshold_static,
    "threshold_adaptive":       threshold_adaptive,
    "threshold_stochastic":     threshold_stochastic,
    "threshold_combined":       threshold_combined,
    "threshold_hysteresis":     threshold_hysteresis,
    # realign
    "realign_linear":           realign_linear,
    "realign_tanh":             realign_tanh,
    "realign_bounded":          realign_bounded,
    "realign_decay_adaptive":   realign_decay_adaptive,
    "realign_antioscillatory":  realign_antioscillatory,
    # collapse
    "collapse_reset":           collapse_reset,
    "collapse_soft_decay":      collapse_soft_decay,
    "collapse_adopt_R":         collapse_adopt_R,
    "collapse_adopt_inertia":   collapse_adopt_inertia,
    "collapse_randomized":      collapse_randomized,
}

# Unified view: nested + flat
POLICY_REGISTRY: Dict[str, Any] = {**POLICY_REGISTRY_FLAT, **POLICY_REGISTRY_NESTED}
REGISTRY = POLICY_REGISTRY  # back-compat

# ----------------------------------------------------------------------
# Safe preset PolicySpecs (plug-and-play for PolicyManager)
# ----------------------------------------------------------------------

try:
    # Soft import to avoid hard coupling for simple users
    from cognize.epistemic import PolicySpec  # type: ignore
except Exception:
    PolicySpec = None  # type: ignore

def build_safe_specs(
    *,
    conservative_k: float = 0.20,
    cautious_k: float = 0.15,
    adoptive_k: float = 0.25,
    theta_base: float = 0.35,
    step_cap: float = 1.0,
) -> List["PolicySpec"]:
    """
    Create a standard set of safe PolicySpec presets.
    Returns [] if PolicySpec is unavailable.

    Parameter names map to EpistemicState attributes (k, Θ, step_cap, ...)
    or to policy kwargs (a, sigma, cap, gamma, beta). The kernel's
    PolicyManager will apply state params and signature-bound policy kwargs.
    """
    if PolicySpec is None:
        return []

    conservative = PolicySpec(
        id="conservative",
        threshold_fn=threshold_static,
        realign_fn=realign_linear,
        collapse_fn=collapse_soft_decay,
        params={
            "k": float(np.clip(conservative_k, 0.05, 0.5)),
            "Θ": float(theta_base),
            "step_cap": float(step_cap),
            "gamma": 0.6,  # collapse_soft_decay
            "beta": 0.4,
            "note": "slow linear realign; favors stability; soft decay on Θ",
        },
    )

    cautious = PolicySpec(
        id="cautious",
        threshold_fn=threshold_adaptive,
        realign_fn=realign_tanh,
        collapse_fn=collapse_soft_decay,
        params={
            "k": float(np.clip(cautious_k, 0.05, 0.5)),
            "Θ": float(theta_base),
            "step_cap": float(step_cap),
            "a": 0.05,      # threshold_adaptive
            "cap": 1.0,     # threshold_adaptive cap
            "gamma": 0.55,  # collapse_soft_decay
            "beta": 0.35,
            "note": "adaptive Θ and bounded ⊙; balanced drift handling",
        },
    )

    adoptive = PolicySpec(
        id="adoptive",
        threshold_fn=threshold_combined,
        realign_fn=realign_decay_adaptive,
        collapse_fn=collapse_adopt_R,
        params={
            "k": float(np.clip(adoptive_k, 0.1, 0.6)),
            "Θ": float(theta_base),
            "step_cap": float(step_cap),
            "a": 0.05,      # threshold_combined
            "sigma": 0.01,  # threshold_combined
            "note": "embraces regime shifts under Θ by adopting R; E-damped gain",
        },
    )

    turbulent = PolicySpec(
        id="turbulent",
        threshold_fn=threshold_hysteresis,     # slightly more conservative after Θ
        realign_fn=realign_bounded,
        collapse_fn=collapse_randomized,
        params={
            "k": float(np.clip(adoptive_k, 0.15, 0.8)),
            "Θ": float(theta_base),
            "step_cap": float(step_cap),
            "band": 0.10,   # threshold_hysteresis
            "refractory": 5,
            "cap": 0.6,     # realign_bounded cap
            "note": "exploratory; caps ⊙ and uses hysteresis to avoid flip-flop",
        },
    )

    return [conservative, cautious, adoptive, turbulent]

# Ready-to-use presets
SAFE_SPECS: List["PolicySpec"] = build_safe_specs()

# ----------------------------------------------------------------------
# Legacy helpers (for third-party code; forward to the safe versions)
# ----------------------------------------------------------------------

def _collapse_reset_legacy(R, V, E):                                     return 0.0, 0.0
def _collapse_soft_decay_legacy(R, V, E, gamma=0.5, beta=0.3):            return float(V) * float(gamma), float(E) * float(beta)
def _collapse_adopt_R_legacy(R, V, E):                                    return float(R), 0.0
def _collapse_randomized_legacy(R, V, E, sigma=0.1):                      return float(np.random.normal(0.0, float(sigma))), 0.0

def _realign_linear_legacy(V, delta, E, k):           return float(V) + float(k) * float(delta) * (1.0 + float(E))
def _realign_tanh_legacy(V, delta, E, k):             return float(V) + float(np.tanh(float(k) * float(delta))) * (1.0 + float(E))
def _realign_bounded_legacy(V, delta, E, k, cap=1.):  return float(V) + float(min(float(k) * float(delta) * (1.0 + float(E)), float(cap)))
def _realign_decay_adaptive_legacy(V, delta, E, k):   return float(V) + (float(k) / (1.0 + float(E))) * float(delta)

def _threshold_static_legacy(E, t, base=0.35):                 return float(base)
def _threshold_adaptive_legacy(E, t, base=0.35, a=0.05):       return float(base + float(a) * float(E))
def _threshold_stochastic_legacy(E, t, base=0.35, sigma=0.02): return float(base + float(np.random.normal(0.0, float(sigma))))
def _threshold_combined_legacy(E, t, base=0.35, a=0.05, sigma=0.01):
    return float(base + float(a) * float(E) + float(np.random.normal(0.0, float(sigma))))

__all__ = [
    # thresholds
    "threshold_static", "threshold_adaptive", "threshold_stochastic",
    "threshold_combined", "threshold_hysteresis",
    # realign
    "realign_linear", "realign_tanh", "realign_bounded",
    "realign_decay_adaptive", "realign_antioscillatory",
    # collapse
    "collapse_reset", "collapse_soft_decay", "collapse_adopt_R",
    "collapse_adopt_inertia", "collapse_randomized",
    # registries
    "POLICY_REGISTRY", "POLICY_REGISTRY_NESTED", "POLICY_REGISTRY_FLAT", "REGISTRY",
    # injectable wrappers
    "threshold_static_fn", "threshold_adaptive_fn", "threshold_stochastic_fn",
    "threshold_combined_fn", "threshold_hysteresis_fn",
    "realign_linear_fn", "realign_tanh_fn", "realign_bounded_fn",
    "realign_decay_adaptive_fn", "realign_antioscillatory_fn",
    "collapse_reset_fn", "collapse_soft_decay_fn", "collapse_adopt_R_fn",
    "collapse_adopt_inertia_fn", "collapse_randomized_fn",
    # presets
    "build_safe_specs", "SAFE_SPECS",
]
